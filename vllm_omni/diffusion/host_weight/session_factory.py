# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Composition root for pre-loader diffusion host-weight sessions."""

from __future__ import annotations

import importlib.metadata
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn
from vllm.config.load import LoadConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_fp8_supported,
)

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    MiniMaxH3Pipeline,
)
from vllm_omni.diffusion.registry import _prepare_diffusion_quant_config
from vllm_omni.host_weight_runtime import (
    ArtifactManifest,
    ArtifactSpec,
    ArtifactTopologyDescriptor,
    BuilderStartFailed,
    CapabilitiesUnavailable,
    CapabilityGrant,
    FatalFailure,
    HostWeightRuntime,
    LocalArtifactRepository,
    ProducerDescriptor,
    Ready,
    ResolvedAccess,
    RetryableFailure,
    TopologyCoordinate,
    WeightFormatDescriptor,
    canonical_digest,
    create_default_host_weight_runtime,
    derive_weight_format_plan_digest,
)
from vllm_omni.quantization import resolve_component_quant_config

from .binding import DiffusionConsumerBinder
from .build_coordination import (
    BuildComposition,
    BuildCoordinationError,
    BuilderCoordinator,
    FileBuilderCoordinator,
    compose_first_c4_build,
    launch_id_for_first_c4,
)
from .evidence import (
    HostWeightResolutionEvidence,
    evidence_from_outcome,
)
from .formats.fp8_per_tensor import (
    FORMAT_ADAPTER_ABI,
    FORMAT_ID,
    FORMAT_RECIPE_SCHEMA_VERSION,
    KERNEL_ID,
    TARGET_MODULE_TYPE_ID,
    Fp8PerTensorFormatAdapter,
)
from .ownership import (
    FatalPreparationFailure,
    LegacyReason,
    PreparationCleanupHandle,
    PreparedSessionReady,
    RetryablePreparationFailure,
    UseLegacy,
    WeightConsumerOwner,
)
from .pipeline_catalog import compile_pipeline_weight_catalog
from .producer import (
    DiffusionArtifactProducer,
    DiffusionArtifactProducerError,
    resolve_minimax_h3_transformer_source,
)
from .session import (
    HostCopyMode,
    PreparedWeightAccessSession,
    SessionCapabilities,
    SessionRequirements,
)
from .skeleton import (
    MINIMAX_H3_PIPELINE_FAMILY_ID,
    ConsumerCandidateOrigin,
    MiniMaxH3TransformerSkeletonFactory,
)
from .transfer import TransferPlanError


class WeightAccessPreparationError(RuntimeError):
    pass


class WeightAccessPreparationFallback(WeightAccessPreparationError):  # noqa: N818
    pass


def _safe_exception_detail(exc: BaseException) -> str:
    try:
        return str(exc)
    except BaseException:
        return f"<{type(exc).__name__} detail unavailable>"


def _add_cleanup_note(
    primary_error: BaseException,
    action: str,
    cleanup_error: BaseException,
) -> None:
    detail = _safe_exception_detail(cleanup_error)
    try:
        primary_error.add_note(f"{action} also failed: {type(cleanup_error).__name__}: {detail}")
    except BaseException:
        pass


def _raise_resolution_failure(
    outcome: RetryableFailure | FatalFailure,
) -> None:
    message = f"{outcome.code}: {outcome.detail}"
    if isinstance(outcome, RetryableFailure):
        raise WeightAccessPreparationFallback(message)
    raise WeightAccessPreparationError(message)


def _stable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, Mapping):
        return {str(key): _stable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_stable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_stable(item) for item in value), key=repr)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _stable(to_dict())
    if hasattr(value, "__dict__"):
        return {
            key: _stable(item)
            for key, item in sorted(vars(value).items())
            if not key.startswith("_") and not callable(item)
        }
    return f"{type(value).__module__}.{type(value).__qualname__}:{value}"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "source"


def _pre_resolve_process_memory() -> dict[str, int]:
    """Capture the exact process baseline immediately before core resolution."""

    result: dict[str, int] = {}
    try:
        for line in Path("/proc/self/smaps_rollup").read_text(encoding="utf-8").splitlines():
            if line.startswith(("Pss:", "Private_", "Locked:")):
                key, value, *_ = line.replace(":", "").split()
                result[f"{key.lower()}_kib"] = int(value)
        fields = Path("/proc/self/stat").read_text(encoding="utf-8").split()
        result["minor_faults"] = int(fields[9])
        result["major_faults"] = int(fields[11])
    except (OSError, ValueError, IndexError) as exc:
        raise WeightAccessPreparationError(f"pre_resolve_memory_unavailable: {type(exc).__name__}") from exc
    private_kib = result.get("private_clean_kib", 0) + result.get("private_dirty_kib", 0)
    result["private_kib"] = private_kib
    result["private_and_locked_kib"] = private_kib + result.get("locked_kib", 0)
    if "pss_kib" not in result:
        raise WeightAccessPreparationError("pre_resolve_memory_unavailable: smaps_rollup lacks Pss")
    return result


@dataclass(frozen=True)
class DiffusionPipelineSpec:
    pipeline_family_id: str
    model_config_digest: str
    normalized_init_config: Mapping[str, Any]


def _validate_scope(od_config: OmniDiffusionConfig, device: torch.device) -> None:
    parallel = od_config.parallel_config
    selected = sum(
        bool(value)
        for value in (
            od_config.enable_cpu_offload,
            od_config.enable_layerwise_offload,
            getattr(od_config, "enable_distributed_layerwise_offload", False),
        )
    )
    if selected != 1:
        raise WeightAccessPreparationError("Host Weight Runtime requires exactly one offload strategy")

    unsupported: list[str] = []
    if int(getattr(parallel, "pipeline_parallel_size", 1)) != 1:
        unsupported.append("pipeline parallelism")
    if int(getattr(parallel, "tensor_parallel_size", 1)) != 1:
        unsupported.append("tensor parallelism")
    if int(getattr(parallel, "sequence_parallel_size", 1) or 1) != 1:
        unsupported.append("sequence parallelism")
    if int(getattr(parallel, "cfg_parallel_size", 1) or 1) != 1:
        unsupported.append("CFG parallelism")
    if bool(getattr(parallel, "use_hsdp", False)):
        unsupported.append("HSDP")
    if bool(getattr(parallel, "enable_expert_parallel", False)):
        unsupported.append("expert parallelism")
    if getattr(od_config, "enable_distributed_layerwise_offload", False) and getattr(
        od_config,
        "dlo_use_allgather",
        True,
    ):
        unsupported.append("DLO AllGather")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if world_size != local_world_size:
        unsupported.append("multi-node execution")
    if device.type != "cuda":
        unsupported.append("non-CUDA producer")
    if unsupported:
        raise WeightAccessPreparationFallback("Host Weight Runtime v1 does not support " + ", ".join(unsupported))


def _resolved_fp8_config(od_config: OmniDiffusionConfig) -> Fp8Config:
    resolved = resolve_component_quant_config(
        od_config.quantization_config,
        "transformer",
    )
    if not isinstance(resolved, Fp8Config):
        raise WeightAccessPreparationFallback("Host Weight Runtime v1 supports only transformer FP8 quantization")
    if resolved.is_checkpoint_fp8_serialized:
        raise WeightAccessPreparationFallback("Host Weight Runtime v1 requires online FP8 from a dense checkpoint")
    if resolved.weight_block_size is not None:
        raise WeightAccessPreparationFallback("Host Weight Runtime v1 excludes blockwise FP8")
    if resolved.activation_scheme != "dynamic":
        raise WeightAccessPreparationFallback("Host Weight Runtime v1 requires dynamic activation scaling")
    if not cutlass_fp8_supported():
        raise WeightAccessPreparationFallback("Host Weight Runtime v1 requires the Cutlass FP8 kernel")
    return resolved


def _weight_format_descriptor(
    quant_config: Fp8Config,
    device: torch.device,
    runtime_dtype: torch.dtype,
) -> WeightFormatDescriptor:
    if not isinstance(runtime_dtype, torch.dtype):
        raise WeightAccessPreparationError(
            f"Host Weight Runtime requires a resolved torch dtype, got {runtime_dtype!r}"
        )
    runtime_dtype_name = str(runtime_dtype).removeprefix("torch.")
    index = device.index
    if index is None:
        index = torch.cuda.current_device()  # noqa: TID251
    major, minor = torch.cuda.get_device_capability(index)
    normalized_config = {
        "method": "fp8",
        "activation_scheme": quant_config.activation_scheme,
        "ignored_layers": sorted(quant_config.ignored_layers or []),
        "is_checkpoint_fp8_serialized": False,
        "store_dtype": quant_config.store_dtype,
        "weight_block_size": None,
        "input_dtype": runtime_dtype_name,
        "runtime_dtype": runtime_dtype_name,
    }
    kernel_identity = {
        "kernel_id": KERNEL_ID,
        "alignment": 16,
        "cuda_runtime": torch.version.cuda,
        "compute_capability": [major, minor],
        "torch_version": torch.__version__,
        "vllm_version": _package_version("vllm"),
    }
    semantic = canonical_digest(
        {
            "format_id": FORMAT_ID,
            "adapter_abi": FORMAT_ADAPTER_ABI,
            "normalized_config": normalized_config,
            "kernel_identity": kernel_identity,
            "target_module_type_id": TARGET_MODULE_TYPE_ID,
        }
    )
    format_plan_digest = derive_weight_format_plan_digest(
        format_id=FORMAT_ID,
        adapter_abi=FORMAT_ADAPTER_ABI,
        semantic_fingerprint=semantic,
        format_recipe_schema_version=FORMAT_RECIPE_SCHEMA_VERSION,
        target_module_type_id=TARGET_MODULE_TYPE_ID,
        normalized_config=normalized_config,
        kernel_identity=kernel_identity,
    )
    return WeightFormatDescriptor(
        format_id=FORMAT_ID,
        adapter_abi=FORMAT_ADAPTER_ABI,
        semantic_fingerprint=semantic,
        format_plan_digest=format_plan_digest,
        format_recipe_schema_version=FORMAT_RECIPE_SCHEMA_VERSION,
        target_module_type_id=TARGET_MODULE_TYPE_ID,
        normalized_config=normalized_config,
        kernel_identity=kernel_identity,
    )


def _producer_descriptor() -> ProducerDescriptor:
    fingerprint = canonical_digest(
        {
            "producer": "minimax_h3_fl2va_transformer",
            "producer_abi": "1",
            "target_module_type_id": TARGET_MODULE_TYPE_ID,
            "loader": "DiffusersPipelineLoader/ordinary-stream-online-fp8",
            "minimax_transforms": ["grouped_qkv_to_qkv", "fc1_gate_up"],
            "torch": torch.__version__,
            "vllm": _package_version("vllm"),
            "vllm_omni": _package_version("vllm-omni"),
        }
    )
    return ProducerDescriptor(
        producer_id="vllm_omni.minimax_h3.fp8_transformer",
        producer_abi="1",
        semantic_fingerprint=fingerprint,
    )


def _pipeline_spec(od_config: OmniDiffusionConfig) -> DiffusionPipelineSpec:
    config = od_config.tf_model_config
    config_value = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    if not isinstance(od_config.dtype, torch.dtype):
        raise WeightAccessPreparationError(
            f"Host Weight Runtime requires a resolved torch dtype, got {od_config.dtype!r}"
        )
    runtime_dtype = str(od_config.dtype).removeprefix("torch.")
    model_digest = canonical_digest(
        {
            "runtime_dtype": runtime_dtype,
            "transformer_config": _stable(config_value),
        }
    )
    return DiffusionPipelineSpec(
        pipeline_family_id=MINIMAX_H3_PIPELINE_FAMILY_ID,
        model_config_digest=model_digest,
        normalized_init_config={
            "partition": "fl2va",
            "task_type": str(getattr(od_config, "task_type", "fl2va")),
            "model_config_digest": model_digest,
            "runtime_dtype": runtime_dtype,
        },
    )


def _current_dp_rank(od_config: OmniDiffusionConfig) -> int:
    parallel_config = getattr(od_config, "parallel_config", None)
    dp_size = int(getattr(parallel_config, "data_parallel_size", 1))
    if dp_size == 1:
        return 0
    if torch.distributed.is_initialized():
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_data_parallel_rank,
        )

        return int(get_data_parallel_rank())
    # V1's accepted topology has TP=SP=PP=CFG=1, so global rank equals DP
    # rank.  This path also keeps composition tests independent of process
    # group initialization.
    return int(os.environ.get("RANK", "0"))


def _publish_pre_start_failure(
    composition: BuildComposition | None,
    artifact_key: str,
    error: BaseException,
) -> None:
    if composition is None or composition.publisher is None:
        return
    code = getattr(error, "code", "builder_preparation_failed")
    detail = getattr(error, "detail", None)
    if detail is None:
        detail = _safe_exception_detail(error)
    try:
        composition.publisher.publish_failed_if_pending(
            BuilderStartFailed(
                composition.authorization.launch_id,
                artifact_key,
                str(code),
                str(detail),
            )
        )
    except BaseException as signal_error:
        _add_cleanup_note(
            error,
            "publishing the ordered builder-start failure",
            signal_error,
        )


class WeightAccessSessionFactory:
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        load_config: LoadConfig,
        producer_device: torch.device,
        builder_coordinator: BuilderCoordinator | None = None,
        dp_rank: int | None = None,
    ) -> None:
        self.od_config = od_config
        self.load_config = load_config
        self.producer_device = producer_device
        self._builder_coordinator = builder_coordinator
        self._dp_rank = dp_rank
        self.manifest: ArtifactManifest | None = None
        self.resolution_info: object | None = None
        self.resolution_evidence: HostWeightResolutionEvidence | None = None

    @staticmethod
    def _legacy_reason(error: BaseException) -> LegacyReason:
        detail = _safe_exception_detail(error).lower()
        if any(token in detail for token in ("unsupported", "requires", "capability", "topology")):
            return LegacyReason.OPTIONAL_CAPABILITY_UNAVAILABLE
        return LegacyReason.OPTIONAL_ARTIFACT_UNAVAILABLE

    @staticmethod
    def _failure_code(error: BaseException, default: str) -> str:
        code = getattr(error, "code", None)
        if isinstance(code, str) and code:
            return code
        detail = _safe_exception_detail(error)
        prefix, separator, _ = detail.partition(": ")
        return prefix if separator and prefix else default

    def _publish_legacy_after_cleanup(
        self,
        *,
        owner: WeightConsumerOwner,
        cleanup: PreparationCleanupHandle | None,
        error: BaseException,
    ) -> None:
        if cleanup is not None and not cleanup.closed:
            try:
                cleanup.close()
            except BaseException as cleanup_error:
                _add_cleanup_note(error, "cleaning up optional HWR preparation", cleanup_error)
                owner.publish_preparation_result(
                    FatalPreparationFailure(
                        code="preparation_cleanup_incomplete",
                        detail=_safe_exception_detail(error),
                        cleanup_required=True,
                    )
                )
                return
        parallel = getattr(self.od_config, "parallel_config", None)
        dp_size = int(getattr(parallel, "data_parallel_size", 1))
        if dp_size > 1:
            # No rank may independently enter the ordinary loader while peers
            # retain or consume the shared HWR artifact.  A future composition
            # layer may turn this into UseLegacy only after an all-rank policy
            # decision; v1 fails closed at this boundary.
            owner.publish_preparation_result(
                RetryablePreparationFailure(
                    code="coordinated_fallback_required",
                    detail=(
                        f"{self._failure_code(error, 'optional_hwr_unavailable')}: {_safe_exception_detail(error)}"
                    ),
                    cleanup_required=False,
                )
            )
            return
        owner.publish_preparation_result(
            UseLegacy(
                reason=self._legacy_reason(error),
                detail=_safe_exception_detail(error),
            )
        )

    def _publish_fatal_after_cleanup(
        self,
        *,
        owner: WeightConsumerOwner,
        cleanup: PreparationCleanupHandle | None,
        error: BaseException,
    ) -> None:
        if cleanup is not None and not cleanup.closed:
            try:
                cleanup.close()
            except BaseException as cleanup_error:
                _add_cleanup_note(error, "cleaning up failed HWR preparation", cleanup_error)
        cleanup_required = cleanup is not None and not cleanup.closed
        owner.publish_preparation_result(
            FatalPreparationFailure(
                code=self._failure_code(error, "preparation_failed"),
                detail=_safe_exception_detail(error),
                cleanup_required=cleanup_required,
            )
        )

    def prepare_into(
        self,
        *,
        owner: WeightConsumerOwner,
        consumer_requirements: SessionRequirements,
        pipeline_builder: Callable[[nn.Module], nn.Module],
    ) -> None:
        """Publish one closed preparation result into a preinstalled owner."""

        cleanup: PreparationCleanupHandle | None = None
        composition: BuildComposition | None = None
        spec: ArtifactSpec | None = None
        try:
            _validate_scope(self.od_config, self.producer_device)
            source = resolve_minimax_h3_transformer_source(self.od_config)
            pipeline_spec = _pipeline_spec(self.od_config)
            _prepare_diffusion_quant_config(self.od_config, MiniMaxH3Pipeline)
            quant_config = _resolved_fp8_config(self.od_config)
            weight_format = _weight_format_descriptor(
                quant_config,
                self.producer_device,
                self.od_config.dtype,
            )
            if consumer_requirements.required_weight_format_id != weight_format.format_id:
                raise WeightAccessPreparationError("consumer requirements select a different finalized weight format")
            if consumer_requirements.host_copy_mode is not HostCopyMode.SYNCHRONOUS:
                raise WeightAccessPreparationFallback("Host Weight Runtime v1 requires synchronous host reads")
            spec = ArtifactSpec(
                source_fingerprint=canonical_digest(
                    {
                        "checkpoint_source": source.source_fingerprint,
                        "model_config": pipeline_spec.model_config_digest,
                    }
                ),
                producer=_producer_descriptor(),
                weight_format=weight_format,
                topology=ArtifactTopologyDescriptor(
                    (
                        TopologyCoordinate("pp", 1, 0),
                        TopologyCoordinate("tp", 1, 0),
                    )
                ),
                layout_abi="minimax_h3_dit_runtime/v1",
            )

            mode = str(self.od_config.host_weight_runtime_mode)
            root = Path(str(self.od_config.host_weight_runtime_root))
            parallel_config = getattr(self.od_config, "parallel_config", None)
            dp_size = int(getattr(parallel_config, "data_parallel_size", 1))
            dp_rank = self._dp_rank
            if dp_rank is None:
                dp_rank = _current_dp_rank(self.od_config)
            launch_id = launch_id_for_first_c4(
                artifact_key=spec.artifact_key,
                stage_id=int(getattr(self.od_config, "stage_id", 0)),
                master_port=getattr(self.od_config, "master_port", None),
            )
            coordinator = self._builder_coordinator or FileBuilderCoordinator(root)
            composition = compose_first_c4_build(
                mode=mode,
                dp_rank=dp_rank,
                dp_size=dp_size,
                launch_id=launch_id,
                artifact_key=spec.artifact_key,
                wait_timeout_s=float(self.od_config.host_weight_runtime_wait_timeout_s),
                coordinator=coordinator,
            )
            repository = LocalArtifactRepository(root)
            runtime = create_default_host_weight_runtime(
                repository,
                writable=mode == "read_write",
                verify_mmap_integrity=True,
            )
            cleanup = owner.begin_preparation(runtime)
            prepared = self._prepare_with_runtime(
                runtime=runtime,
                cleanup_handle=cleanup,
                source=source,
                spec=spec,
                pipeline_spec=pipeline_spec,
                quant_config=quant_config,
                weight_format=weight_format,
                mode=mode,
                composition=composition,
                pipeline_builder=pipeline_builder,
                consumer_requirements=consumer_requirements,
            )
            cleanup.transfer_to_prepared_session(prepared)
            owner.publish_preparation_result(PreparedSessionReady(prepared))
        except (
            DiffusionArtifactProducerError,
            BuildCoordinationError,
            OSError,
            WeightAccessPreparationFallback,
        ) as exc:
            if composition is not None and spec is not None:
                _publish_pre_start_failure(composition, spec.artifact_key, exc)
            self._publish_legacy_after_cleanup(
                owner=owner,
                cleanup=cleanup,
                error=exc,
            )
        except BaseException as exc:
            if composition is not None and spec is not None:
                _publish_pre_start_failure(composition, spec.artifact_key, exc)
            self._publish_fatal_after_cleanup(owner=owner, cleanup=cleanup, error=exc)

    def _prepare_with_runtime(
        self,
        *,
        runtime: HostWeightRuntime,
        cleanup_handle: PreparationCleanupHandle,
        source: Any,
        spec: ArtifactSpec,
        pipeline_spec: DiffusionPipelineSpec,
        quant_config: Fp8Config,
        weight_format: WeightFormatDescriptor,
        mode: str,
        composition: BuildComposition,
        pipeline_builder: Callable[[nn.Module], nn.Module],
        consumer_requirements: SessionRequirements,
    ) -> PreparedWeightAccessSession:
        capability = runtime.negotiate(consumer_requirements.access)
        if isinstance(capability, CapabilitiesUnavailable):
            missing = {
                kind.value: sorted(feature.value for feature in features)
                for kind, features in capability.missing_features_by_backing.items()
            }
            raise WeightAccessPreparationFallback(f"{capability.reason_code}: missing={missing}")
        if not isinstance(capability, CapabilityGrant):
            raise WeightAccessPreparationError(f"runtime returned an unknown capability decision {capability!r}")

        format_adapter = Fp8PerTensorFormatAdapter(weight_format)
        producer = None
        if composition.producer_allowed:
            producer = DiffusionArtifactProducer(
                spec=spec,
                od_config=self.od_config,
                source=source,
                device=self.producer_device,
                quant_config=quant_config,
                format_adapter=format_adapter,
                load_config=self.load_config,
            )
        pre_resolve = _pre_resolve_process_memory()
        outcome = runtime.resolve(
            spec,
            capability,
            producer,
            composition.authorization,
            cleanup_handle,
            composition.publisher,
            wait_timeout_s=float(self.od_config.host_weight_runtime_wait_timeout_s),
        )
        if isinstance(outcome, (Ready, RetryableFailure, FatalFailure)):
            self.resolution_evidence = evidence_from_outcome(
                outcome,
                runtime_mode=mode,
                expected_artifact_key=spec.artifact_key,
            )
        if not isinstance(outcome, Ready):
            if isinstance(outcome, (RetryableFailure, FatalFailure)):
                _raise_resolution_failure(outcome)
            raise AssertionError(f"unknown HWR outcome {outcome!r}")

        artifact = outcome.artifact
        expected_access = ResolvedAccess(
            runtime_instance_id=capability.runtime_instance_id,
            grant_id=capability.grant_id,
            backing_kind=capability.backing_kind,
            provider_id=capability.provider_id,
            provider_abi=capability.provider_abi,
            features=capability.features,
        )
        if outcome.access != expected_access:
            raise WeightAccessPreparationError("resolved access does not match the negotiated capability grant")
        self.manifest = artifact.manifest
        self.resolution_info = outcome.info
        self.resolution_evidence = evidence_from_outcome(
            outcome,
            runtime_mode=mode,
            expected_artifact_key=spec.artifact_key,
            artifact_compatibility_digest=artifact.manifest.compatibility_digest,
        )
        skeleton_factory = MiniMaxH3TransformerSkeletonFactory(
            self.od_config,
            quant_config=quant_config,
            pipeline_factory=lambda **kwargs: pipeline_builder(kwargs["transformer"]),
        )
        if outcome.info.path.value == "mmap_built":
            origin = ConsumerCandidateOrigin.BUILDER_REBIND
        elif outcome.info.path.value == "mmap_wait_hit":
            origin = ConsumerCandidateOrigin.ORDERED_WAITER
        else:
            origin = ConsumerCandidateOrigin.WARM_HIT
        candidate = skeleton_factory.create_candidate(
            pipeline_spec,
            weight_format,
            origin,
            spec.artifact_key,
            cleanup_handle,
        )
        skeleton = candidate.skeleton
        binder = DiffusionConsumerBinder(
            target_module_type=skeleton_factory.target_module_type,
        )
        # The logical manifest provides exact recipe coverage before final
        # FP8 shapes have been hydrated into the meta skeleton.
        prepared_binding = binder.prepare(
            skeleton,
            artifact.manifest,
            artifact.manifest,
            format_adapter,
            cleanup_handle,
        )
        prepared_binding.hydrate()
        compiled = compile_pipeline_weight_catalog(
            skeleton.pipeline,  # type: ignore[arg-type]
            artifact_compatibility_digest=artifact.manifest.compatibility_digest,
        )
        set_catalog = getattr(prepared_binding, "set_transfer_catalog", None)
        if not callable(set_catalog):
            raise WeightAccessPreparationError("consumer binder cannot accept the finalized transfer catalog")
        set_catalog(compiled.transfer_catalog)
        prepared_binding.validate()

        try:
            transfer_plan = compiled.transfer_catalog.plan_for_kind(consumer_requirements.required_transfer_plan_kind)
        except TransferPlanError as exc:
            raise WeightAccessPreparationFallback(str(exc)) from exc
        selected_kinds = frozenset(
            compiled.transfer_catalog.unit(unit_id).unit_kind for unit_id in transfer_plan.unit_ids
        )
        capabilities = SessionCapabilities(
            runtime_instance_id=outcome.access.runtime_instance_id,
            capability_grant_id=outcome.access.grant_id,
            access_features=outcome.access.features,
            selected_transfer_plan_id=transfer_plan.plan_id,
            selected_transfer_plan_kind=transfer_plan.plan_kind,
            unit_kinds=selected_kinds,
            weight_format_id=weight_format.format_id,
            backing_kind=outcome.access.backing_kind,
            provider_id=outcome.access.provider_id,
            provider_abi=outcome.access.provider_abi,
            host_copy_mode=HostCopyMode.SYNCHRONOUS,
        )
        builder_started: dict[str, object] | None = None
        observed_builder_started: dict[str, object] | None = None
        if outcome.info.path.value == "mmap_built":
            builder_started = {
                "launch_id": composition.authorization.launch_id,
                "artifact_key": spec.artifact_key,
                "lease_id": outcome.info.generation_id,
                "builder_actor_id": composition.authorization.actor_id,
            }
        if composition.authorization.observed_start is not None:
            observed_builder_started = asdict(composition.authorization.observed_start)
        if self.resolution_evidence is None:
            raise WeightAccessPreparationError("ready resolution did not produce Host Weight Runtime evidence")
        self.resolution_evidence = replace(
            self.resolution_evidence,
            negotiated_capability_grant_id=capability.grant_id,
            selected_transfer_plan_id=transfer_plan.plan_id,
            selected_transfer_plan_kind=transfer_plan.plan_kind.value,
            exact_coverage_digest=transfer_plan.exact_coverage_digest,
            unit_kinds=tuple(sorted(kind.value for kind in selected_kinds)),
            pre_resolve=pre_resolve,
            builder_started=builder_started,
            observed_builder_started=observed_builder_started,
            producer_present=producer is not None,
        )
        prepared = PreparedWeightAccessSession(
            pipeline=skeleton.pipeline,
            catalog=compiled.transfer_catalog,
            transfer_plan=transfer_plan,
            capabilities=capabilities,
            artifact=artifact,
            binding=prepared_binding,
            runtime=runtime,
        )
        return prepared


__all__ = [
    "DiffusionPipelineSpec",
    "PreparationCleanupHandle",
    "WeightAccessPreparationError",
    "WeightAccessPreparationFallback",
    "WeightAccessSessionFactory",
]
