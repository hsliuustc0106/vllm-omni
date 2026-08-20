# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Composition root for pre-loader diffusion host-weight sessions."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch import nn
from vllm.config.load import LoadConfig

from vllm_omni.diffusion.data import OmniDiffusionConfig
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
    Ready,
    ResolvedAccess,
    RetryableFailure,
    TopologyCoordinate,
    canonical_digest,
    create_default_host_weight_runtime,
)

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
from .integrations import create_builtin_model_integration_registry
from .model_integration import (
    DiffusionPipelineSpec,
    HostWeightModelIntegrationBundle,
    ModelIntegrationLegacyClassification,
    ModelIntegrationRegistry,
    ModelIntegrationUnavailableError,
    PreparedModelIntegration,
    ProducerFactoryContext,
    SkeletonFactoryContext,
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
from .session import (
    HostCopyMode,
    PreparedWeightAccessSession,
    SessionCapabilities,
    SessionRequirements,
)
from .skeleton import ConsumerCandidateOrigin
from .transfer import TransferPlanError


class WeightAccessPreparationError(RuntimeError):
    def __init__(self, detail: str, *, code: str = "preparation_failed") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class WeightAccessPreparationFallback(WeightAccessPreparationError):  # noqa: N818
    def __init__(
        self,
        detail: str,
        *,
        code: str = "optional_hwr_unavailable",
        legacy_reason: LegacyReason = LegacyReason.OPTIONAL_ARTIFACT_UNAVAILABLE,
    ) -> None:
        self.legacy_reason = LegacyReason(legacy_reason)
        super().__init__(detail, code=code)


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
    if isinstance(outcome, RetryableFailure):
        raise WeightAccessPreparationFallback(
            outcome.detail,
            code=outcome.code,
            legacy_reason=LegacyReason.OPTIONAL_ARTIFACT_UNAVAILABLE,
        )
    raise WeightAccessPreparationError(outcome.detail, code=outcome.code)


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
        raise WeightAccessPreparationError(
            type(exc).__name__,
            code="pre_resolve_memory_unavailable",
        ) from exc
    private_kib = result.get("private_clean_kib", 0) + result.get("private_dirty_kib", 0)
    result["private_kib"] = private_kib
    result["private_and_locked_kib"] = private_kib + result.get("locked_kib", 0)
    if "pss_kib" not in result:
        raise WeightAccessPreparationError(
            "smaps_rollup lacks Pss",
            code="pre_resolve_memory_unavailable",
        )
    return result


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
        raise WeightAccessPreparationError(
            "Host Weight Runtime requires exactly one offload strategy",
            code="offload_strategy_invalid",
        )

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
        raise WeightAccessPreparationFallback(
            "Host Weight Runtime v1 does not support " + ", ".join(unsupported),
            code="topology_unsupported",
            legacy_reason=LegacyReason.OPTIONAL_CAPABILITY_UNAVAILABLE,
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
        model_integration_registry: ModelIntegrationRegistry | None = None,
    ) -> None:
        self.od_config = od_config
        self.load_config = load_config
        self.producer_device = producer_device
        self._builder_coordinator = builder_coordinator
        self._dp_rank = dp_rank
        self._model_integration_registry = (
            model_integration_registry
            if model_integration_registry is not None
            else create_builtin_model_integration_registry()
        )
        if not isinstance(self._model_integration_registry, ModelIntegrationRegistry):
            raise TypeError("model_integration_registry must be a ModelIntegrationRegistry")
        self.manifest: ArtifactManifest | None = None
        self.resolution_info: object | None = None
        self.resolution_evidence: HostWeightResolutionEvidence | None = None

    @staticmethod
    def _legacy_reason(error: BaseException) -> LegacyReason:
        if isinstance(error, ModelIntegrationUnavailableError):
            if error.legacy_classification is ModelIntegrationLegacyClassification.CAPABILITY:
                return LegacyReason.OPTIONAL_CAPABILITY_UNAVAILABLE
            return LegacyReason.OPTIONAL_ARTIFACT_UNAVAILABLE
        if isinstance(error, WeightAccessPreparationFallback):
            return error.legacy_reason
        return LegacyReason.OPTIONAL_ARTIFACT_UNAVAILABLE

    @staticmethod
    def _failure_code(error: BaseException, default: str) -> str:
        code = getattr(error, "code", None)
        if isinstance(code, str) and code:
            return code
        return default

    @staticmethod
    def _failure_detail(error: BaseException) -> str:
        detail = getattr(error, "detail", None)
        if isinstance(detail, str) and detail:
            return detail
        return _safe_exception_detail(error)

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
                    detail=f"{self._failure_code(error, 'optional_hwr_unavailable')}: {self._failure_detail(error)}",
                    cleanup_required=False,
                )
            )
            return
        owner.publish_preparation_result(
            UseLegacy(
                reason=self._legacy_reason(error),
                detail=self._failure_detail(error),
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
                detail=self._failure_detail(error),
                cleanup_required=cleanup_required,
            )
        )

    def prepare_into(
        self,
        *,
        owner: WeightConsumerOwner,
        consumer_requirements_factory: Callable[[str], SessionRequirements],
        pipeline_builder: Callable[[nn.Module], nn.Module],
    ) -> None:
        """Publish one closed preparation result into a preinstalled owner."""

        cleanup: PreparationCleanupHandle | None = None
        composition: BuildComposition | None = None
        spec: ArtifactSpec | None = None
        try:
            _validate_scope(self.od_config, self.producer_device)
            integration = self._model_integration_registry.select(self.od_config)
            try:
                consumer_requirements = consumer_requirements_factory(integration.capabilities.weight_format_id)
            except Exception as exc:
                raise WeightAccessPreparationError(
                    f"consumer requirements factory raised {type(exc).__name__}: {_safe_exception_detail(exc)}",
                    code="consumer_requirements_factory_failed",
                ) from exc
            if not isinstance(consumer_requirements, SessionRequirements):
                raise WeightAccessPreparationError(
                    "consumer requirements factory returned an unsupported value",
                    code="consumer_requirements_invalid",
                )
            if consumer_requirements.required_weight_format_id != integration.capabilities.weight_format_id:
                raise WeightAccessPreparationError(
                    "consumer requirements changed the selected integration format",
                    code="consumer_weight_format_mismatch",
                )
            self._model_integration_registry.validate_transfer_plan(
                integration,
                consumer_requirements.required_transfer_plan_kind,
            )
            if consumer_requirements.host_copy_mode is not HostCopyMode.SYNCHRONOUS:
                raise WeightAccessPreparationFallback(
                    "Host Weight Runtime v1 requires synchronous host reads",
                    code="host_copy_mode_unsupported",
                    legacy_reason=LegacyReason.OPTIONAL_CAPABILITY_UNAVAILABLE,
                )
            prepared_model = integration.prepare(self.od_config, self.producer_device)
            weight_format = prepared_model.weight_format
            if consumer_requirements.required_weight_format_id != weight_format.format_id:
                raise WeightAccessPreparationError(
                    "consumer requirements select a different finalized weight format",
                    code="consumer_weight_format_mismatch",
                )
            spec = ArtifactSpec(
                source_fingerprint=canonical_digest(
                    {
                        "checkpoint_source": prepared_model.source.source_fingerprint,
                        "model_config": prepared_model.pipeline_spec.model_config_digest,
                    }
                ),
                producer=prepared_model.producer_descriptor,
                weight_format=weight_format,
                topology=ArtifactTopologyDescriptor(
                    (
                        TopologyCoordinate("pp", 1, 0),
                        TopologyCoordinate("tp", 1, 0),
                    )
                ),
                layout_abi=integration.capabilities.artifact_layout_abi,
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
                integration=integration,
                prepared_model=prepared_model,
                spec=spec,
                mode=mode,
                composition=composition,
                pipeline_builder=pipeline_builder,
                consumer_requirements=consumer_requirements,
            )
            cleanup.transfer_to_prepared_session(prepared)
            owner.publish_preparation_result(PreparedSessionReady(prepared))
        except (
            BuildCoordinationError,
            ModelIntegrationUnavailableError,
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
        integration: HostWeightModelIntegrationBundle,
        prepared_model: PreparedModelIntegration,
        spec: ArtifactSpec,
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
            raise WeightAccessPreparationFallback(
                f"missing={missing}",
                code=capability.reason_code,
                legacy_reason=LegacyReason.OPTIONAL_CAPABILITY_UNAVAILABLE,
            )
        if not isinstance(capability, CapabilityGrant):
            raise WeightAccessPreparationError(
                f"runtime returned an unknown capability decision {capability!r}",
                code="capability_decision_invalid",
            )

        weight_format = prepared_model.weight_format
        format_adapter = prepared_model.format_adapter
        producer = None
        if composition.producer_allowed:
            producer = integration.create_producer(
                ProducerFactoryContext(
                    spec=spec,
                    od_config=self.od_config,
                    source=prepared_model.source,
                    device=self.producer_device,
                    quant_config=prepared_model.quant_config,
                    format_adapter=format_adapter,
                    load_config=self.load_config,
                )
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
            raise WeightAccessPreparationError(
                "resolved access does not match the negotiated capability grant",
                code="resolved_access_mismatch",
            )
        self.manifest = artifact.manifest
        self.resolution_info = outcome.info
        self.resolution_evidence = evidence_from_outcome(
            outcome,
            runtime_mode=mode,
            expected_artifact_key=spec.artifact_key,
            artifact_compatibility_digest=artifact.manifest.compatibility_digest,
        )
        skeleton_factory = integration.create_skeleton_factory(
            SkeletonFactoryContext(
                od_config=self.od_config,
                quant_config=prepared_model.quant_config,
                pipeline_builder=pipeline_builder,
            )
        )
        if outcome.info.path.value == "mmap_built":
            origin = ConsumerCandidateOrigin.BUILDER_REBIND
        elif outcome.info.path.value == "mmap_wait_hit":
            origin = ConsumerCandidateOrigin.ORDERED_WAITER
        else:
            origin = ConsumerCandidateOrigin.WARM_HIT
        candidate = skeleton_factory.create_candidate(
            prepared_model.pipeline_spec,
            weight_format,
            origin,
            spec.artifact_key,
            cleanup_handle,
        )
        skeleton = candidate.skeleton
        binder = DiffusionConsumerBinder(
            target_module_type=skeleton_factory.target_module_type,
        )
        # The logical manifest provides exact recipe coverage before the
        # bundle-owned finalized shapes are hydrated into the meta skeleton.
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
            raise WeightAccessPreparationError(
                "consumer binder cannot accept the finalized transfer catalog",
                code="consumer_catalog_unsupported",
            )
        set_catalog(compiled.transfer_catalog)
        prepared_binding.validate()

        try:
            transfer_plan = compiled.transfer_catalog.plan_for_kind(consumer_requirements.required_transfer_plan_kind)
        except TransferPlanError as exc:
            raise WeightAccessPreparationFallback(
                str(exc),
                code="transfer_plan_unavailable",
                legacy_reason=LegacyReason.OPTIONAL_CAPABILITY_UNAVAILABLE,
            ) from exc
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
            raise WeightAccessPreparationError(
                "ready resolution did not produce Host Weight Runtime evidence",
                code="resolution_evidence_missing",
            )
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
