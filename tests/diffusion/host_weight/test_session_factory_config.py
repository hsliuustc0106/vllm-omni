# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration and ownership tests for the host-weight session factory."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.config.load import LoadConfig

from vllm_omni.diffusion.host_weight import session_factory as factory_module
from vllm_omni.diffusion.host_weight.formats.fp8_per_tensor import (
    FORMAT_ADAPTER_ABI,
    FORMAT_ID,
    FORMAT_RECIPE_SCHEMA_VERSION,
    TARGET_MODULE_TYPE_ID,
)
from vllm_omni.diffusion.host_weight.integrations import minimax_h3_fp8
from vllm_omni.diffusion.host_weight.model_integration import (
    HOST_WEIGHT_MODEL_INTEGRATION_ABI,
    DiffusionPipelineSpec,
    HostWeightModelIntegrationBundle,
    ModelIntegrationCapabilities,
    ModelIntegrationLegacyClassification,
    ModelIntegrationRegistry,
    ModelIntegrationSupportDecision,
    ModelIntegrationUnavailableError,
)
from vllm_omni.diffusion.host_weight.ownership import (
    FatalPreparationFailure,
    LegacyReason,
    PreparedSessionReady,
    RetryablePreparationFailure,
    UseLegacy,
    WeightConsumerOwner,
    WeightConsumerOwnerPhase,
)
from vllm_omni.diffusion.host_weight.session import (
    HostCopyMode,
    SessionRequirements,
)
from vllm_omni.diffusion.host_weight.transfer import TransferPlanKind
from vllm_omni.host_weight_runtime import (
    AccessFeature,
    AccessRequirements,
    BackingKind,
    FatalFailure,
    ProducerDescriptor,
    RetryableFailure,
    WeightFormatDescriptor,
    canonical_digest,
    derive_weight_format_plan_digest,
)


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (
            RetryableFailure("wait_timeout", "builder still active"),
            factory_module.WeightAccessPreparationFallback,
        ),
        (
            FatalFailure("corrupt_store", "digest mismatch"),
            factory_module.WeightAccessPreparationError,
        ),
    ],
)
def test_resolution_failure_policy(outcome: object, expected: type[Exception]) -> None:
    with pytest.raises(expected, match=outcome.code):  # type: ignore[attr-defined]
        factory_module._raise_resolution_failure(outcome)  # type: ignore[arg-type]


def test_fallback_classification_and_code_are_typed_not_message_parsed() -> None:
    artifact_error = factory_module.WeightAccessPreparationFallback(
        "unsupported topology words are only detail",
        code="artifact_missing",
        legacy_reason=LegacyReason.OPTIONAL_ARTIFACT_UNAVAILABLE,
    )
    capability_error = ModelIntegrationUnavailableError(
        "format_unsupported",
        "cache artifact words are only detail",
        legacy_classification=ModelIntegrationLegacyClassification.CAPABILITY,
    )

    assert (
        factory_module.WeightAccessSessionFactory._legacy_reason(artifact_error)
        is LegacyReason.OPTIONAL_ARTIFACT_UNAVAILABLE
    )
    assert (
        factory_module.WeightAccessSessionFactory._legacy_reason(capability_error)
        is LegacyReason.OPTIONAL_CAPABILITY_UNAVAILABLE
    )
    assert factory_module.WeightAccessSessionFactory._failure_code(artifact_error, "default") == "artifact_missing"
    assert (
        factory_module.WeightAccessSessionFactory._failure_code(RuntimeError("parsed_code: do not parse"), "default")
        == "default"
    )


def _scope_config(**parallel_overrides: object) -> SimpleNamespace:
    parallel_values = {
        "pipeline_parallel_size": 1,
        "tensor_parallel_size": 1,
        "sequence_parallel_size": 1,
        "cfg_parallel_size": 1,
        "data_parallel_size": 1,
        "use_hsdp": False,
        "enable_expert_parallel": False,
    }
    parallel_values.update(parallel_overrides)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(**parallel_values),
        enable_cpu_offload=True,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=False,
        dlo_use_allgather=False,
    )


def _config(tmp_path, *, dp_size: int = 1, mode: str = "read_only") -> SimpleNamespace:
    config = _scope_config(data_parallel_size=dp_size)
    config.host_weight_runtime_mode = mode
    config.host_weight_runtime_root = str(tmp_path)
    config.host_weight_runtime_wait_timeout_s = 17.5
    config.task_type = "t2va"
    config.tf_model_config = {}
    config.dtype = torch.bfloat16
    config.stage_id = 0
    config.master_port = None
    config.model_class_name = "TestPipeline"
    return config


def _requirements(required_weight_format_id: str = FORMAT_ID) -> SessionRequirements:
    return SessionRequirements(
        access=AccessRequirements(
            required_features=frozenset(
                {
                    AccessFeature.COMPLETE_TENSOR_READ,
                    AccessFeature.SHARED_PAGES,
                }
            ),
            accepted_backings=frozenset({BackingKind.RUNTIME_MMAP}),
        ),
        required_transfer_plan_kind=TransferPlanKind.COMPONENT,
        required_weight_format_id=required_weight_format_id,
        host_copy_mode=HostCopyMode.SYNCHRONOUS,
    )


def _weight_format() -> WeightFormatDescriptor:
    semantic = canonical_digest("format")
    normalized_config: dict[str, object] = {}
    kernel_identity: dict[str, object] = {}
    plan_digest = derive_weight_format_plan_digest(
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
        format_plan_digest=plan_digest,
        format_recipe_schema_version=FORMAT_RECIPE_SCHEMA_VERSION,
        target_module_type_id=TARGET_MODULE_TYPE_ID,
        normalized_config=normalized_config,
        kernel_identity=kernel_identity,
    )


class _FormatAdapter:
    def __init__(self, descriptor: WeightFormatDescriptor) -> None:
        self.descriptor = descriptor


def _test_integration_registry() -> ModelIntegrationRegistry:
    bundle = HostWeightModelIntegrationBundle(
        integration_id="test.pipeline.format",
        integration_abi=HOST_WEIGHT_MODEL_INTEGRATION_ABI,
        capabilities=ModelIntegrationCapabilities(
            model_class_names=frozenset({"TestPipeline"}),
            pipeline_family_id="test_pipeline",
            weight_format_id=FORMAT_ID,
            target_module_type_id=TARGET_MODULE_TYPE_ID,
            artifact_layout_abi="test_pipeline/v1",
            supported_transfer_plan_kinds=frozenset({TransferPlanKind.COMPONENT}),
        ),
        support_probe=lambda _config: ModelIntegrationSupportDecision.accepted(),
        source_resolver=lambda _config: SimpleNamespace(source_fingerprint=canonical_digest("source")),
        pipeline_spec_factory=lambda _config: DiffusionPipelineSpec(
            pipeline_family_id="test_pipeline",
            model_config_digest=canonical_digest("model"),
            normalized_init_config={},
        ),
        quantization_preparer=lambda _config: object(),
        weight_format_factory=lambda *_args: _weight_format(),
        format_adapter_factory=_FormatAdapter,
        producer_descriptor_factory=lambda: ProducerDescriptor(
            "test.pipeline.producer",
            "1",
            canonical_digest("producer"),
        ),
        producer_factory=lambda _context: (_ for _ in ()).throw(AssertionError("producer factory not expected")),
        skeleton_factory=lambda _context: (_ for _ in ()).throw(AssertionError("skeleton factory not expected")),
    )
    return ModelIntegrationRegistry((bundle,))


def _patch_preparation_preamble(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory_module, "_validate_scope", lambda *_args: None)
    monkeypatch.setattr(
        factory_module,
        "create_builtin_model_integration_registry",
        _test_integration_registry,
    )
    monkeypatch.setattr(
        factory_module,
        "LocalArtifactRepository",
        lambda root: ("repository", root),
    )


class _Runtime:
    def __init__(self, *, fail_first_close: bool = False) -> None:
        self.fail_first_close = fail_first_close
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.fail_first_close and self.close_calls == 1:
            raise RuntimeError("injected runtime cleanup failure")


class _Prepared:
    def __init__(self, runtime: _Runtime, pipeline: object) -> None:
        self._runtime = runtime
        self._artifact = None
        self._binding = None
        self.pipeline = pipeline
        self.rollback_calls = 0

    def rollback(self) -> None:
        self.rollback_calls += 1
        self._runtime.close()


def test_scope_capability_rejection_is_optional_fallback() -> None:
    config = _scope_config(tensor_parallel_size=2)

    with pytest.raises(
        factory_module.WeightAccessPreparationFallback,
        match="tensor parallelism",
    ):
        factory_module._validate_scope(config, torch.device("cuda:0"))


def test_scope_configuration_error_remains_hard() -> None:
    config = _scope_config()
    config.enable_cpu_offload = False

    with pytest.raises(
        factory_module.WeightAccessPreparationError,
        match="exactly one offload strategy",
    ) as raised:
        factory_module._validate_scope(config, torch.device("cuda:0"))

    assert not isinstance(
        raised.value,
        factory_module.WeightAccessPreparationFallback,
    )


def test_prepare_into_uses_explicit_composition_and_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _patch_preparation_preamble(monkeypatch)
    observed: dict[str, object] = {}
    runtime = _Runtime()
    pipeline = object()

    def compose(**kwargs: object) -> SimpleNamespace:
        observed["composition"] = kwargs
        return SimpleNamespace(
            authorization=SimpleNamespace(),
            publisher=None,
            producer_allowed=False,
        )

    def create_runtime(repository: object, **kwargs: object) -> _Runtime:
        observed["repository"] = repository
        observed["runtime_kwargs"] = kwargs
        return runtime

    monkeypatch.setattr(factory_module, "compose_first_c4_build", compose)
    monkeypatch.setattr(
        factory_module,
        "create_default_host_weight_runtime",
        create_runtime,
    )
    config = _config(tmp_path)
    factory = factory_module.WeightAccessSessionFactory(
        od_config=config,
        load_config=LoadConfig(),
        producer_device=torch.device("cuda:0"),
        builder_coordinator=object(),  # type: ignore[arg-type]
        dp_rank=0,
    )

    def prepare(**kwargs: object) -> _Prepared:
        observed["prepare"] = kwargs
        return _Prepared(runtime, pipeline)

    monkeypatch.setattr(factory, "_prepare_with_runtime", prepare)
    owner = WeightConsumerOwner()

    factory.prepare_into(
        owner=owner,
        consumer_requirements_factory=_requirements,
        pipeline_builder=lambda _transformer: pipeline,  # type: ignore[return-value]
    )

    result = owner.preparation_result
    assert isinstance(result, PreparedSessionReady)
    assert result.prepared_session.pipeline is pipeline
    composition = observed["composition"]
    assert isinstance(composition, dict)
    assert composition["wait_timeout_s"] == 17.5
    assert composition["mode"] == "read_only"
    assert observed["runtime_kwargs"] == {
        "writable": False,
        "verify_mmap_integrity": True,
    }
    assert observed["prepare"]["runtime"] is runtime  # type: ignore[index]
    owner.close()
    assert runtime.close_calls == 1


def test_prepare_into_validates_selected_plan_before_source_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(factory_module, "_validate_scope", lambda *_args: None)
    base = _test_integration_registry().bundles[0]
    source_calls = 0

    def source_resolver(_config: object) -> SimpleNamespace:
        nonlocal source_calls
        source_calls += 1
        return SimpleNamespace(source_fingerprint=canonical_digest("source"))

    block_only = replace(
        base,
        integration_id="test.pipeline.block_only",
        capabilities=replace(
            base.capabilities,
            supported_transfer_plan_kinds=frozenset({TransferPlanKind.BLOCKS_PLUS_RESIDENT}),
        ),
        source_resolver=source_resolver,
    )
    factory = factory_module.WeightAccessSessionFactory(
        od_config=_config(tmp_path),
        load_config=LoadConfig(),
        producer_device=torch.device("cuda:0"),
        model_integration_registry=ModelIntegrationRegistry((block_only,)),
    )
    owner = WeightConsumerOwner()

    factory.prepare_into(
        owner=owner,
        consumer_requirements_factory=_requirements,
        pipeline_builder=lambda transformer: transformer,
    )

    result = owner.preparation_result
    assert isinstance(result, UseLegacy)
    assert result.reason is LegacyReason.OPTIONAL_CAPABILITY_UNAVAILABLE
    assert "does not support transfer plan" in result.detail
    assert source_calls == 0
    owner.close()


def test_probe_no_match_falls_back_but_ambiguous_match_is_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(factory_module, "_validate_scope", lambda *_args: None)
    base = _test_integration_registry().bundles[0]
    rejected = replace(
        base,
        support_probe=lambda _config: ModelIntegrationSupportDecision.rejected(
            "test_format_unsupported",
            "test quantization is unsupported",
        ),
    )
    fallback_factory = factory_module.WeightAccessSessionFactory(
        od_config=_config(tmp_path),
        load_config=LoadConfig(),
        producer_device=torch.device("cuda:0"),
        model_integration_registry=ModelIntegrationRegistry((rejected,)),
    )
    fallback_owner = WeightConsumerOwner()
    fallback_factory.prepare_into(
        owner=fallback_owner,
        consumer_requirements_factory=_requirements,
        pipeline_builder=lambda transformer: transformer,
    )
    fallback = fallback_owner.preparation_result
    assert isinstance(fallback, UseLegacy)
    assert fallback.reason is LegacyReason.OPTIONAL_CAPABILITY_UNAVAILABLE
    assert "test quantization is unsupported" in fallback.detail
    fallback_owner.close()

    alternate = replace(
        base,
        integration_id="test.pipeline.alternate",
        capabilities=replace(
            base.capabilities,
            weight_format_id="test_alternate",
        ),
    )
    fatal_factory = factory_module.WeightAccessSessionFactory(
        od_config=_config(tmp_path),
        load_config=LoadConfig(),
        producer_device=torch.device("cuda:0"),
        model_integration_registry=ModelIntegrationRegistry((base, alternate)),
    )
    fatal_owner = WeightConsumerOwner()
    fatal_factory.prepare_into(
        owner=fatal_owner,
        consumer_requirements_factory=_requirements,
        pipeline_builder=lambda transformer: transformer,
    )
    fatal = fatal_owner.preparation_result
    assert isinstance(fatal, FatalPreparationFailure)
    assert fatal.code == "model_integration_ambiguous"
    fatal_owner.close()


@pytest.mark.parametrize(
    ("failure_point", "expected_code"),
    [
        ("probe", "model_integration_probe_failed"),
        ("requirements", "consumer_requirements_factory_failed"),
    ],
)
def test_callback_exception_is_fatal_before_source_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    failure_point: str,
    expected_code: str,
) -> None:
    monkeypatch.setattr(factory_module, "_validate_scope", lambda *_args: None)
    base = _test_integration_registry().bundles[0]
    source_calls = 0

    def source_resolver(_config: object) -> SimpleNamespace:
        nonlocal source_calls
        source_calls += 1
        return SimpleNamespace(source_fingerprint=canonical_digest("source"))

    def support_probe(_config: object) -> ModelIntegrationSupportDecision:
        if failure_point == "probe":
            raise OSError("injected probe failure")
        return ModelIntegrationSupportDecision.accepted()

    bundle = replace(
        base,
        support_probe=support_probe,
        source_resolver=source_resolver,
    )
    factory = factory_module.WeightAccessSessionFactory(
        od_config=_config(tmp_path),
        load_config=LoadConfig(),
        producer_device=torch.device("cuda:0"),
        model_integration_registry=ModelIntegrationRegistry((bundle,)),
    )
    owner = WeightConsumerOwner()

    def requirements(format_id: str) -> SessionRequirements:
        if failure_point == "requirements":
            raise OSError("injected requirements failure")
        return _requirements(format_id)

    factory.prepare_into(
        owner=owner,
        consumer_requirements_factory=requirements,
        pipeline_builder=lambda transformer: transformer,
    )

    result = owner.preparation_result
    assert isinstance(result, FatalPreparationFailure)
    assert result.code == expected_code
    assert result.cleanup_required is False
    assert source_calls == 0
    owner.close()


def test_single_rank_optional_failure_publishes_use_legacy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        factory_module,
        "_validate_scope",
        lambda *_args: (_ for _ in ()).throw(factory_module.WeightAccessPreparationFallback("unsupported topology")),
    )
    factory = factory_module.WeightAccessSessionFactory(
        od_config=_config(tmp_path, dp_size=1),
        load_config=LoadConfig(),
        producer_device=torch.device("cuda:0"),
    )
    owner = WeightConsumerOwner()

    factory.prepare_into(
        owner=owner,
        consumer_requirements_factory=_requirements,
        pipeline_builder=lambda transformer: transformer,
    )

    assert isinstance(owner.preparation_result, UseLegacy)
    owner.close()


def test_multi_rank_optional_failure_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        factory_module,
        "_validate_scope",
        lambda *_args: (_ for _ in ()).throw(factory_module.WeightAccessPreparationFallback("artifact unavailable")),
    )
    factory = factory_module.WeightAccessSessionFactory(
        od_config=_config(tmp_path, dp_size=4),
        load_config=LoadConfig(),
        producer_device=torch.device("cuda:0"),
        dp_rank=2,
    )
    owner = WeightConsumerOwner()

    factory.prepare_into(
        owner=owner,
        consumer_requirements_factory=_requirements,
        pipeline_builder=lambda transformer: transformer,
    )

    result = owner.preparation_result
    assert isinstance(result, RetryablePreparationFailure)
    assert result.code == "coordinated_fallback_required"
    assert result.cleanup_required is False
    owner.close()


def test_cleanup_failure_retains_owner_authority_for_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _patch_preparation_preamble(monkeypatch)
    runtime = _Runtime(fail_first_close=True)
    monkeypatch.setattr(
        factory_module,
        "compose_first_c4_build",
        lambda **_kwargs: SimpleNamespace(
            authorization=SimpleNamespace(),
            publisher=None,
            producer_allowed=False,
        ),
    )
    monkeypatch.setattr(
        factory_module,
        "create_default_host_weight_runtime",
        lambda *_args, **_kwargs: runtime,
    )
    factory = factory_module.WeightAccessSessionFactory(
        od_config=_config(tmp_path),
        load_config=LoadConfig(),
        producer_device=torch.device("cuda:0"),
        builder_coordinator=object(),  # type: ignore[arg-type]
        dp_rank=0,
    )
    monkeypatch.setattr(
        factory,
        "_prepare_with_runtime",
        lambda **_kwargs: (_ for _ in ()).throw(factory_module.WeightAccessPreparationFallback("artifact unavailable")),
    )
    owner = WeightConsumerOwner()

    factory.prepare_into(
        owner=owner,
        consumer_requirements_factory=_requirements,
        pipeline_builder=lambda transformer: transformer,
    )

    result = owner.preparation_result
    assert isinstance(result, FatalPreparationFailure)
    assert result.code == "preparation_cleanup_incomplete"
    assert result.cleanup_required is True
    assert runtime.close_calls == 1

    owner.close()
    assert owner.phase is WeightConsumerOwnerPhase.CLOSED
    assert runtime.close_calls == 2


def test_pipeline_and_format_identity_include_runtime_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bf16_config = SimpleNamespace(
        tf_model_config={},
        dtype=torch.bfloat16,
        task_type="t2va",
    )
    fp16_config = SimpleNamespace(
        tf_model_config={},
        dtype=torch.float16,
        task_type="t2va",
    )

    bf16_pipeline = minimax_h3_fp8._pipeline_spec(bf16_config)
    fp16_pipeline = minimax_h3_fp8._pipeline_spec(fp16_config)

    assert bf16_pipeline.model_config_digest != fp16_pipeline.model_config_digest
    assert bf16_pipeline.normalized_init_config["runtime_dtype"] == "bfloat16"
    assert fp16_pipeline.normalized_init_config["runtime_dtype"] == "float16"

    monkeypatch.setattr(
        minimax_h3_fp8.torch.cuda,
        "get_device_capability",
        lambda _index: (9, 0),
    )
    quant_config = SimpleNamespace(
        activation_scheme="dynamic",
        ignored_layers=[],
        store_dtype=None,
    )
    bf16_format = minimax_h3_fp8._weight_format_descriptor(
        quant_config,
        torch.device("cuda:0"),
        torch.bfloat16,
    )
    fp16_format = minimax_h3_fp8._weight_format_descriptor(
        quant_config,
        torch.device("cuda:0"),
        torch.float16,
    )

    assert bf16_format.semantic_fingerprint != fp16_format.semantic_fingerprint
    assert bf16_format.normalized_config["runtime_dtype"] == "bfloat16"
    assert fp16_format.normalized_config["runtime_dtype"] == "float16"
