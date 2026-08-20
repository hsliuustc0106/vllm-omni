# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration and ownership tests for the host-weight session factory."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config.load import LoadConfig

from vllm_omni.diffusion.host_weight import session_factory as factory_module
from vllm_omni.diffusion.host_weight.ownership import (
    FatalPreparationFailure,
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
    return config


def _requirements() -> SessionRequirements:
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
        required_weight_format_id=factory_module.FORMAT_ID,
        host_copy_mode=HostCopyMode.SYNCHRONOUS,
    )


def _weight_format() -> WeightFormatDescriptor:
    semantic = canonical_digest("format")
    normalized_config: dict[str, object] = {}
    kernel_identity: dict[str, object] = {}
    plan_digest = derive_weight_format_plan_digest(
        format_id=factory_module.FORMAT_ID,
        adapter_abi=factory_module.FORMAT_ADAPTER_ABI,
        semantic_fingerprint=semantic,
        format_recipe_schema_version=factory_module.FORMAT_RECIPE_SCHEMA_VERSION,
        target_module_type_id=factory_module.TARGET_MODULE_TYPE_ID,
        normalized_config=normalized_config,
        kernel_identity=kernel_identity,
    )
    return WeightFormatDescriptor(
        format_id=factory_module.FORMAT_ID,
        adapter_abi=factory_module.FORMAT_ADAPTER_ABI,
        semantic_fingerprint=semantic,
        format_plan_digest=plan_digest,
        format_recipe_schema_version=factory_module.FORMAT_RECIPE_SCHEMA_VERSION,
        target_module_type_id=factory_module.TARGET_MODULE_TYPE_ID,
        normalized_config=normalized_config,
        kernel_identity=kernel_identity,
    )


def _patch_preparation_preamble(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory_module, "_validate_scope", lambda *_args: None)
    monkeypatch.setattr(
        factory_module,
        "_prepare_diffusion_quant_config",
        lambda *_args: None,
    )
    monkeypatch.setattr(factory_module, "_resolved_fp8_config", lambda _config: object())
    monkeypatch.setattr(
        factory_module,
        "resolve_minimax_h3_transformer_source",
        lambda _config: SimpleNamespace(source_fingerprint=canonical_digest("source")),
    )
    monkeypatch.setattr(
        factory_module,
        "_weight_format_descriptor",
        lambda *_args: _weight_format(),
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
        consumer_requirements=_requirements(),
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
        consumer_requirements=_requirements(),
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
        consumer_requirements=_requirements(),
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
        consumer_requirements=_requirements(),
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

    bf16_pipeline = factory_module._pipeline_spec(bf16_config)
    fp16_pipeline = factory_module._pipeline_spec(fp16_config)

    assert bf16_pipeline.model_config_digest != fp16_pipeline.model_config_digest
    assert bf16_pipeline.normalized_init_config["runtime_dtype"] == "bfloat16"
    assert fp16_pipeline.normalized_init_config["runtime_dtype"] == "float16"

    monkeypatch.setattr(
        factory_module.torch.cuda,
        "get_device_capability",
        lambda _index: (9, 0),
    )
    quant_config = SimpleNamespace(
        activation_scheme="dynamic",
        ignored_layers=[],
        store_dtype=None,
    )
    bf16_format = factory_module._weight_format_descriptor(
        quant_config,
        torch.device("cuda:0"),
        torch.bfloat16,
    )
    fp16_format = factory_module._weight_format_descriptor(
        quant_config,
        torch.device("cuda:0"),
        torch.float16,
    )

    assert bf16_format.semantic_fingerprint != fp16_format.semantic_fingerprint
    assert bf16_format.normalized_config["runtime_dtype"] == "bfloat16"
    assert fp16_format.normalized_config["runtime_dtype"] == "float16"
