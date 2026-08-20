# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loader boundary tests for the owner-published Host Weight Runtime path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.config.load import LoadConfig

from vllm_omni.diffusion.host_weight.evidence import (
    HostWeightResolutionEvidence,
)
from vllm_omni.diffusion.host_weight.formats.fp8_per_tensor import FORMAT_ID
from vllm_omni.diffusion.host_weight.ownership import (
    FatalPreparationFailure,
    LegacyReason,
    PreparedSessionReady,
    RetryablePreparationFailure,
    UseLegacy,
)
from vllm_omni.diffusion.host_weight.session_factory import (
    WeightAccessPreparationError,
    WeightAccessPreparationFallback,
)
from vllm_omni.diffusion.model_loader.diffusers_loader import (
    DiffusersPipelineLoader,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _Prepared:
    def __init__(self, pipeline: nn.Module) -> None:
        self.pipeline = pipeline
        self.rollback_count = 0

    def rollback(self) -> None:
        self.rollback_count += 1


class _Owner:
    """Small owner double that makes cleanup authority observable."""

    def __init__(self, *, close_failures: int = 0) -> None:
        self._result: object | None = None
        self.close_failures = close_failures
        self.close_calls = 0
        self.closed = False

    @property
    def preparation_result(self) -> object:
        if self._result is None:
            raise RuntimeError("preparation result is unavailable")
        return self._result

    @property
    def prepared_session(self) -> _Prepared:
        result = self.preparation_result
        if not isinstance(result, PreparedSessionReady):
            raise RuntimeError("preparation did not produce a ready session")
        return result.prepared_session  # type: ignore[return-value]

    def publish_preparation_result(self, result: object) -> None:
        if self._result is not None:
            raise RuntimeError("preparation result was already published")
        self._result = result

    def close(self) -> None:
        if self.closed:
            return
        self.close_calls += 1
        if self.close_failures:
            self.close_failures -= 1
            raise RuntimeError("injected owner cleanup failure")
        result = self._result
        if isinstance(result, PreparedSessionReady):
            result.prepared_session.rollback()
        self._result = None
        self.closed = True


class _ConsumerFactory:
    def __init__(self) -> None:
        self.requested_format_ids: list[str] = []
        self.requirement = object()

    def requirements(self, required_weight_format_id: str) -> object:
        self.requested_format_ids.append(required_weight_format_id)
        return self.requirement


class _MixedPipeline(nn.Module):
    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.transformer = transformer
        self.text_encoder = nn.Linear(2, 2, bias=False)
        self.weights_sources = (
            DiffusersPipelineLoader.ComponentSource(
                "unused",
                "transformer",
                None,
                prefix="transformer.",
            ),
            DiffusersPipelineLoader.ComponentSource(
                "unused",
                "text_encoder",
                None,
                prefix="text_encoder.",
            ),
        )

    def load_weights(self, weights):
        parameters = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            parameters[name].data.copy_(tensor)
            loaded.add(name)
        return loaded


class _ReadySessionFactory:
    last: _ReadySessionFactory | None = None
    evidence_override: object | None = None

    def __init__(self, **_kwargs) -> None:
        self.manifest = SimpleNamespace(tensors=(SimpleNamespace(tensor_id="weight"),))
        self.resolution_evidence = self.evidence_override or (
            HostWeightResolutionEvidence(
                runtime_mode="read_only",
                outcome="ready",
                events=("cache_hit", "ready"),
                artifact_key="a" * 64,
                artifact_compatibility_digest="b" * 64,
                resolution_path="mmap_hit",
                claim_role="cache_hit",
                cache_hit=True,
                generation_id="generation-1",
                backing_kind="runtime_mmap",
            )
        )
        self.owner: _Owner | None = None
        self.consumer_requirements: object | None = None
        self.prepared: _Prepared | None = None
        _ReadySessionFactory.last = self

    def prepare_into(
        self,
        *,
        owner: _Owner,
        consumer_requirements: object,
        pipeline_builder,
    ) -> None:
        self.owner = owner
        self.consumer_requirements = consumer_requirements
        transformer = nn.Linear(2, 2, bias=False)
        pipeline = pipeline_builder(transformer)
        assert pipeline.transformer is transformer
        self.prepared = _Prepared(pipeline)
        owner.publish_preparation_result(PreparedSessionReady(self.prepared))  # type: ignore[arg-type]


def _od_config(
    *,
    required: bool = False,
    quantization_config=None,
    runtime_mode: str = "read_only",
) -> SimpleNamespace:
    return SimpleNamespace(
        dtype=torch.float32,
        parallel_config=SimpleNamespace(use_hsdp=False),
        quantization_config=quantization_config,
        host_weight_runtime_mode=runtime_mode,
        host_weight_runtime_root="/unused",
        host_weight_runtime_required=required,
        enable_distributed_layerwise_offload=False,
    )


def _loader(
    *,
    required: bool = False,
    quantization_config=None,
    owner: _Owner | None = None,
    consumer_factory: _ConsumerFactory | None = None,
) -> tuple[DiffusersPipelineLoader, _Owner, _ConsumerFactory]:
    owner = owner or _Owner()
    consumer_factory = consumer_factory or _ConsumerFactory()
    loader = DiffusersPipelineLoader(
        LoadConfig(),
        _od_config(
            required=required,
            quantization_config=quantization_config,
        ),
        weight_consumer_owner=owner,  # type: ignore[arg-type]
        weight_consumer_factory=consumer_factory,  # type: ignore[arg-type]
    )
    return loader, owner, consumer_factory


def _install_ready_factory(monkeypatch) -> None:
    import vllm_omni.diffusion.host_weight.session_factory as factory_module

    _ReadySessionFactory.last = None
    _ReadySessionFactory.evidence_override = None
    monkeypatch.setattr(
        factory_module,
        "WeightAccessSessionFactory",
        _ReadySessionFactory,
    )


def _install_result_factory(
    monkeypatch,
    result: object,
    *,
    evidence: object | None = None,
) -> type:
    import vllm_omni.diffusion.host_weight.session_factory as factory_module

    class ResultFactory:
        manifest = None
        resolution_evidence = evidence

        def __init__(self, **_kwargs) -> None:
            pass

        def prepare_into(
            self,
            *,
            owner: _Owner,
            consumer_requirements: object,
            pipeline_builder,
        ) -> None:
            del consumer_requirements, pipeline_builder
            owner.publish_preparation_result(result)

    monkeypatch.setattr(factory_module, "WeightAccessSessionFactory", ResultFactory)
    return ResultFactory


def test_loader_requires_owner_and_consumer_factory_as_a_pair() -> None:
    owner = _Owner()
    consumer_factory = _ConsumerFactory()

    with pytest.raises(ValueError, match="must be provided together"):
        DiffusersPipelineLoader(
            LoadConfig(),
            _od_config(),
            weight_consumer_owner=owner,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="must be provided together"):
        DiffusersPipelineLoader(
            LoadConfig(),
            _od_config(),
            weight_consumer_factory=consumer_factory,  # type: ignore[arg-type]
        )

    legacy_loader = DiffusersPipelineLoader(
        LoadConfig(),
        _od_config(runtime_mode="disabled"),
    )
    assert legacy_loader.weight_consumer_owner is None
    assert legacy_loader.weight_consumer_factory is None


def test_hwr_prepares_into_owner_before_non_artifact_source_iteration(
    monkeypatch,
) -> None:
    _install_ready_factory(monkeypatch)
    loader, owner, consumer_factory = _loader()
    requested_prefixes: list[str] = []
    processed: list[torch.device] = []

    def initialize(_format, _device, *, model_init_kwargs=None, **_kwargs):
        assert model_init_kwargs is not None
        return _MixedPipeline(model_init_kwargs["transformer_override"])

    def weights(source, model=None):
        del model
        requested_prefixes.append(source.prefix)
        if source.prefix == "transformer.":
            pytest.fail("HWR transformer ComponentSource must remain unopened")
        yield "text_encoder.weight", torch.ones(2, 2)

    monkeypatch.setattr(loader, "_init_from_load_format", initialize)
    monkeypatch.setattr(loader, "_get_weights_iterator", weights)
    monkeypatch.setattr(
        loader,
        "_process_weights_after_loading",
        lambda _model, device: processed.append(device),
    )

    pipeline = loader._load_model_with_host_weight_runtime(
        target_device=torch.device("cpu"),
        producer_device=torch.device("cpu"),
        offload_after_quant=False,
    )

    assert requested_prefixes == ["text_encoder."]
    assert processed == [torch.device("cpu")]
    assert torch.equal(
        pipeline.text_encoder.weight,
        torch.ones_like(pipeline.text_encoder.weight),
    )
    factory = _ReadySessionFactory.last
    assert factory is not None and factory.prepared is not None
    assert factory.owner is owner
    assert factory.consumer_requirements is consumer_factory.requirement
    assert consumer_factory.requested_format_ids == [FORMAT_ID]
    assert owner.prepared_session is factory.prepared
    assert factory.prepared.rollback_count == 0
    assert owner.close_calls == 0
    assert not hasattr(loader, "prepared_weight_session")
    assert not hasattr(loader, "preparation_cleanup_handle")
    assert not hasattr(loader, "take_prepared_weight_session")
    assert not hasattr(loader, "take_preparation_cleanup_handle")

    evidence = loader.take_host_weight_runtime_evidence()
    assert evidence is not None
    assert evidence["events"] == ["cache_hit", "ready"]
    assert evidence["artifact_key"] == "a" * 64
    assert loader.take_host_weight_runtime_evidence() is None


def test_non_artifact_failure_leaves_ready_session_owned_until_owner_close(
    monkeypatch,
) -> None:
    _install_ready_factory(monkeypatch)
    loader, owner, _ = _loader()

    def initialize(_format, _device, *, model_init_kwargs=None, **_kwargs):
        assert model_init_kwargs is not None
        return _MixedPipeline(model_init_kwargs["transformer_override"])

    def fail_weights(source, model=None):
        del source, model
        raise RuntimeError("injected component failure")
        yield  # pragma: no cover

    monkeypatch.setattr(loader, "_init_from_load_format", initialize)
    monkeypatch.setattr(loader, "_get_weights_iterator", fail_weights)

    with pytest.raises(RuntimeError, match="injected component failure"):
        loader._load_model_with_host_weight_runtime(
            target_device=torch.device("cpu"),
            producer_device=torch.device("cpu"),
            offload_after_quant=False,
        )

    factory = _ReadySessionFactory.last
    assert factory is not None and factory.prepared is not None
    assert owner.prepared_session is factory.prepared
    assert factory.prepared.rollback_count == 0
    assert owner.close_calls == 0

    owner.close()
    assert factory.prepared.rollback_count == 1
    assert owner.closed


@pytest.mark.parametrize(
    ("result", "exception_type", "outcome", "code"),
    [
        pytest.param(
            UseLegacy(
                LegacyReason.OPTIONAL_ARTIFACT_UNAVAILABLE,
                "cache miss",
            ),
            WeightAccessPreparationFallback,
            "fallback",
            "optional_artifact_unavailable",
            id="legacy",
        ),
        pytest.param(
            RetryablePreparationFailure(
                "repository_busy",
                "try again",
                retry_after_s=0.5,
                cleanup_required=True,
            ),
            WeightAccessPreparationError,
            "retryable_failure",
            "repository_busy",
            id="retryable",
        ),
        pytest.param(
            FatalPreparationFailure(
                "manifest_invalid",
                "digest mismatch",
                cleanup_required=True,
            ),
            WeightAccessPreparationError,
            "fatal",
            "manifest_invalid",
            id="fatal",
        ),
    ],
)
def test_hwr_branches_on_owner_published_preparation_result(
    monkeypatch,
    result: object,
    exception_type: type[BaseException],
    outcome: str,
    code: str,
) -> None:
    _install_result_factory(monkeypatch, result)
    loader, owner, consumer_factory = _loader()

    with pytest.raises(exception_type, match=code):
        loader._load_model_with_host_weight_runtime(
            target_device=torch.device("cpu"),
            producer_device=torch.device("cpu"),
            offload_after_quant=False,
        )

    assert owner.preparation_result is result
    assert owner.close_calls == 0
    assert consumer_factory.requested_format_ids == [FORMAT_ID]
    evidence = loader.take_host_weight_runtime_evidence()
    assert evidence is not None
    assert evidence["outcome"] == outcome
    assert evidence["code"] == code


def test_retryable_preparation_never_enters_optional_ordinary_loading(
    monkeypatch,
) -> None:
    result = RetryablePreparationFailure(
        "repository_busy",
        "try again",
        retry_after_s=0.5,
        cleanup_required=True,
    )
    _install_result_factory(monkeypatch, result)
    loader, owner, _ = _loader(required=False)
    monkeypatch.setattr(
        loader,
        "_init_from_load_format",
        lambda *_args, **_kwargs: pytest.fail("retryable HWR preparation must not enter ordinary loading"),
    )

    with pytest.raises(WeightAccessPreparationError, match="repository_busy"):
        loader.load_model("cpu", device=torch.device("cuda", 0))

    assert owner.preparation_result is result
    assert owner.close_calls == 0
    assert loader.host_weight_runtime_fell_back is False


def test_optional_fallback_closes_owner_before_ordinary_online_quant_route(
    monkeypatch,
) -> None:
    online_quant = SimpleNamespace(
        data_type=None,
        is_checkpoint_quantized=False,
    )
    loader, owner, _ = _loader(quantization_config=online_quant)
    initialized_on: list[torch.device] = []
    processed_on: list[torch.device] = []
    load_calls: list[bool] = []

    def unavailable(**_kwargs):
        raise WeightAccessPreparationFallback("artifact_not_found")

    def initialize(_format, device, *_args, **_kwargs):
        assert owner.closed
        initialized_on.append(device)
        return nn.Linear(2, 2, bias=False)

    monkeypatch.setattr(loader, "_load_model_with_host_weight_runtime", unavailable)
    monkeypatch.setattr(loader, "_init_from_load_format", initialize)
    monkeypatch.setattr(loader, "_has_online_quant", lambda _model: False)
    monkeypatch.setattr(
        loader,
        "load_weights",
        lambda _model=None, *, stream_online_quant_to_cpu=False, **_kwargs: load_calls.append(
            stream_online_quant_to_cpu
        ),
    )
    monkeypatch.setattr(
        loader,
        "_process_weights_after_loading",
        lambda _model, device: processed_on.append(device),
    )
    monkeypatch.setattr(loader, "_apply_skip_softmax_calibration", lambda _model: None)

    model = loader.load_model(
        "cpu",
        device=torch.device("cuda", 0),
    )

    assert isinstance(model, nn.Linear)
    assert initialized_on == [torch.device("cuda")]
    assert processed_on == [torch.device("cuda")]
    assert load_calls == [True]
    assert owner.close_calls == 1
    assert loader.host_weight_runtime_fell_back is True


def test_optional_fallback_does_not_start_legacy_load_if_owner_close_fails(
    monkeypatch,
) -> None:
    owner = _Owner(close_failures=1)
    loader, _, _ = _loader(owner=owner)
    primary = WeightAccessPreparationFallback("artifact_not_found: cold miss")

    def unavailable(**_kwargs):
        raise primary

    monkeypatch.setattr(loader, "_load_model_with_host_weight_runtime", unavailable)
    monkeypatch.setattr(
        loader,
        "_init_from_load_format",
        lambda *_args, **_kwargs: pytest.fail("ordinary loading must not start while owner cleanup is retained"),
    )

    with pytest.raises(WeightAccessPreparationFallback) as exc_info:
        loader.load_model(
            "cpu",
            device=torch.device("cuda", 0),
        )

    assert exc_info.value is primary
    assert owner.close_calls == 1
    assert not owner.closed
    assert loader.host_weight_runtime_fell_back is False
    assert any("owner cleanup before ordinary-loader fallback" in note for note in primary.__notes__)

    owner.close()
    assert owner.close_calls == 2
    assert owner.closed


def test_optional_hsdp_fallback_uses_ordinary_hsdp_loader(monkeypatch) -> None:
    loader, owner, _ = _loader()
    loader.parallel_config.use_hsdp = True
    hsdp_calls: list[tuple[torch.device, str | None, object]] = []

    def unavailable(**_kwargs):
        raise WeightAccessPreparationFallback("unsupported_topology: HSDP")

    def load_with_hsdp(*, target_device, load_format, custom_pipeline_name):
        assert owner.closed
        hsdp_calls.append((target_device, load_format, custom_pipeline_name))
        return nn.Linear(2, 2, bias=False)

    monkeypatch.setattr(loader, "_load_model_with_host_weight_runtime", unavailable)
    monkeypatch.setattr(loader, "_load_model_with_hsdp", load_with_hsdp)
    monkeypatch.setattr(loader, "_apply_skip_softmax_calibration", lambda _model: None)

    model = loader.load_model(
        "cpu",
        device=torch.device("cuda", 0),
    )

    assert isinstance(model, nn.Linear)
    assert model.training is False
    assert hsdp_calls == [(torch.device("cuda", 0), "default", None)]
    assert owner.close_calls == 1
    assert loader.host_weight_runtime_fell_back is True


@pytest.mark.parametrize("failure_point", ["calibration", "eval"])
def test_post_load_failure_leaves_cleanup_in_owner(
    monkeypatch,
    failure_point: str,
) -> None:
    class TailFailureModel(nn.Linear):
        def eval(self):
            if failure_point == "eval":
                raise RuntimeError("injected eval failure")
            return super().eval()

    loader, owner, _ = _loader()
    model = TailFailureModel(2, 2, bias=False)
    prepared = _Prepared(model)

    def load_with_hwr(**_kwargs):
        owner.publish_preparation_result(PreparedSessionReady(prepared))  # type: ignore[arg-type]
        return model

    def calibrate(_model):
        if failure_point == "calibration":
            raise RuntimeError("injected calibration failure")

    monkeypatch.setattr(loader, "_load_model_with_host_weight_runtime", load_with_hwr)
    monkeypatch.setattr(loader, "_apply_skip_softmax_calibration", calibrate)

    with pytest.raises(RuntimeError, match=f"injected {failure_point} failure"):
        loader.load_model(
            "cpu",
            device=torch.device("cuda", 0),
        )

    assert owner.prepared_session is prepared
    assert prepared.rollback_count == 0
    assert owner.close_calls == 0

    owner.close()
    assert prepared.rollback_count == 1


def test_ready_owner_survives_evidence_serialization_failure(monkeypatch) -> None:
    class BrokenEvidence:
        def to_dict(self):
            raise SystemExit("injected evidence serialization failure")

    _install_ready_factory(monkeypatch)
    loader, owner, _ = _loader()
    monkeypatch.setattr(
        _ReadySessionFactory,
        "evidence_override",
        BrokenEvidence(),
    )

    def initialize(_format, _device, *, model_init_kwargs=None, **_kwargs):
        assert model_init_kwargs is not None
        return _MixedPipeline(model_init_kwargs["transformer_override"])

    monkeypatch.setattr(loader, "_init_from_load_format", initialize)

    with pytest.raises(SystemExit, match="evidence serialization failure"):
        loader._load_model_with_host_weight_runtime(
            target_device=torch.device("cpu"),
            producer_device=torch.device("cpu"),
            offload_after_quant=False,
        )

    factory = _ReadySessionFactory.last
    assert factory is not None and factory.prepared is not None
    assert owner.prepared_session is factory.prepared
    assert factory.prepared.rollback_count == 0

    owner.close()
    assert factory.prepared.rollback_count == 1


def test_required_hwr_does_not_close_owner_or_enter_ordinary_loader(
    monkeypatch,
) -> None:
    result = UseLegacy(
        LegacyReason.OPTIONAL_ARTIFACT_UNAVAILABLE,
        "artifact not found",
    )
    _install_result_factory(monkeypatch, result)
    loader, owner, _ = _loader(required=True)
    loader.parallel_config.use_hsdp = True
    monkeypatch.setattr(
        loader,
        "_load_model_with_hsdp",
        lambda *_args, **_kwargs: pytest.fail("required HWR must fail closed"),
    )

    with pytest.raises(
        WeightAccessPreparationError,
        match="optional_artifact_unavailable",
    ) as exc_info:
        loader.load_model(
            "cpu",
            device=torch.device("cuda", 0),
        )

    assert not isinstance(exc_info.value, WeightAccessPreparationFallback)
    assert owner.preparation_result is result
    assert owner.close_calls == 0
    assert loader.host_weight_runtime_fell_back is False
    evidence = loader.take_host_weight_runtime_evidence()
    assert evidence is not None
    assert evidence["outcome"] == "fatal"
    assert evidence["code"] == "optional_artifact_unavailable"

    owner.close()
    assert owner.close_calls == 1
