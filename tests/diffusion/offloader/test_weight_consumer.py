# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for the built-in HWR-to-offloader adapter."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.host_weight.session import (
    HostCopyMode,
    PreparedWeightAccessSession,
)
from vllm_omni.diffusion.host_weight.transfer import TransferPlanKind
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.weight_consumer import (
    BuiltinOffloadWeightConsumer,
    BuiltinOffloadWeightConsumerFactory,
    ConsumerLifecyclePhase,
    HostWeightOffloadBackend,
)
from vllm_omni.host_weight_runtime import AccessFeature, BackingKind


def _od_config(
    *,
    model_level: bool = False,
    layer_wise: bool = False,
    distributed_layer_wise: bool = False,
    dlo_use_allgather: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        enable_cpu_offload=model_level,
        enable_layerwise_offload=layer_wise,
        enable_distributed_layerwise_offload=distributed_layer_wise,
        pin_cpu_memory=True,
        dlo_use_allgather=dlo_use_allgather,
        dlo_resident_layers=0,
        parallel_config=SimpleNamespace(
            data_parallel_size=4,
            sequence_parallel_size=1,
            use_hsdp=False,
        ),
    )


@pytest.mark.parametrize(
    ("config", "expected_plan_kind"),
    [
        (_od_config(model_level=True), TransferPlanKind.COMPONENT),
        (_od_config(layer_wise=True), TransferPlanKind.BLOCKS_PLUS_RESIDENT),
        (
            _od_config(distributed_layer_wise=True),
            TransferPlanKind.BLOCKS_PLUS_RESIDENT,
        ),
    ],
)
def test_factory_negotiates_one_exact_transfer_plan(
    config: SimpleNamespace,
    expected_plan_kind: TransferPlanKind,
) -> None:
    factory = BuiltinOffloadWeightConsumerFactory(config, torch.device("cpu"))

    requirements = factory.requirements("fp8-per-tensor")

    assert requirements.required_transfer_plan_kind is expected_plan_kind
    assert requirements.required_weight_format_id == "fp8-per-tensor"
    assert requirements.host_copy_mode is HostCopyMode.SYNCHRONOUS
    assert requirements.access.accepted_backings == frozenset({BackingKind.RUNTIME_MMAP})
    assert requirements.access.required_features == frozenset(
        {
            AccessFeature.COMPLETE_TENSOR_READ,
            AccessFeature.SHARED_PAGES,
        }
    )


def test_factory_rejects_non_consumer_and_dlo_allgather_modes() -> None:
    with pytest.raises(ValueError, match="one offload strategy"):
        BuiltinOffloadWeightConsumerFactory(_od_config(), torch.device("cpu"))

    with pytest.raises(ValueError, match="DLO no-AllGather"):
        BuiltinOffloadWeightConsumerFactory(
            _od_config(
                distributed_layer_wise=True,
                dlo_use_allgather=True,
            ),
            torch.device("cpu"),
        )


def test_factory_publishes_an_allocation_free_candidate() -> None:
    pipeline = object()
    prepared = SimpleNamespace(pipeline=pipeline)

    class Owner:
        prepared_session = prepared
        candidate: object | None = None

        def publish_consumer(self, candidate: object) -> None:
            self.candidate = candidate

    owner = Owner()
    factory = BuiltinOffloadWeightConsumerFactory(
        _od_config(model_level=True),
        torch.device("cpu"),
    )

    factory.create_into(owner=owner, pipeline=pipeline)  # type: ignore[arg-type]

    candidate = owner.candidate
    assert isinstance(candidate, BuiltinOffloadWeightConsumer)
    assert candidate.phase is ConsumerLifecyclePhase.PREPARED
    with pytest.raises(RuntimeError, match="has not been constructed"):
        _ = candidate.backend


def test_factory_rejects_a_different_pipeline_identity() -> None:
    class Owner:
        prepared_session = SimpleNamespace(pipeline=object())

        def publish_consumer(self, _candidate: object) -> None:
            raise AssertionError("a mismatched candidate must not be published")

    factory = BuiltinOffloadWeightConsumerFactory(
        _od_config(model_level=True),
        torch.device("cpu"),
    )

    with pytest.raises(RuntimeError, match="pipeline differs"):
        factory.create_into(owner=Owner(), pipeline=object())  # type: ignore[arg-type]


class _Prepared:
    def __init__(self, *, rollback_errors: list[BaseException] | None = None) -> None:
        self.rollback_errors = list(rollback_errors or ())
        self.rollback_calls = 0

    def rollback(self) -> None:
        self.rollback_calls += 1
        if self.rollback_errors:
            raise self.rollback_errors.pop(0)


class _Backend:
    def __init__(
        self,
        *,
        adopt_error: BaseException | None = None,
        enable_error: BaseException | None = None,
        session_state: dict[str, object] | None = None,
    ) -> None:
        self.adopt_error = adopt_error
        self.enable_error = enable_error
        self.adopt_attempts: list[PreparedWeightAccessSession] = []
        self.prepared: PreparedWeightAccessSession | None = None
        self.enable_calls: list[object] = []
        self.disable_calls = 0
        self.rollback_sessions: list[PreparedWeightAccessSession] = []
        self.session_state = session_state

    def adopt_prepared_session(self, prepared: PreparedWeightAccessSession) -> None:
        self.adopt_attempts.append(prepared)
        if self.adopt_error is not None:
            raise self.adopt_error
        if self.prepared is not None:
            raise RuntimeError("fake backend already owns a prepared session")
        self.prepared = prepared

    def enable(self, pipeline: object) -> None:
        self.enable_calls.append(pipeline)
        if self.enable_error is not None:
            raise self.enable_error

    def disable(self) -> None:
        self.disable_calls += 1
        prepared = self.prepared
        if prepared is not None:
            prepared.rollback()
            self.rollback_sessions.append(prepared)
            if self.prepared is prepared:
                self.prepared = None

    def is_enabled(self) -> bool:
        return bool(self.enable_calls) and self.prepared is not None

    def host_weight_diagnostics(self) -> dict[str, int]:
        return {
            "pinned_slot_budget_bytes": 4096,
            "events": 2,
        }

    def host_weight_session_idle_state(self) -> dict[str, object]:
        if self.session_state is None:
            return {
                "outstanding_units": 0,
                "bindings": 0,
                "resident_bindings": 0,
                "total_bindings": 0,
            }
        return dict(self.session_state)


class _InjectedConsumer(BuiltinOffloadWeightConsumer):
    def __init__(self, backend: _Backend, pipeline: object) -> None:
        super().__init__(
            config=OffloadConfig(OffloadStrategy.MODEL_LEVEL),
            device=torch.device("cpu"),
            pipeline=pipeline,
        )
        self.injected_backend = backend

    def _build_backend(self) -> HostWeightOffloadBackend:
        return self.injected_backend


def test_consumer_enable_and_disable_are_transactional() -> None:
    pipeline = object()
    backend = _Backend()
    prepared = _Prepared()
    consumer = _InjectedConsumer(backend, pipeline)
    consumer.adopt_prepared_session(prepared)  # type: ignore[arg-type]

    consumer.enable_transactionally()

    assert consumer.phase is ConsumerLifecyclePhase.ACTIVE
    assert consumer.backend is backend
    assert consumer.backend.is_enabled()
    assert backend.adopt_attempts == [prepared]
    assert backend.prepared is prepared
    assert backend.enable_calls == [pipeline]
    assert prepared.rollback_calls == 0

    consumer.disable()
    consumer.disable()

    assert consumer.phase is ConsumerLifecyclePhase.CLOSED
    assert backend.disable_calls == 1
    assert backend.rollback_sessions == [prepared]
    assert prepared.rollback_calls == 1


def test_consumer_adoption_failure_rolls_back_the_caller_owned_session() -> None:
    failure = RuntimeError("injected adoption failure")
    backend = _Backend(adopt_error=failure)
    prepared = _Prepared()
    consumer = _InjectedConsumer(backend, object())
    consumer.adopt_prepared_session(prepared)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="injected adoption failure") as raised:
        consumer.enable_transactionally()

    assert raised.value is failure
    assert consumer.phase is ConsumerLifecyclePhase.CLOSED
    assert backend.adopt_attempts == [prepared]
    assert backend.prepared is None
    assert backend.enable_calls == []
    assert backend.disable_calls == 0
    assert prepared.rollback_calls == 1


def test_consumer_enable_failure_rolls_back_the_backend_owned_session() -> None:
    failure = RuntimeError("injected enable failure")
    backend = _Backend(enable_error=failure)
    prepared = _Prepared()
    consumer = _InjectedConsumer(backend, object())
    consumer.adopt_prepared_session(prepared)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="injected enable failure") as raised:
        consumer.enable_transactionally()

    assert raised.value is failure
    assert consumer.phase is ConsumerLifecyclePhase.CLOSED
    assert backend.adopt_attempts == [prepared]
    assert backend.disable_calls == 1
    assert backend.rollback_sessions == [prepared]
    assert prepared.rollback_calls == 1


def test_consumer_enable_cleanup_failure_retains_backend_for_retry() -> None:
    failure = RuntimeError("injected enable failure")
    cleanup_failure = RuntimeError("injected cleanup failure")
    backend = _Backend(enable_error=failure)
    prepared = _Prepared(
        rollback_errors=[cleanup_failure],
    )
    consumer = _InjectedConsumer(backend, object())
    consumer.adopt_prepared_session(prepared)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="injected enable failure") as raised:
        consumer.enable_transactionally()

    assert raised.value is failure
    assert any("injected cleanup failure" in note for note in raised.value.__notes__)
    assert consumer.phase is ConsumerLifecyclePhase.QUIESCING
    assert consumer.backend is backend
    assert backend.prepared is prepared
    assert prepared.rollback_calls == 1

    consumer.disable()

    assert consumer.phase is ConsumerLifecyclePhase.CLOSED
    assert backend.disable_calls == 2
    assert backend.rollback_sessions == [prepared]
    assert prepared.rollback_calls == 2
    with pytest.raises(RuntimeError, match="has not been constructed"):
        _ = consumer.backend


def test_consumer_rolls_back_an_unpublished_prepared_session() -> None:
    prepared = _Prepared()
    consumer = BuiltinOffloadWeightConsumer(
        config=OffloadConfig(OffloadStrategy.MODEL_LEVEL),
        device=torch.device("cpu"),
        pipeline=object(),
    )
    consumer.adopt_prepared_session(prepared)  # type: ignore[arg-type]

    consumer.disable()

    assert consumer.phase is ConsumerLifecyclePhase.CLOSED
    assert prepared.rollback_calls == 1


def test_consumer_composes_backend_and_session_diagnostics() -> None:
    backend = _Backend(
        session_state={
            "state": "active",
            "outstanding_units": 0,
            "bindings": 0,
        }
    )
    consumer = _InjectedConsumer(backend, object())
    consumer.adopt_prepared_session(_Prepared())  # type: ignore[arg-type]
    consumer.enable_transactionally()

    assert consumer.host_weight_diagnostics() == {
        "pinned_slot_budget_bytes": 4096,
        "idle_state": {
            "state": "active",
            "outstanding_units": 0,
            "bindings": 0,
            "events": 2,
        },
    }
