# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Built-in pluggability seam between HWR sessions and offload strategies."""

from __future__ import annotations

from enum import Enum

import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.host_weight.ownership import (
    WeightConsumer,
    WeightConsumerOwner,
)
from vllm_omni.diffusion.host_weight.session import (
    HostCopyMode,
    PreparedWeightAccessSession,
    SessionRequirements,
)
from vllm_omni.diffusion.host_weight.transfer import TransferPlanKind
from vllm_omni.host_weight_runtime import (
    AccessFeature,
    AccessRequirements,
    BackingKind,
)

from .base import OffloadConfig, OffloadStrategy


class ConsumerLifecyclePhase(str, Enum):
    PREPARED = "prepared"
    ACTIVE = "active"
    QUIESCING = "quiescing"
    CLOSED = "closed"


def _add_cleanup_note(
    primary: BaseException,
    action: str,
    cleanup: BaseException,
) -> None:
    try:
        primary.add_note(f"{action} also failed: {type(cleanup).__name__}: {cleanup}")
    except BaseException:
        pass


class BuiltinOffloadWeightConsumer(WeightConsumer):
    """Lazy adapter that leaves scheduling in the selected existing backend.

    Candidate construction allocates no streams, slots, hooks, or process
    groups.  The concrete backend is created only from
    :meth:`enable_transactionally`, after the composition-owned owner has
    injected the exact prepared session.
    """

    def __init__(
        self,
        *,
        config: OffloadConfig,
        device: torch.device,
        pipeline: object,
    ) -> None:
        self._config = config
        self._device = device
        self._pipeline = pipeline
        self._prepared: PreparedWeightAccessSession | None = None
        self._backend: object | None = None
        self._phase = ConsumerLifecyclePhase.PREPARED

    @property
    def phase(self) -> ConsumerLifecyclePhase:
        return self._phase

    @property
    def backend(self) -> object:
        if self._backend is None:
            raise RuntimeError("offload backend has not been constructed")
        return self._backend

    def host_weight_diagnostics(self) -> dict[str, object]:
        """Compose strategy diagnostics without exposing ownership-bearing state."""

        backend = self._backend
        if backend is None:
            return {
                "pinned_slot_budget_bytes": 0,
                "idle_state": {
                    "outstanding_units": 0,
                    "bindings": 0,
                    "resident_bindings": 0,
                    "total_bindings": 0,
                    "events": 0,
                },
            }
        backend_snapshot = getattr(backend, "host_weight_diagnostics", None)
        if not callable(backend_snapshot):
            raise RuntimeError(f"{type(backend).__name__} does not expose host-weight diagnostics")
        snapshot = dict(backend_snapshot())
        session = getattr(backend, "_weight_session", None)
        if session is None:
            session_state: dict[str, object] = {
                "outstanding_units": 0,
                "bindings": 0,
            }
        else:
            session_state = dict(session.idle_state())
        session_state["events"] = int(snapshot.pop("events", 0))
        snapshot["idle_state"] = session_state
        return snapshot

    def adopt_prepared_session(self, prepared: PreparedWeightAccessSession) -> None:
        # Owner-only valid input: an allocation-free exact-reference store.
        if self._prepared is not None or self._backend is not None:
            raise RuntimeError("weight consumer already owns a prepared session")
        self._prepared = prepared

    def _build_backend(self, prepared: PreparedWeightAccessSession) -> object:
        strategy = self._config.strategy
        if strategy is OffloadStrategy.MODEL_LEVEL:
            from .sequential_backend import ModelLevelOffloadBackend

            return ModelLevelOffloadBackend(
                self._config,
                self._device,
                prepared_weight_session=prepared,
            )
        if strategy is OffloadStrategy.LAYER_WISE:
            from .layerwise_backend import LayerWiseOffloadBackend

            return LayerWiseOffloadBackend(
                self._config,
                self._device,
                prepared_weight_session=prepared,
            )
        if strategy is OffloadStrategy.DISTRIBUTED_LAYER_WISE:
            from .distributed_layerwise_backend import (
                DistributedLayerwiseOffloadBackend,
            )

            return DistributedLayerwiseOffloadBackend(
                self._config,
                self._device,
                host_weight_plan=None,
                prepared_weight_session=prepared,
            )
        raise RuntimeError(f"offload strategy {strategy.value!r} is not an HWR weight consumer")

    def enable_transactionally(self) -> None:
        if self._phase is ConsumerLifecyclePhase.ACTIVE:
            return
        if self._phase is not ConsumerLifecyclePhase.PREPARED:
            raise RuntimeError(f"cannot enable weight consumer in phase {self._phase.value}")
        prepared = self._prepared
        if prepared is None:
            raise RuntimeError("weight consumer has no adopted prepared session")
        try:
            backend = self._build_backend(prepared)
            # Internal authority marker: from here disable delegates to the
            # exact backend, which supports cleanup before enable completes.
            self._backend = backend
            self._prepared = None
            enable = getattr(backend, "enable")
            enable(self._pipeline)
            self._phase = ConsumerLifecyclePhase.ACTIVE
        except BaseException as primary:
            try:
                self.disable()
            except BaseException as cleanup:
                _add_cleanup_note(primary, "disabling the failed HWR weight consumer", cleanup)
            raise

    def disable(self) -> None:
        if self._phase is ConsumerLifecyclePhase.CLOSED:
            return
        self._phase = ConsumerLifecyclePhase.QUIESCING
        backend = self._backend
        if backend is not None:
            disable = getattr(backend, "disable")
            disable()
            if self._backend is backend:
                self._backend = None
            self._pipeline = None
            self._phase = ConsumerLifecyclePhase.CLOSED
            return
        prepared = self._prepared
        if prepared is not None:
            prepared.rollback()
            if self._prepared is prepared:
                self._prepared = None
        self._pipeline = None
        self._phase = ConsumerLifecyclePhase.CLOSED


class BuiltinOffloadWeightConsumerFactory:
    """Pure built-in consumer selection plus owner-only candidate publication."""

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        device: torch.device,
    ) -> None:
        self._config = OffloadConfig.from_od_config(od_config)
        self._device = device
        if self._config.strategy is OffloadStrategy.NONE:
            raise ValueError("Host Weight Runtime requires one offload strategy")
        if self._config.strategy is OffloadStrategy.DISTRIBUTED_LAYER_WISE and self._config.dlo_use_allgather:
            raise ValueError("Host Weight Runtime v1 supports only DLO no-AllGather")

    def requirements(self, required_weight_format_id: str) -> SessionRequirements:
        if not required_weight_format_id:
            raise ValueError("required weight format ID must not be empty")
        plan_kind = (
            TransferPlanKind.COMPONENT
            if self._config.strategy is OffloadStrategy.MODEL_LEVEL
            else TransferPlanKind.BLOCKS_PLUS_RESIDENT
        )
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
            required_transfer_plan_kind=plan_kind,
            required_weight_format_id=required_weight_format_id,
            host_copy_mode=HostCopyMode.SYNCHRONOUS,
        )

    def create_into(
        self,
        *,
        owner: WeightConsumerOwner,
        pipeline: object,
    ) -> None:
        prepared = owner.prepared_session
        if prepared.pipeline is not pipeline:
            raise RuntimeError("consumer pipeline differs from the owner-held prepared pipeline")
        candidate = BuiltinOffloadWeightConsumer(
            config=self._config,
            device=self._device,
            pipeline=pipeline,
        )
        owner.publish_consumer(candidate)


__all__ = [
    "BuiltinOffloadWeightConsumer",
    "BuiltinOffloadWeightConsumerFactory",
    "ConsumerLifecyclePhase",
]
