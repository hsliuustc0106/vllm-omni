# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import chain
from types import MappingProxyType
from typing import Any

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.diffusion.host_weight.session import (
    DetachMode,
    DeviceBindingLifetime,
    PreparedWeightAccessSession,
    ReleaseTarget,
    UnitReadRequest,
    WeightAccessSession,
)
from vllm_omni.diffusion.host_weight.transfer import TransferPlanKind, UnitKind
from vllm_omni.platforms import current_omni_platform

from .base import OffloadBackend, OffloadConfig
from .module_collector import ModuleDiscovery

logger = init_logger(__name__)


def _report_cleanup_failure(
    primary_error: BaseException,
    action: str,
    cleanup_error: BaseException,
) -> None:
    """Report cleanup failure without replacing the primary exception."""

    try:
        note = f"{action} also failed: {type(cleanup_error).__name__}: {cleanup_error}"
    except BaseException:
        note = f"{action} also failed"
    try:
        primary_error.add_note(note)
    except BaseException:
        pass
    try:
        logger.error(
            "%s while handling %s",
            note,
            type(primary_error).__name__,
            exc_info=(
                type(cleanup_error),
                cleanup_error,
                cleanup_error.__traceback__,
            ),
        )
    except BaseException:
        pass


class HostWeightLayerwiseError(RuntimeError):
    """Raised when the immutable host source cannot safely serve a block ring."""


class _SessionTeardownPhase(Enum):
    ACTIVE = "active"
    QUIESCED = "quiesced"
    CLOSED = "closed"


@dataclass
class _HostWeightBlockBinding:
    block_path: str
    block: nn.Module
    unit: Any
    device_binding: Any | None = None
    device_planes: Mapping[Any, torch.Tensor] | None = None


@dataclass
class _WeightSessionHandle:
    session: WeightAccessSession | None = None
    fail_closed_callback: Callable[[], None] | None = None

    def require(self) -> WeightAccessSession:
        if self.session is None:
            raise HostWeightLayerwiseError("host-weight session is not committed")
        return self.session

    def fail_closed(self, primary_error: BaseException) -> None:
        """Abort the owning backend without replacing the hook failure."""
        callback = self.fail_closed_callback
        if callback is None:
            return
        try:
            callback()
        except BaseException as cleanup_error:
            _report_cleanup_failure(
                primary_error,
                "layerwise host-weight terminal cleanup",
                cleanup_error,
            )


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _pinned_cpu_storage_bytes(tensors: Iterable[torch.Tensor]) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0
    for tensor in tensors:
        if tensor.device.type != "cpu" or not tensor.is_pinned():
            continue
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        if key in seen:
            continue
        seen.add(key)
        total += storage.nbytes()
    return total


def _incomplete_event_count(events: Iterable[Any | None]) -> int:
    seen: set[int] = set()
    count = 0
    for event in events:
        if event is None or id(event) in seen:
            continue
        seen.add(id(event))
        query = getattr(event, "query", None)
        if callable(query):
            try:
                if bool(query()):
                    continue
            except BaseException:
                pass
        count += 1
    return count


def _resolve_execution_module(target: nn.Module, module_path: object) -> nn.Module:
    relative_path = str(module_path)
    if relative_path == ".":
        return target
    try:
        return target.get_submodule(relative_path)
    except AttributeError as exc:
        raise HostWeightLayerwiseError(
            f"selected transfer plan references missing target-relative module {relative_path!r}"
        ) from exc


def _validate_session_capabilities(prepared: PreparedWeightAccessSession) -> None:
    capabilities = prepared.capabilities
    plan = prepared.transfer_plan
    if _enum_value(plan.plan_kind) != TransferPlanKind.BLOCKS_PLUS_RESIDENT.value:
        raise HostWeightLayerwiseError("layerwise offload requires the 'blocks_plus_resident' transfer plan")
    if _enum_value(capabilities.selected_transfer_plan_kind) != TransferPlanKind.BLOCKS_PLUS_RESIDENT.value:
        raise HostWeightLayerwiseError("host-weight capabilities select a different transfer plan kind")
    if capabilities.selected_transfer_plan_id != plan.plan_id:
        raise HostWeightLayerwiseError("host-weight capabilities select a different transfer plan ID")
    supported_kinds = {_enum_value(kind) for kind in capabilities.unit_kinds}
    if UnitKind.BLOCK.value not in supported_kinds:
        raise HostWeightLayerwiseError("host-weight session does not support block transfer units")
    access_features = {_enum_value(feature) for feature in capabilities.access_features}
    if "complete_tensor_read" not in access_features:
        raise HostWeightLayerwiseError("host-weight session does not support complete-tensor reads")
    if _enum_value(capabilities.host_copy_mode) != "synchronous":
        raise HostWeightLayerwiseError("layerwise offload requires synchronous host copies")


class _HostWeightStagingPool:
    """Two caller-owned pinned host slots shared by every layerwise hook.

    The immutable source copies synchronously into a slot.  The slot is then
    protected by the H2D completion event until it can be reused.  Capacity is
    the per-dtype maximum across all prepared transfer units, not the sum of
    their sizes.
    """

    _NUM_SLOTS = 2

    def __init__(self, units: Sequence[Any], *, pin_memory: bool) -> None:
        if not units:
            raise HostWeightLayerwiseError("host-weight layerwise offload requires at least one transfer unit")

        capacities: dict[torch.dtype, int] = {}
        for unit in units:
            totals: dict[torch.dtype, int] = {}
            for plane in unit.planes:
                totals[plane.dtype] = totals.get(plane.dtype, 0) + plane.storage_numel
            for dtype, total in totals.items():
                capacities[dtype] = max(capacities.get(dtype, 0), total)
        if not capacities:
            raise HostWeightLayerwiseError("host-weight transfer units contain no dtype planes")

        self.capacities = MappingProxyType(capacities)
        self.slots: list[dict[torch.dtype, torch.Tensor]] = [
            {
                dtype: torch.empty(
                    numel,
                    dtype=dtype,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                for dtype, numel in capacities.items()
            }
            for _ in range(self._NUM_SLOTS)
        ]
        self.events: list[Any | None] = [None] * self._NUM_SLOTS
        self._pending_units: list[Any] = []
        self._next_slot = 0
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @staticmethod
    def _synchronize_event(event: Any) -> None:
        synchronize = getattr(event, "synchronize", None)
        if callable(synchronize):
            synchronize()
        else:
            current_omni_platform.synchronize()

    def stage(
        self,
        session: WeightAccessSession,
        binding: _HostWeightBlockBinding,
    ) -> tuple[int, dict[Any, torch.Tensor]]:
        if self._closed:
            raise HostWeightLayerwiseError("host-weight staging pool is terminal")

        slot = self._next_slot
        self._next_slot = (slot + 1) % self._NUM_SLOTS
        previous_copy = self.events[slot]
        if previous_copy is not None:
            self._synchronize_event(previous_copy)
            self.events[slot] = None

        offsets: dict[torch.dtype, int] = {}
        destinations: dict[Any, torch.Tensor] = {}
        for plane in binding.unit.planes:
            offset = offsets.get(plane.dtype, 0)
            destinations[plane.plane_id] = self.slots[slot][plane.dtype][offset : offset + plane.storage_numel]
            offsets[plane.dtype] = offset + plane.storage_numel

        prepared_unit = session.open_unit(UnitReadRequest(unit_id=binding.unit.unit_id))
        self._pending_units.append(prepared_unit)
        try:
            publish = getattr(prepared_unit, "publish", None)
            if callable(publish):
                publish()
            if prepared_unit.layout != binding.unit:
                raise HostWeightLayerwiseError(f"open_unit returned the wrong layout for {binding.unit.unit_id!r}")
            prepared_unit.copy_into(destinations)
        except BaseException as primary_error:
            try:
                prepared_unit.close()
            except BaseException as cleanup_error:
                _report_cleanup_failure(
                    primary_error,
                    "closing the failed layerwise weight read",
                    cleanup_error,
                )
            else:
                self._pending_units.remove(prepared_unit)
            try:
                self.close()
            except BaseException as cleanup_error:
                _report_cleanup_failure(
                    primary_error,
                    "closing layerwise host staging",
                    cleanup_error,
                )
            raise
        try:
            prepared_unit.close()
        except BaseException as primary_error:
            try:
                self.close()
            except BaseException as cleanup_error:
                _report_cleanup_failure(
                    primary_error,
                    "closing layerwise host staging",
                    cleanup_error,
                )
            raise
        self._pending_units.remove(prepared_unit)
        return slot, destinations

    def record_copy(self, slot: int, event: Any) -> None:
        if self._closed:
            raise HostWeightLayerwiseError("cannot record a copy on a terminal staging pool")
        if slot < 0 or slot >= self._NUM_SLOTS:
            raise HostWeightLayerwiseError(f"invalid host staging slot {slot}")
        if self.events[slot] is not None:
            raise HostWeightLayerwiseError(f"host staging slot {slot} already has an in-flight copy")
        self.events[slot] = event

    def close(self) -> None:
        if self._closed:
            return
        first_error: BaseException | None = None
        for prepared_unit in tuple(self._pending_units):
            try:
                prepared_unit.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                else:
                    _report_cleanup_failure(
                        first_error,
                        "closing another pending layerwise weight read",
                        exc,
                    )
            else:
                self._pending_units.remove(prepared_unit)
        for slot, event in enumerate(self.events):
            if event is not None:
                try:
                    self._synchronize_event(event)
                except BaseException as exc:  # pragma: no cover - platform failure
                    if first_error is None:
                        first_error = exc
                    else:
                        _report_cleanup_failure(
                            first_error,
                            "synchronizing another layerwise staging event",
                            exc,
                        )
                else:
                    self.events[slot] = None
        if not self._pending_units and all(event is None for event in self.events):
            self.slots.clear()
            self._closed = True
        if first_error is not None:
            raise first_error


@dataclass
class _ResidentUnitState:
    unit: Any
    device_binding: Any | None = None
    device_planes: Mapping[Any, torch.Tensor] | None = None


class _ResidentWeightController:
    """Materialize declared resident units once through the active session."""

    def __init__(
        self,
        units: Sequence[Any],
        *,
        device: torch.device,
        pin_memory: bool,
    ) -> None:
        self.device = device
        self.states = [_ResidentUnitState(unit) for unit in units]
        capacities: dict[torch.dtype, int] = {}
        for unit in units:
            totals: dict[torch.dtype, int] = {}
            for plane in unit.planes:
                totals[plane.dtype] = totals.get(plane.dtype, 0) + plane.storage_numel
            for dtype, total in totals.items():
                capacities[dtype] = max(capacities.get(dtype, 0), total)
        self.staging = {
            dtype: torch.empty(
                size,
                dtype=dtype,
                device="cpu",
                pin_memory=pin_memory,
            )
            for dtype, size in capacities.items()
        }
        self._pending_units: list[Any] = []

    def load(self, session: WeightAccessSession) -> None:
        for state in self.states:
            if state.device_binding is not None:
                continue
            offsets: dict[torch.dtype, int] = {}
            host_planes: dict[Any, torch.Tensor] = {}
            for plane in state.unit.planes:
                offset = offsets.get(plane.dtype, 0)
                host_planes[plane.plane_id] = self.staging[plane.dtype][offset : offset + plane.storage_numel]
                offsets[plane.dtype] = offset + plane.storage_numel
            prepared_unit = session.open_unit(UnitReadRequest(unit_id=state.unit.unit_id))
            self._pending_units.append(prepared_unit)
            try:
                publish = getattr(prepared_unit, "publish", None)
                if callable(publish):
                    publish()
                if prepared_unit.layout != state.unit:
                    raise HostWeightLayerwiseError(f"open_unit returned the wrong layout for {state.unit.unit_id!r}")
                prepared_unit.copy_into(host_planes)
            except BaseException as primary_error:
                try:
                    prepared_unit.close()
                except BaseException as cleanup_error:
                    _report_cleanup_failure(
                        primary_error,
                        "closing a failed resident weight read",
                        cleanup_error,
                    )
                else:
                    self._pending_units.remove(prepared_unit)
                raise
            try:
                prepared_unit.close()
            except BaseException:
                raise
            else:
                self._pending_units.remove(prepared_unit)

            device_planes: dict[Any, torch.Tensor] = {}
            for plane in state.unit.planes:
                host_plane = host_planes[plane.plane_id]
                device_plane = torch.empty(
                    plane.storage_numel,
                    dtype=plane.dtype,
                    device=self.device,
                )
                device_plane.copy_(
                    host_plane,
                    non_blocking=host_plane.is_pinned(),
                )
                device_planes[plane.plane_id] = device_plane
            current_omni_platform.synchronize()
            state.device_binding = session.bind_device(
                state.unit.unit_id,
                device_planes,
                lifetime=DeviceBindingLifetime.RESIDENT,
            )
            state.device_planes = MappingProxyType(device_planes)
            publish = getattr(state.device_binding, "publish", None)
            if callable(publish):
                publish()

    def release(self) -> None:
        first_error: BaseException | None = None
        synchronized = False
        for prepared_unit in tuple(self._pending_units):
            try:
                prepared_unit.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                else:
                    _report_cleanup_failure(
                        first_error,
                        "closing another resident weight read",
                        exc,
                    )
            else:
                self._pending_units.remove(prepared_unit)
        try:
            current_omni_platform.synchronize()
            synchronized = True
        except BaseException as exc:
            if first_error is None:
                first_error = exc
            else:
                _report_cleanup_failure(
                    first_error,
                    "synchronizing resident weight teardown",
                    exc,
                )
        if not synchronized:
            assert first_error is not None
            raise first_error
        for state in self.states:
            if state.device_binding is not None:
                try:
                    state.device_binding.release(ReleaseTarget.PLACEHOLDER)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                    else:
                        _report_cleanup_failure(
                            first_error,
                            "releasing another resident device binding",
                            exc,
                        )
                else:
                    state.device_binding = None
                    state.device_planes = None
        if synchronized and not self._pending_units and all(state.device_binding is None for state in self.states):
            self.staging.clear()
        if first_error is not None:
            raise first_error


class LayerwiseOffloadHook(ModelHook):
    """Hook for layerwise (transformer-block-wise) CPU offloading.

    The hook instance retains parameters for both the current registered block
    module and those for the next block, as well as flattened CPU tensors which
    record the parameters of the current block module, so that these parameters
    could be re-materialized on device in an overlapping way.
    This hook should be registered to each of the transformer blocks in DiT
    module(s) of the target pipeline.

    Based on implementations from:
    https://github.com/sgl-project/sglang/blob/v0.5.8/python/sglang/multimodal_gen/runtime/utils/layerwise_offload.py
    """

    _HOOK_NAME = "layerwise_offload"

    def __init__(
        self,
        next_block: nn.Module,
        device: torch.device,
        stream: current_omni_platform.Stream | None = None,
        pin_memory: bool = True,
        *,
        current_weight_binding: _HostWeightBlockBinding | None = None,
        next_weight_binding: _HostWeightBlockBinding | None = None,
        host_staging_pool: _HostWeightStagingPool | None = None,
        weight_session_handle: _WeightSessionHandle | None = None,
    ):
        assert isinstance(next_block, nn.Module), "transformer block must be type `torch.nn.Module`"

        self.next_block = next_block
        self.device = device
        self.copy_stream = stream or current_omni_platform.current_stream()
        self.pin_memory = pin_memory

        source_args = (
            current_weight_binding,
            next_weight_binding,
            host_staging_pool,
            weight_session_handle,
        )
        if any(arg is not None for arg in source_args) and not all(arg is not None for arg in source_args):
            raise HostWeightLayerwiseError("current binding, next binding, and staging pool must be supplied together")
        self.current_weight_binding = current_weight_binding
        self.next_weight_binding = next_weight_binding
        self.host_staging_pool = host_staging_pool
        self.weight_session_handle = weight_session_handle
        self._uses_weight_session = next_weight_binding is not None

        # Per-block synchronization primitive: set after H2D copy completes.
        self._prefetch_done: current_omni_platform.Event | None = None

        # Backward link to the hook that is responsible for prefetching *this* block's weights
        self._prev_hook: LayerwiseOffloadHook | None = None

        self.next_block_parameters: dict[str, nn.Parameter] = {}
        self.next_block_buffers: dict[str, torch.Tensor] = {}
        self.dtype_cpu_flattened_weights: dict[torch.dtype, torch.Tensor] = {}
        self.dtype_metadata: dict[torch.dtype, list[dict[str, Any]]] = {}

    @staticmethod
    def _is_dtensor(t: torch.Tensor) -> bool:
        return isinstance(t, DTensor)

    @staticmethod
    def _set_tensor_storage(target: torch.Tensor, value: torch.Tensor) -> None:
        if LayerwiseOffloadHook._is_dtensor(target):
            target._local_tensor = value
        else:
            target.data = value

    @staticmethod
    def _make_offload_placeholder(tensor: torch.Tensor) -> torch.Tensor:
        if LayerwiseOffloadHook._is_dtensor(tensor):
            local_shape = tuple(tensor.to_local().shape)
            return torch.empty(local_shape, device="meta", dtype=tensor.dtype)
        return torch.empty((0,), device=tensor.device, dtype=tensor.dtype)

    @staticmethod
    def _is_materialized_tensor(t: torch.Tensor) -> bool:
        if LayerwiseOffloadHook._is_dtensor(t):
            local_t = t.to_local()
            return not local_t.is_meta
        return not t.is_meta and t.data.numel() > 0

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        # This all happen during the hook instance being registered to hook registry;
        # the input module is kept intact
        module = super().initialize_hook(module)

        if self._uses_weight_session:
            assert self.current_weight_binding is not None
            assert self.next_weight_binding is not None
            if self.current_weight_binding.block is not module:
                raise HostWeightLayerwiseError("current host-weight binding does not match registered block")
            if self.next_weight_binding.block is not self.next_block:
                raise HostWeightLayerwiseError("next host-weight binding does not match prefetch block")

            # The integration binder owns target discovery and rebinding.
            # Hooks retain only scheduling state and immutable unit layouts.
            self.block_weight_targets = {}
            self.next_weight_targets = {}
            self.block_parameters = {}
            self.block_buffers = {}
            self.next_block_parameters = {}
            self.next_block_buffers = {}
        else:
            self.block_weight_targets = {}
            self.next_weight_targets = {}
            self.block_parameters = dict(module.named_parameters())
            self.block_buffers = dict(module.named_buffers())

            self.next_block_parameters = dict(self.next_block.named_parameters())
            self.next_block_buffers = dict(self.next_block.named_buffers())

            # Legacy behavior: retain one private flattened CPU copy per block.
            self.dtype_cpu_flattened_weights, self.dtype_metadata = LayerwiseOffloadHook._to_cpu(
                self.next_block_parameters,
                self.next_block_buffers,
                self.device,
                self.pin_memory,
            )

        return module

    @staticmethod
    def _to_cpu(
        params: dict[str, nn.Parameter],
        bufs: dict[str, torch.Tensor],
        device: torch.device,
        pin_memory: bool = True,
    ) -> tuple[dict[torch.dtype, torch.Tensor], dict[torch.dtype, list[dict[str, Any]]]]:
        """Helper method to move block parameters and buffers to CPU, flattening by dtype.

        Consolidates parameters and buffers into contiguous CPU tensors grouped by dtype
        for GPU transfers. Replaces original tensors with empty placeholders.

        Returns:
            Tuple of
                flattened CPU tensors by dtype,
                metadata for reconstruction by dtype
        """
        dtype_grouped_weights: dict[torch.dtype, dict[str, torch.Tensor]] = {}
        dtype_cpu_flattened_weights: dict[torch.dtype, torch.Tensor] = {}
        # NOTE: order does matter
        dtype_metadata: dict[torch.dtype, list[dict[str, Any]]] = {}

        for name, param_or_buf in chain(params.items(), bufs.items()):
            dtype = param_or_buf.dtype
            if dtype not in dtype_grouped_weights:
                dtype_grouped_weights[dtype] = {}
            dtype_grouped_weights[dtype][name] = param_or_buf

        for dtype, name2weights in dtype_grouped_weights.items():
            # total # of parameters + buffers
            weights_with_local = []
            for name, t in name2weights.items():
                local_t = t.to_local() if hasattr(t, "to_local") else t
                stride = local_t.stride()
                storage_numel = (
                    0
                    if local_t.numel() == 0
                    else 1 + sum((size - 1) * axis_stride for size, axis_stride in zip(local_t.shape, stride))
                )
                weights_with_local.append((name, t, local_t, storage_numel, stride))
            total_numel = sum(storage_numel for _, _, _, storage_numel, _ in weights_with_local)
            cpu_tensor = torch.empty(total_numel, dtype=dtype, device="cpu", pin_memory=pin_memory)

            current_offset = 0
            for (
                name,
                original_tensor,
                local_tensor,
                storage_numel,
                stride,
            ) in weights_with_local:
                if local_tensor.is_contiguous():
                    flat_storage = local_tensor.flatten()
                else:
                    # Cutlass FP8 weights use a transposed physical layout.
                    # Preserve it across the flattened CPU staging buffer.
                    flat_storage = torch.zeros(
                        storage_numel,
                        dtype=dtype,
                        device=local_tensor.device,
                    )
                    physical_view = torch.as_strided(
                        flat_storage,
                        size=local_tensor.shape,
                        stride=stride,
                    )
                    physical_view.copy_(local_tensor)
                cpu_tensor[current_offset : current_offset + storage_numel].copy_(flat_storage)
                if dtype not in dtype_metadata:
                    dtype_metadata[dtype] = []
                dtype_metadata[dtype].append(
                    {
                        "name": name,
                        "offset": current_offset,
                        "numel": storage_numel,
                        "shape": local_tensor.shape,
                        "stride": stride,
                    }
                )

                LayerwiseOffloadHook._set_tensor_storage(
                    original_tensor, LayerwiseOffloadHook._make_offload_placeholder(original_tensor)
                )
                current_offset += storage_numel

            dtype_cpu_flattened_weights[dtype] = cpu_tensor

        return dtype_cpu_flattened_weights, dtype_metadata

    @property
    def is_materialized(self) -> bool:
        """Check whether this block's parameters hold real data on device."""
        if self._uses_weight_session:
            assert self.current_weight_binding is not None
            return self.current_weight_binding.device_binding is not None
        for param in self.block_parameters.values():
            return LayerwiseOffloadHook._is_materialized_tensor(param)

        return True

    @torch.compiler.disable
    def prefetch_layer(self, non_blocking: bool = True) -> None:
        """Copy layer weights from CPU -> GPU.

        Pre-fetch target block in an asynchronous way with compute - memory copy overlap,
        with non_blocking set to True.
        """
        self.copy_stream.wait_stream(current_omni_platform.current_stream())

        if self._uses_weight_session:
            self._prefetch_from_weight_session(non_blocking=non_blocking)
            return

        layer_params = self.next_block_parameters
        layer_bufs = self.next_block_buffers

        evt = current_omni_platform.Event()
        gpu_weights: dict[torch.dtype, torch.Tensor] = {}

        with current_omni_platform.stream(self.copy_stream):
            for dtype, cpu_weight in self.dtype_cpu_flattened_weights.items():
                gpu_weight = torch.empty(cpu_weight.shape, dtype=dtype, device=self.device)
                gpu_weight.copy_(cpu_weight, non_blocking=non_blocking)
                gpu_weights[dtype] = gpu_weight

            evt.record(self.copy_stream)

        for dtype, ordered_metadata in self.dtype_metadata.items():
            # ordered_metadata: list[dict[str, Any]]
            gpu_weight = gpu_weights[dtype]

            for metadata in ordered_metadata:
                target_name = metadata["name"]
                target_param_or_buf = (
                    layer_params[target_name] if target_name in layer_params else layer_bufs[target_name]
                )

                LayerwiseOffloadHook._set_tensor_storage(
                    target_param_or_buf,
                    torch.as_strided(
                        gpu_weight[metadata["offset"] : metadata["offset"] + metadata["numel"]],
                        size=metadata["shape"],
                        stride=metadata["stride"],
                    ),
                )

        self._prefetch_done = evt

    def _prefetch_from_weight_session(self, *, non_blocking: bool) -> None:
        binding = self.next_weight_binding
        pool = self.host_staging_pool
        handle = self.weight_session_handle
        if binding is None or pool is None or handle is None:
            raise HostWeightLayerwiseError("host-weight hook is missing its active session")
        if binding.device_binding is not None:
            return
        session = handle.require()

        evt = current_omni_platform.Event()
        try:
            slot, cpu_weights = pool.stage(session, binding)
            gpu_weights: dict[Any, torch.Tensor] = {}
            with current_omni_platform.stream(self.copy_stream):
                for plane in binding.unit.planes:
                    cpu_weight = cpu_weights[plane.plane_id]
                    gpu_weight = torch.empty(
                        plane.storage_numel,
                        dtype=plane.dtype,
                        device=self.device,
                    )
                    gpu_weight.copy_(
                        cpu_weight,
                        non_blocking=non_blocking and cpu_weight.is_pinned(),
                    )
                    gpu_weights[plane.plane_id] = gpu_weight
                evt.record(self.copy_stream)

            pool.record_copy(slot, evt)
            binding.device_binding = session.bind_device(binding.unit.unit_id, gpu_weights)
            binding.device_planes = MappingProxyType(gpu_weights)
            publish = getattr(binding.device_binding, "publish", None)
            if callable(publish):
                publish()
        except BaseException as primary_error:
            # A partially enqueued H2D must not leave a reusable adapter.  The
            # platform-wide sync is an exceptional-path correctness barrier.
            try:
                current_omni_platform.synchronize()
                pool.close()
            except BaseException as cleanup_error:
                _report_cleanup_failure(
                    primary_error,
                    "draining layerwise host staging after prefetch failure",
                    cleanup_error,
                )
            raise

        self._prefetch_done = evt

    @torch.compiler.disable
    def offload_layer(self) -> None:
        """Free GPU memory for layer by replacing tensors with empty placeholders.
        This function does not actually offload weights from GPU back to CPU.
        """
        evt = self._prefetch_done
        if evt is not None:
            current_omni_platform.current_stream().wait_event(evt)

        self._prefetch_done = None

        # free GPU residency
        if self._uses_weight_session:
            assert self.current_weight_binding is not None
            if self.current_weight_binding.device_binding is not None:
                self.current_weight_binding.device_binding.release(ReleaseTarget.PLACEHOLDER)
                self.current_weight_binding.device_binding = None
                self.current_weight_binding.device_planes = None
            return

        for _, param in self.block_parameters.items():
            LayerwiseOffloadHook._set_tensor_storage(param, LayerwiseOffloadHook._make_offload_placeholder(param))
        for _, buf in self.block_buffers.items():
            LayerwiseOffloadHook._set_tensor_storage(buf, LayerwiseOffloadHook._make_offload_placeholder(buf))

    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple, dict]:
        try:
            # if the previous hook was skipped and the weights are not on device,
            # (e.g. by cache-dit block caching), ask the previous hook to
            # synchronously prefetch *this* block's weights before computation
            if not self.is_materialized and self._prev_hook is not None:
                self._prev_hook.prefetch_layer(non_blocking=False)

            if self._uses_weight_session and self._prev_hook is not None:
                ready = self._prev_hook._prefetch_done
                if ready is not None:
                    current_omni_platform.current_stream().wait_event(ready)

            self.prefetch_layer(non_blocking=True)
        except BaseException as primary_error:
            if self._uses_weight_session and self.weight_session_handle is not None:
                self.weight_session_handle.fail_closed(primary_error)
            raise

        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        try:
            self.offload_layer()
        except BaseException as primary_error:
            if self._uses_weight_session and self.weight_session_handle is not None:
                self.weight_session_handle.fail_closed(primary_error)
            raise

        return output


def apply_block_hook(
    module: nn.Module,
    next_block: nn.Module,
    device: torch.device,
    stream: current_omni_platform.Stream | None = None,
    pin_memory: bool = True,
    *,
    current_weight_binding: _HostWeightBlockBinding | None = None,
    next_weight_binding: _HostWeightBlockBinding | None = None,
    host_staging_pool: _HostWeightStagingPool | None = None,
    weight_session_handle: _WeightSessionHandle | None = None,
) -> LayerwiseOffloadHook:
    registry = HookRegistry.get_or_create(module)
    hook = LayerwiseOffloadHook(
        next_block,
        device,
        stream,
        pin_memory,
        current_weight_binding=current_weight_binding,
        next_weight_binding=next_weight_binding,
        host_staging_pool=host_staging_pool,
        weight_session_handle=weight_session_handle,
    )
    registry.register_hook(LayerwiseOffloadHook._HOOK_NAME, hook)

    return hook


def remove_block_hook(module: nn.Module) -> None:
    registry: HookRegistry | None = getattr(module, "_hook_registry", None)
    if registry is not None:
        registry.remove_hook(LayerwiseOffloadHook._HOOK_NAME)
        logger.debug("Removed offload hook from %s", module.__class__.__name__)


class LayerWiseOffloadBackend(OffloadBackend):
    """Layer-wise (block-level) offloading backend.

    Implements sliding window offloading where only a small number of transformer
    blocks reside on GPU at a time. Blocks are prefetched asynchronously while
    previous blocks compute, and freed after use.
    """

    def __init__(
        self,
        config: OffloadConfig,
        device: torch.device,
        *,
        prepared_weight_session: PreparedWeightAccessSession | None = None,
    ):
        super().__init__(config, device)

        self.copy_stream = current_omni_platform.Stream()
        self._blocks: list[list[nn.Module]] = []
        self._prepared_weight_session = prepared_weight_session
        self._weight_session: WeightAccessSession | None = None
        self._weight_session_handle = _WeightSessionHandle(fail_closed_callback=self._terminate_host_weight_session)
        self._uses_weight_session = prepared_weight_session is not None
        self._host_weight_terminal = False
        self._host_weight_teardown_phase = _SessionTeardownPhase.ACTIVE
        self._host_weight_bindings: dict[int, _HostWeightBlockBinding] = {}
        self._host_staging_pool: _HostWeightStagingPool | None = None
        self._source_hooked_blocks: list[nn.Module] = []
        self._session_initial_hooks: list[LayerwiseOffloadHook] = []
        self._resident_weight_controller: _ResidentWeightController | None = None

    def enable(self, pipeline: nn.Module) -> None:
        if self.enabled:
            logger.warning("LayerWiseOffloadBackend already enabled")
            return
        if self._uses_weight_session and self._host_weight_terminal:
            raise HostWeightLayerwiseError("host-weight layerwise backend has reached terminal teardown")

        try:
            # Discovery and block preflight are part of the prepared-session
            # transaction so protocol errors cannot retain the artifact lease.
            modules = ModuleDiscovery.discover(pipeline)
            if not modules.dits:
                if self._uses_weight_session:
                    raise HostWeightLayerwiseError(
                        "host-weight session was supplied, but no DiT/transformer module was discovered"
                    )
                logger.warning("No DiT/transformer modules found, skipping layer-wise offloading")
                return

            if self._uses_weight_session:
                prepared = self._prepared_weight_session
                if prepared is None:
                    raise HostWeightLayerwiseError("prepared host-weight session was already consumed")
                _validate_session_capabilities(prepared)
                if len(modules.dits) != 1:
                    raise HostWeightLayerwiseError(
                        "v1 layerwise host-weight offload requires exactly one managed DiT target"
                    )
                eligible_blocks: list[nn.Module] = []
                for dit_module in modules.dits:
                    _, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(dit_module)
                    if len(blocks) > 1:
                        eligible_blocks.extend(blocks)
                if not eligible_blocks:
                    raise HostWeightLayerwiseError(
                        "host-weight session was supplied, but no DiT has more than one offloadable block"
                    )

                # Compile every unit and allocate both bounded staging slots before
                # any encoder, DiT tensor, or hook is mutated.
                (
                    self._host_weight_bindings,
                    stationary_block_units,
                ) = self._prepare_host_weight_bindings(
                    modules.dits[0],
                    eligible_blocks,
                    prepared,
                )
                self._host_staging_pool = _HostWeightStagingPool(
                    [binding.unit for binding in self._host_weight_bindings.values()],
                    pin_memory=self.config.pin_cpu_memory,
                )
                resident_units = []
                for unit_id in prepared.transfer_plan.unit_ids:
                    unit = prepared.unit_spec(unit_id)
                    if _enum_value(unit.unit_kind) == UnitKind.RESIDENT.value:
                        resident_units.append(unit)
                resident_units.extend(stationary_block_units)
                if resident_units:
                    supported = {_enum_value(kind) for kind in prepared.capabilities.unit_kinds}
                    required = {_enum_value(unit.unit_kind) for unit in resident_units}
                    if not required <= supported:
                        raise HostWeightLayerwiseError(
                            f"stationary transfer units require unsupported kinds: {sorted(required - supported)}"
                        )
                    self._resident_weight_controller = _ResidentWeightController(
                        resident_units,
                        device=self.device,
                        pin_memory=self.config.pin_cpu_memory,
                    )

            self._enable_discovered(modules)
        except BaseException as primary_error:
            if self._uses_weight_session:
                cleanup_errors: list[tuple[str, BaseException]] = []
                try:
                    self._adopt_committed_weight_session()
                except BaseException as cleanup_error:
                    cleanup_errors.append(("active-session adoption", cleanup_error))
                if self._weight_session is None:
                    try:
                        self._rollback_prepared_weight_session()
                    except BaseException as cleanup_error:
                        cleanup_errors.append(("prepared-session rollback", cleanup_error))
                try:
                    self._terminate_host_weight_session()
                except BaseException as cleanup_error:
                    cleanup_errors.append(("terminal teardown", cleanup_error))
                for action, cleanup_error in cleanup_errors:
                    _report_cleanup_failure(
                        primary_error,
                        f"layerwise host-weight {action}",
                        cleanup_error,
                    )
            raise

    def _enable_discovered(self, modules: Any) -> None:
        # Move encoders to GPU (they stay resident)
        for enc in modules.encoders:
            enc.to(self.device)

        # Move VAE(s) to GPU if available
        for vae in modules.vaes:
            try:
                vae.to(self.device, non_blocking=True)
            except Exception as exc:
                logger.debug("Failed to move VAE to GPU: %s", exc)

        # Move resident modules to GPU (small modules needed every forward)
        for name, module in zip(modules.resident_names, modules.resident_modules):
            if self._uses_weight_session and any(
                module is descendant for dit in modules.dits for descendant in dit.modules()
            ):
                logger.debug(
                    "Resident module %s is owned by the host-weight resident unit",
                    name,
                )
                continue
            try:
                module.to(self.device)
            except Exception as exc:
                logger.debug("Failed to move resident module %s to GPU: %s", name, exc)

        logger.info("Applying layer-wise offloading on %s", modules.dit_names)

        # Apply block-wise offloading hook for each of the blocks in DiT model(s)
        # Note that there might exist multiple DiT models in specific pipelines
        for i, dit_module in enumerate(modules.dits):
            dit_name = modules.dit_names[i]
            logger.info(f"Applying hooks on {dit_name} ({dit_module.__class__.__name__})")

            blocks_attr_names, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(dit_module)

            if not blocks:
                logger.warning(
                    "Target layers (blocks) not found. Skipping offloading on %s (%s)",
                    dit_name,
                    dit_module.__class__.__name__,
                )
                if not self._uses_weight_session:
                    dit_module.to(self.device)
                continue

            num_blocks = len(blocks)
            if num_blocks <= 1:
                logger.warning(
                    "#Target layers (blocks) <= 1. Skipping offloading on %s (%s)",
                    dit_name,
                    dit_module.__class__.__name__,
                )
                if not self._uses_weight_session:
                    dit_module.to(self.device)
                continue

            # Move non-block modules to GPU (they stay resident)
            if self._uses_weight_session:
                logger.debug(
                    "Leaving non-block state for %s as structural placeholders until the resident unit is bound",
                    dit_name,
                )
            else:
                for name, m in dit_module.named_children():
                    if name not in blocks_attr_names:
                        m.to(self.device)
                        logger.debug("Moved %s to device %s", name, self.device)
                    else:
                        logger.debug("Skipped blocks module %s", name)

                # Move top-level params/buffers to GPU (dit_module's own, not sub-modules)
                for param in dit_module._parameters.values():
                    if param is not None:
                        param.data = param.data.to(self.device, non_blocking=True)

                for buffer in dit_module._buffers.values():
                    if buffer is not None:
                        buffer.data = buffer.data.to(self.device, non_blocking=True)

            # Pre-fetch the first layer by manually calling the hook function on the last layer;
            # For subsequent requests, the first layer/block will be pre-fetched
            # during the last layer compute of the previous request.
            last_block, first_block = blocks[-1], blocks[0]
            if self._uses_weight_session:
                assert self._host_staging_pool is not None
                self._source_hooked_blocks.append(last_block)
                last_hook = apply_block_hook(
                    last_block,
                    first_block,
                    self.device,
                    self.copy_stream,
                    self.config.pin_cpu_memory,
                    current_weight_binding=self._host_weight_bindings[id(last_block)],
                    next_weight_binding=self._host_weight_bindings[id(first_block)],
                    host_staging_pool=self._host_staging_pool,
                    weight_session_handle=self._weight_session_handle,
                )
            else:
                last_hook = apply_block_hook(
                    last_block,
                    first_block,
                    self.device,
                    self.copy_stream,
                    self.config.pin_cpu_memory,
                )
            if self._uses_weight_session:
                self._session_initial_hooks.append(last_hook)
            else:
                last_hook.prefetch_layer(non_blocking=False)

            block_hooks: list[LayerwiseOffloadHook] = [last_hook]
            # Register hook for each of blocks
            for i, block in enumerate(blocks[:-1]):
                next_block = blocks[(i + 1) % num_blocks]
                if self._uses_weight_session:
                    assert self._host_staging_pool is not None
                    self._source_hooked_blocks.append(block)
                    hook = apply_block_hook(
                        block,
                        next_block,
                        self.device,
                        self.copy_stream,
                        self.config.pin_cpu_memory,
                        current_weight_binding=self._host_weight_bindings[id(block)],
                        next_weight_binding=self._host_weight_bindings[id(next_block)],
                        host_staging_pool=self._host_staging_pool,
                        weight_session_handle=self._weight_session_handle,
                    )
                else:
                    hook = apply_block_hook(
                        block,
                        next_block,
                        self.device,
                        self.copy_stream,
                        self.config.pin_cpu_memory,
                    )
                block_hooks.append(hook)

            # NOTE(yuanheng-zhao): We make each hook gets a backward reference to the hook
            # that is responsible for prefetching its block's weights. This is specifically a
            # workaround for that arbitrary blocks are skipped by caching systems (e.g., cache-dit)
            for i in range(len(block_hooks)):
                block_hooks[i]._prev_hook = block_hooks[i - 1]

            logger.info(f"Layer-wise offloading enabled on {num_blocks} layers (blocks)")

            # Track hooked blocks for cleanup
            self._blocks.append(blocks)

        if self._uses_weight_session and self._blocks:
            prepared = self._prepared_weight_session
            if prepared is None:
                raise HostWeightLayerwiseError("prepared host-weight session was already consumed")
            self._weight_session = prepared.commit()
            self._adopt_committed_weight_session()
            self._weight_session_handle.session = self._weight_session
            if self._resident_weight_controller is not None:
                self._resident_weight_controller.load(self._weight_session)
            # Keep the committed startup boundary free of transient block
            # bindings.  The first block's pre-forward hook detects its
            # placeholder and performs the initial synchronous fill; later
            # blocks retain the ordinary overlapped prefetch ring.

        if len(self._blocks) > 0 and len(self._blocks[0]) > 0:
            self.enabled = True

    def disable(self) -> None:
        if self._uses_weight_session:
            if self._prepared_weight_session is not None:
                self._rollback_prepared_weight_session()
            if self._host_weight_teardown_phase is _SessionTeardownPhase.CLOSED:
                return
            self._terminate_host_weight_session()
            logger.info("Layer-wise host-weight offloading disabled (terminal teardown)")
            return

        if not self.enabled:
            return

        for blocks in self._blocks:
            for block in blocks:
                remove_block_hook(block)

        self._blocks.clear()
        self.enabled = False
        logger.info("Layer-wise offloading disabled")

    def host_weight_diagnostics(self) -> dict[str, object]:
        """Return deduplicated pinned staging bytes and in-flight events."""

        tensors: list[torch.Tensor] = []
        events: list[Any | None] = []
        pool = self._host_staging_pool
        if pool is not None:
            for slot in pool.slots:
                tensors.extend(slot.values())
            events.extend(pool.events)
        resident = self._resident_weight_controller
        if resident is not None:
            tensors.extend(resident.staging.values())
        for blocks in self._blocks:
            for block in blocks:
                registry = getattr(block, "_hook_registry", None)
                hook = registry.get_hook("layerwise_offload") if registry is not None else None
                if hook is not None:
                    events.append(getattr(hook, "_prefetch_done", None))
        return {
            "pinned_slot_budget_bytes": _pinned_cpu_storage_bytes(tensors),
            "events": _incomplete_event_count(events),
        }

    @classmethod
    def _prepare_host_weight_bindings(
        cls,
        managed_target: nn.Module,
        blocks: Sequence[nn.Module],
        prepared: PreparedWeightAccessSession,
    ) -> tuple[dict[int, _HostWeightBlockBinding], tuple[Any, ...]]:
        block_ids = {id(block) for block in blocks}
        if len(block_ids) != len(blocks):
            raise HostWeightLayerwiseError("the discovered layerwise block list contains duplicate modules")

        block_by_id = {id(block): block for block in blocks}
        transfer_by_block_id: dict[int, tuple[str, Any]] = {}
        stationary_units: list[Any] = []
        resolved_execution_ids: set[int] = set()
        for execution in prepared.transfer_plan.execution_bindings:
            unit = prepared.unit_spec(execution.unit_id)
            if _enum_value(unit.unit_kind) != UnitKind.BLOCK.value:
                raise HostWeightLayerwiseError(f"layerwise execution binding selects non-block unit {unit.unit_id!r}")
            execution_module = _resolve_execution_module(managed_target, execution.module_path)
            module_id = id(execution_module)
            if module_id in resolved_execution_ids:
                raise HostWeightLayerwiseError("selected transfer plan binds the same target module more than once")
            resolved_execution_ids.add(module_id)
            if module_id not in block_by_id:
                # A catalog may contain additional plan-declared rings (for
                # example MiniMax-H3's token refiner).  Ordinary layerwise
                # offload keeps those complete units resident instead of
                # silently trying to read their meta placeholders.
                stationary_units.append(unit)
                continue
            transfer_by_block_id[module_id] = (str(execution.module_path), unit)

        bindings: dict[int, _HostWeightBlockBinding] = {}
        for block in blocks:
            selected = transfer_by_block_id.get(id(block))
            if selected is None:
                raise HostWeightLayerwiseError(
                    "selected transfer plan has no execution binding for a discovered layerwise block"
                )
            block_path, unit = selected
            if not unit.planes or not unit.bindings:
                raise HostWeightLayerwiseError(f"block transfer unit {unit.unit_id!r} is empty")

            bindings[id(block)] = _HostWeightBlockBinding(
                block_path=block_path,
                block=block,
                unit=unit,
            )

        return bindings, tuple(stationary_units)

    def _rollback_prepared_weight_session(self) -> None:
        prepared = self._prepared_weight_session
        if prepared is None:
            return
        prepared.rollback()
        if self._prepared_weight_session is prepared:
            self._prepared_weight_session = None

    def _adopt_committed_weight_session(self) -> None:
        prepared = self._prepared_weight_session
        session = self._weight_session
        if prepared is None or session is None:
            return
        prepared.adopt(session)
        if self._prepared_weight_session is prepared:
            self._prepared_weight_session = None

    def _terminate_host_weight_session(self) -> None:
        self._host_weight_terminal = True
        first_error: BaseException | None = None

        if self._host_weight_teardown_phase is _SessionTeardownPhase.ACTIVE:
            if self._host_staging_pool is not None:
                try:
                    self._host_staging_pool.close()
                except BaseException as exc:  # pragma: no cover - platform failure
                    first_error = exc

            synchronized = False
            try:
                current_omni_platform.synchronize()
                synchronized = True
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                else:
                    _report_cleanup_failure(
                        first_error,
                        "synchronizing layerwise terminal teardown",
                        exc,
                    )
            if not synchronized:
                assert first_error is not None
                raise first_error
            for binding in self._host_weight_bindings.values():
                if binding.device_binding is not None:
                    try:
                        binding.device_binding.release(ReleaseTarget.PLACEHOLDER)
                    except BaseException as exc:
                        if first_error is None:
                            first_error = exc
                        else:
                            _report_cleanup_failure(
                                first_error,
                                "releasing another layerwise device binding",
                                exc,
                            )
                    else:
                        binding.device_binding = None
                        binding.device_planes = None

            if self._resident_weight_controller is not None:
                try:
                    self._resident_weight_controller.release()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                    else:
                        _report_cleanup_failure(
                            first_error,
                            "releasing resident layerwise weights",
                            exc,
                        )

            # Suspending with a binding whose release failed would only raise a
            # secondary "session is busy" error.  Keep the active session, hooks,
            # and failed binding as the retry handle for the next disable().
            if first_error is not None:
                raise first_error

            session = self._weight_session
            if session is not None:
                # Cross the strict session boundary exactly once.  Later
                # hook-removal retries resume from the quiesced phase.
                session.suspend()
            self._host_weight_teardown_phase = _SessionTeardownPhase.QUIESCED

        while self._source_hooked_blocks:
            block = self._source_hooked_blocks[0]
            remove_block_hook(block)
            del self._source_hooked_blocks[0]

        session = self._weight_session
        if session is not None:
            session.close(DetachMode.TERMINAL)

        self._source_hooked_blocks.clear()
        self._session_initial_hooks.clear()
        self._blocks.clear()
        self._host_weight_bindings.clear()
        self._host_staging_pool = None
        self._resident_weight_controller = None
        self._weight_session_handle.session = None
        self._weight_session = None
        self.enabled = False
        self._host_weight_teardown_phase = _SessionTeardownPhase.CLOSED

    @staticmethod
    def get_blocks_attr_names(model: nn.Module) -> list[str]:
        """Get block attribute names from model class."""
        attrs: list[str] = getattr(model.__class__, "_layerwise_offload_blocks_attrs", [])

        if not attrs:
            old_attr = getattr(model.__class__, "_layerwise_offload_blocks_attr", None)
            if old_attr is not None:
                logger.warning(
                    "'_layerwise_offload_blocks_attr' is deprecated, "
                    "please use '_layerwise_offload_blocks_attrs' instead. "
                    "Example: _layerwise_offload_blocks_attrs = ['blocks']"
                )
                attrs = [old_attr] if isinstance(old_attr, str) else list(old_attr)

        return attrs

    @staticmethod
    def set_blocks_attr_names(model: nn.Module, names: list[str]) -> None:
        if not hasattr(model.__class__, "_layerwise_offload_blocks_attrs"):
            setattr(model.__class__, "_layerwise_offload_blocks_attrs", names)

    @staticmethod
    def get_blocks_from_dit(model: nn.Module) -> tuple[list[str], list[nn.Module]]:
        """
        Retrieve blocks and attribute names from provided DiT model. Blocks attribute names
        are found by `_layerwise_offload_blocks_attrs` set to DiT models. For example,

        ```
        class WanTransformer3DModel(nn.Module):
            _layerwise_offload_blocks_attrs = ["blocks"]
        ```

        Returns:
            Tuple of (blocks_attr_names, blocks)
        """
        blocks_attr_names = LayerWiseOffloadBackend.get_blocks_attr_names(model)
        if not blocks_attr_names:
            logger.warning(
                f"No _layerwise_offload_blocks_attrs defined for {model.__class__.__name__}, "
                "skipping layerwise offloading"
            )
            return [], []

        blocks = []
        for name in blocks_attr_names:
            attr = getattr(model, name, None)
            if attr is None:
                raise AttributeError(
                    f"Attribute '{name}' declared in _layerwise_offload_blocks_attrs "
                    f"does not exist on model {model.__class__.__name__}"
                )
            try:
                attr_iter = iter(attr)
            except TypeError:
                if isinstance(attr, nn.Module):
                    logger.warning(
                        "Attribute '%s' on %s is not iterable; treating it as one block.",
                        name,
                        model.__class__.__name__,
                    )
                    blocks.append(attr)
                    continue

                logger.warning(
                    "Attribute '%s' on %s is not iterable (got %s); skipping it.",
                    name,
                    model.__class__.__name__,
                    type(attr).__name__,
                )
            else:
                blocks.extend(attr_iter)

        if not blocks:
            logger.warning(
                "No blocks found in %s for %s, skipping layerwise offloading",
                blocks_attr_names,
                model.__class__.__name__,
            )
            return [], []

        return blocks_attr_names, blocks
