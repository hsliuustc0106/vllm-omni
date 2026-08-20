# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

import torch
from torch import nn
from torch.distributed._tensor import DTensor  # type: ignore[attr-defined]
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.diffusion.host_weight.session import (
    DetachMode,
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


class HostWeightModelLevelError(RuntimeError):
    """Raised when a host source cannot safely serve model-level offload."""


class _SessionTeardownPhase(str, Enum):
    ACTIVE = "active"
    QUIESCED = "quiesced"
    CLOSED = "closed"


@dataclass(frozen=True)
class _HostWeightComponentBinding:
    component_path: str
    component: nn.Module
    unit: Any


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


def _validate_session_capabilities(
    prepared: PreparedWeightAccessSession,
    *,
    plan_kind: TransferPlanKind,
) -> None:
    capabilities = prepared.capabilities
    plan = prepared.transfer_plan
    if _enum_value(plan.plan_kind) != plan_kind.value:
        raise HostWeightModelLevelError(
            f"model-level offload requires the {plan_kind.value!r} transfer plan, got {_enum_value(plan.plan_kind)!r}"
        )
    if _enum_value(capabilities.selected_transfer_plan_kind) != plan_kind.value:
        raise HostWeightModelLevelError("host-weight capabilities select a different transfer plan kind")
    if capabilities.selected_transfer_plan_id != plan.plan_id:
        raise HostWeightModelLevelError("host-weight capabilities select a different transfer plan ID")
    access_features = {_enum_value(feature) for feature in capabilities.access_features}
    if "complete_tensor_read" not in access_features:
        raise HostWeightModelLevelError("host-weight session does not support complete-tensor reads")
    if _enum_value(capabilities.host_copy_mode) != "synchronous":
        raise HostWeightModelLevelError("model-level host-weight offload requires synchronous host copies")


class _HostWeightComponentController:
    """Own one bounded host slot and the active component binding."""

    def __init__(
        self,
        bindings: Sequence[_HostWeightComponentBinding],
        *,
        device: torch.device,
        pin_memory: bool,
        fail_closed: Callable[[], None],
    ) -> None:
        if not bindings:
            raise HostWeightModelLevelError("model-level host-weight source has no managed components")

        capacities: dict[torch.dtype, int] = {}
        for binding in bindings:
            totals: dict[torch.dtype, int] = {}
            for plane in binding.unit.planes:
                totals[plane.dtype] = totals.get(plane.dtype, 0) + plane.storage_numel
            for dtype, total in totals.items():
                capacities[dtype] = max(capacities.get(dtype, 0), total)
        if not capacities:
            raise HostWeightModelLevelError("model-level transfer units contain no dtype planes")

        self.device = device
        self.capacities = MappingProxyType(capacities)
        self.staging = {
            dtype: torch.empty(
                numel,
                dtype=dtype,
                device="cpu",
                pin_memory=pin_memory,
            )
            for dtype, numel in capacities.items()
        }
        self._bindings = {id(binding.component): binding for binding in bindings}
        self._active_component_id: int | None = None
        self._active_binding: Any | None = None
        self._active_device_planes: Mapping[Any, torch.Tensor] | None = None
        self._session: WeightAccessSession | None = None
        self._pending_units: list[Any] = []
        self._terminal = False
        self._fail_closed = fail_closed

    @property
    def terminal(self) -> bool:
        return self._terminal

    def manages(self, module: nn.Module) -> bool:
        return id(module) in self._bindings

    def attach(self, session: WeightAccessSession) -> None:
        if self._terminal or self._session is not None:
            raise HostWeightModelLevelError("model-level host-weight controller cannot attach a session")
        self._session = session

    def stop(self) -> None:
        self._terminal = True

    def fail_closed(self, primary_error: BaseException) -> None:
        """Abort the owning backend without replacing the hook failure."""
        try:
            self._fail_closed()
        except BaseException as cleanup_error:
            _report_cleanup_failure(
                primary_error,
                "model-level host-weight terminal cleanup",
                cleanup_error,
            )

    def _binding(self, module: nn.Module) -> _HostWeightComponentBinding:
        try:
            return self._bindings[id(module)]
        except KeyError as exc:
            raise HostWeightModelLevelError(
                f"module {module.__class__.__name__} is not managed by the host source"
            ) from exc

    def activate(self, module: nn.Module) -> None:
        if self._terminal:
            raise HostWeightModelLevelError("model-level host-weight controller is terminal")
        binding = self._binding(module)
        if self._active_component_id == id(module):
            return
        if self._active_component_id is not None:
            raise HostWeightModelLevelError(
                "host-weight component activation requires the previous component to be released"
            )
        session = self._session
        if session is None:
            raise HostWeightModelLevelError("model-level host-weight session is not committed")

        offsets: dict[torch.dtype, int] = {}
        destinations: dict[Any, torch.Tensor] = {}
        for plane in binding.unit.planes:
            offset = offsets.get(plane.dtype, 0)
            destinations[plane.plane_id] = self.staging[plane.dtype][offset : offset + plane.storage_numel]
            offsets[plane.dtype] = offset + plane.storage_numel

        prepared_unit = session.open_unit(UnitReadRequest(unit_id=binding.unit.unit_id))
        self._pending_units.append(prepared_unit)
        try:
            publish = getattr(prepared_unit, "publish", None)
            if callable(publish):
                publish()
            if prepared_unit.layout != binding.unit:
                raise HostWeightModelLevelError(f"open_unit returned the wrong layout for {binding.unit.unit_id!r}")
            prepared_unit.copy_into(destinations)
        except BaseException as primary_error:
            try:
                prepared_unit.close()
            except BaseException as cleanup_error:
                _report_cleanup_failure(
                    primary_error,
                    "closing the failed model-level weight read",
                    cleanup_error,
                )
            else:
                self._pending_units.remove(prepared_unit)
            try:
                self._abort()
            except BaseException as cleanup_error:
                _report_cleanup_failure(
                    primary_error,
                    "aborting model-level host staging",
                    cleanup_error,
                )
            raise
        try:
            prepared_unit.close()
        except BaseException as primary_error:
            try:
                self._abort()
            except BaseException as cleanup_error:
                _report_cleanup_failure(
                    primary_error,
                    "aborting model-level host staging",
                    cleanup_error,
                )
            raise
        self._pending_units.remove(prepared_unit)

        try:
            device_planes: dict[Any, torch.Tensor] = {}
            for plane in binding.unit.planes:
                host_plane = destinations[plane.plane_id]
                device_plane = torch.empty(
                    plane.storage_numel,
                    dtype=plane.dtype,
                    device=self.device,
                )
                device_plane.copy_(host_plane, non_blocking=host_plane.is_pinned())
                device_planes[plane.plane_id] = device_plane

            current_omni_platform.synchronize()
            device_binding = session.bind_device(binding.unit.unit_id, device_planes)
        except BaseException as primary_error:
            try:
                self._abort()
            except BaseException as cleanup_error:
                _report_cleanup_failure(
                    primary_error,
                    "aborting model-level device activation",
                    cleanup_error,
                )
            raise

        self._active_component_id = id(module)
        self._active_binding = device_binding
        self._active_device_planes = MappingProxyType(device_planes)
        publish = getattr(device_binding, "publish", None)
        if callable(publish):
            publish()

    def release(self, module: nn.Module) -> None:
        if self._terminal:
            raise HostWeightModelLevelError("model-level host-weight controller is terminal")
        self._binding(module)
        if self._active_component_id != id(module):
            return
        current_omni_platform.synchronize()
        assert self._active_binding is not None
        self._active_binding.release(ReleaseTarget.PLACEHOLDER)
        self._active_binding = None
        self._active_device_planes = None
        self._active_component_id = None
        current_omni_platform.empty_cache()

    def _abort(self) -> None:
        self.close()

    def close(self) -> None:
        self.stop()
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
                        "closing another pending model-level weight read",
                        exc,
                    )
            else:
                self._pending_units.remove(prepared_unit)
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
                    "synchronizing model-level host teardown",
                    exc,
                )
        # Device tensors may still be in use when synchronization fails.  Keep
        # the binding and its planes as the retry owner; placeholder restore is
        # only safe after a successful synchronization boundary.
        if not synchronized:
            assert first_error is not None
            raise first_error
        if self._active_binding is not None:
            try:
                self._active_binding.release(ReleaseTarget.PLACEHOLDER)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                else:
                    _report_cleanup_failure(
                        first_error,
                        "releasing the active model-level device binding",
                        exc,
                    )
            else:
                self._active_binding = None
                self._active_device_planes = None
                self._active_component_id = None
        if synchronized and not self._pending_units and self._active_binding is None:
            self.staging.clear()
        if first_error is not None:
            raise first_error


class SequentialOffloadHook(ModelHook):
    """Hook for sequential offloading with mutual exclusion on encoder and DiT modules.

    To be used as a model-level (or "component-level") of CPU offloading method;
    When a module's forward is called, this hook offloads target modules to CPU
    and loads the current module to GPU.
    """

    _HOOK_NAME = "sequential_offload"

    def __init__(
        self,
        offload_targets: list[nn.Module],
        device: torch.device,
        pin_memory: bool = True,
        use_hsdp: bool = False,
        host_weight_controller: _HostWeightComponentController | None = None,
    ):
        # Modules to offload to CPU before this module runs
        self.offload_targets = offload_targets
        self.device = device
        self.pin_memory = pin_memory
        self.use_hsdp = use_hsdp
        self.host_weight_controller = host_weight_controller

    @staticmethod
    def _move_params(
        module: nn.Module,
        target_device: torch.device,
        *,
        non_blocking: bool = False,
        pin_memory: bool = False,
    ) -> None:
        """Move module parameters and buffers to device.

        This cls method specifically prevents recursion device movement,
        E.g., Cache-DiT CachedBlocks has attr `transformer` as a ref to original
        transformer blocks, thus `module.to(device)` will fail for recursion calling,
        refer to
        https://github.com/vipshop/cache-dit/blob/v1.2.3/src/cache_dit/caching/cache_blocks/__init__.py#L83
        """
        for p in module.parameters():
            if p.data.device != target_device:
                data = p.data.to(target_device, non_blocking=non_blocking)
                if pin_memory and target_device.type == "cpu" and not isinstance(data, DTensor):
                    data = data.pin_memory()
                p.data = data
        for b in module.buffers():
            if b.device != target_device:
                data = b.data.to(target_device, non_blocking=non_blocking)
                if pin_memory and target_device.type == "cpu" and not isinstance(data, DTensor):
                    data = data.pin_memory()
                b.data = data

    def _to_cpu(self, module: nn.Module) -> None:
        try:
            param = next(module.parameters())
        except StopIteration:
            return

        if param.device.type == "cpu":
            return

        # XPU's allocator doesn't respect stream dependencies in empty_cache,
        # so non-blocking copies can race with cache eviction. Use blocking
        # copies on XPU to avoid NULL pointer errors during DMA.
        non_blocking = not self.use_hsdp and not current_omni_platform.is_xpu()
        self._move_params(
            module,
            torch.device("cpu"),
            non_blocking=non_blocking,
            pin_memory=self.pin_memory,
        )
        current_omni_platform.empty_cache()

    def _to_gpu(self, module: nn.Module) -> None:
        try:
            if next(module.parameters()).device == self.device:
                return
        except StopIteration:
            return

        self._move_params(module, self.device, non_blocking=False)

    def pre_forward(self, module: nn.Module, *args, **kwargs) -> tuple[tuple, dict]:
        try:
            # Offload target modules to CPU
            for target in self.offload_targets:
                if self.host_weight_controller is not None and self.host_weight_controller.manages(target):
                    self.host_weight_controller.release(target)
                else:
                    self._to_cpu(target)

            # Load current module to GPU
            if self.host_weight_controller is not None and self.host_weight_controller.manages(module):
                self.host_weight_controller.activate(module)
            else:
                self._to_gpu(module)
            current_omni_platform.synchronize()
        except BaseException as primary_error:
            if self.host_weight_controller is not None:
                self.host_weight_controller.fail_closed(primary_error)
            raise

        logger.debug(
            "Swapped: %s -> CPU, %s -> %s, free memory: %.4f GB",
            [t.__class__.__name__ for t in self.offload_targets],
            module.__class__.__name__,
            f"{self.device.type}:{self.device.index}",
            current_omni_platform.get_free_memory() / 1024 / 1024 / 1024,
        )

        return args, kwargs


def apply_sequential_offload(
    dit_modules: list[nn.Module],
    encoder_modules: list[nn.Module],
    device: torch.device,
    pin_memory: bool = True,
    use_hsdp: bool = False,
    *,
    host_weight_controller: _HostWeightComponentController | None = None,
) -> None:
    """Apply sequential offloading hooks to DiT and encoder modules.

    Registers hooks on modules to implement mutual-exclusion GPU allocation.
        - Before DiT runs, encoders are offloaded to CPU.
        - Before encoders run, DiT is offloaded to CPU.

    Args:
        dit_modules: DiT/transformer modules to register hooks on
        encoder_modules: Encoder modules to register hooks on
        device: Target GPU device for loading
        pin_memory: Whether to pin CPU memory for faster transfers
        use_hsdp: Whether HSDP is enabled (affects non_blocking behavior)

    Example:
        >>> apply_sequential_offload(
        ...     dit_modules=[pipeline.transformer],
        ...     encoder_modules=[pipeline.text_encoder, pipeline.vae],
        ...     device=torch.device("cuda:0"),
        ... )
        >>> # Modules of pipeline now automatically swap between CPU and GPU
    """
    # Register hooks on DiT modules (offload encoders AND other DiTs when a DiT runs)
    for i, dit_mod in enumerate(dit_modules):
        other_dits = [d for j, d in enumerate(dit_modules) if j != i]
        registry = HookRegistry.get_or_create(dit_mod)
        hook = SequentialOffloadHook(
            offload_targets=encoder_modules + other_dits,
            device=device,
            pin_memory=pin_memory,
            use_hsdp=use_hsdp,
            host_weight_controller=host_weight_controller,
        )
        registry.register_hook(SequentialOffloadHook._HOOK_NAME, hook)
        logger.debug("Registered offload hook for %s", dit_mod.__class__.__name__)

    # Register hooks on encoders (offload DiTs when encoder runs)
    for enc in encoder_modules:
        registry = HookRegistry.get_or_create(enc)
        hook = SequentialOffloadHook(
            offload_targets=dit_modules,
            device=device,
            pin_memory=pin_memory,
            use_hsdp=use_hsdp,
            host_weight_controller=host_weight_controller,
        )
        registry.register_hook(SequentialOffloadHook._HOOK_NAME, hook)
        logger.debug("Registered offload hook for %s", enc.__class__.__name__)


def remove_sequential_offload(modules: list[nn.Module]) -> None:
    """Remove sequential offloading hooks from modules.

    Args:
        modules: Modules to remove hooks from

    Example:
        >>> all_modules = [*dit_modules, *encoder_modules]
        >>> remove_sequential_offload(all_modules)
    """
    for module in modules:
        registry: HookRegistry | None = getattr(module, "_hook_registry", None)
        if registry is not None:
            registry.remove_hook(SequentialOffloadHook._HOOK_NAME)
            logger.debug("Removed offload hook from %s", module.__class__.__name__)


class ModelLevelOffloadBackend(OffloadBackend):
    """Model-level (sequential) offloading backend.

    Uses SequentialOffloadHook registered via HookRegistry for automatic module swapping.
    """

    def __init__(
        self,
        config: OffloadConfig,
        device: torch.device,
        *,
        prepared_weight_session: PreparedWeightAccessSession | None = None,
    ):
        super().__init__(config, device)
        self._offload_modules: list[nn.Module] = []  # Track modules with hooks
        self._custom_pipeline: nn.Module | None = None
        self._prepared_weight_session: PreparedWeightAccessSession | None = None
        self._weight_session: WeightAccessSession | None = None
        self._uses_weight_session = False
        self._host_weight_terminal = False
        self._host_weight_teardown_phase = _SessionTeardownPhase.ACTIVE
        self._host_weight_controller: _HostWeightComponentController | None = None
        if prepared_weight_session is not None:
            self.adopt_prepared_session(prepared_weight_session)

    def adopt_prepared_session(self, prepared: PreparedWeightAccessSession) -> None:
        """Atomically take cleanup authority for one prepared HWR session."""

        if self.enabled or self._uses_weight_session or self._host_weight_terminal:
            raise RuntimeError("model-level offloader already owns host-weight session state")
        self._prepared_weight_session = prepared
        self._uses_weight_session = True

    def enable(self, pipeline: nn.Module) -> None:
        if self.enabled:
            logger.warning("ModelLevelOffloadBackend already enabled")
            return
        if self._uses_weight_session and self._host_weight_terminal:
            raise HostWeightModelLevelError("model-level host-weight backend has reached terminal teardown")

        if self._uses_weight_session:
            self._enable_with_host_weight_session(pipeline)
            return

        # Pipelines with a nested transformer (e.g. Cosmos3's reasoner/generator
        # pathways) own their own mutual-exclusion swaps and cannot be offloaded
        # with the generic encoder<->DiT hooks. Delegate to the custom hook.
        # TODO: Cosmos3 is the only implementer today, so we duck-type the
        # enable/disable_omni_model_cpu_offload pair via getattr. Once a second
        # pipeline needs it, formalize this as a Protocol (like
        # SupportsComponentDiscovery) instead of getattr detection.
        custom_enable = getattr(pipeline, "enable_omni_model_cpu_offload", None)
        if callable(custom_enable):
            custom_enable(
                device=self.device,
                pin_memory=self.config.pin_cpu_memory,
                use_hsdp=self.config.use_hsdp,
            )
            self._custom_pipeline = pipeline
            self.enabled = True
            logger.info(
                "Model-level offloading enabled through %s.enable_omni_model_cpu_offload",
                pipeline.__class__.__name__,
            )
            return

        modules = ModuleDiscovery.discover(pipeline)

        # Move encoders to GPU
        for enc in modules.encoders:
            enc.to(self.device)

        # Move VAE(s) to GPU if available
        for vae in modules.vaes:
            try:
                vae.to(self.device, non_blocking=True)
            except Exception as exc:
                logger.debug("Failed to move VAE to GPU: %s", exc)

        # Pin resident modules on GPU (small hot submodules called inside the DiT loop).
        for res, name in zip(modules.resident_modules, modules.resident_names):
            try:
                res.to(self.device)
            except Exception as exc:
                logger.warning("Failed to move resident module '%s' to GPU: %s", name, exc)

        if not modules.dits:
            logger.warning("No DiT/transformer modules found, skipping model-level offloading")
            return

        if not modules.encoders:
            # Nothing to swap against — move DiTs to GPU and skip hooks.
            for dit in modules.dits:
                dit.to(self.device)
            logger.warning("No encoder modules found, skipping model-level offloading")
            return

        # Apply sequential offloading hooks
        apply_sequential_offload(
            dit_modules=modules.dits,
            encoder_modules=modules.encoders,
            device=self.device,
            pin_memory=self.config.pin_cpu_memory,
            use_hsdp=self.config.use_hsdp,
        )

        # Track modules for cleanup
        self._offload_modules = [*modules.dits, *modules.encoders]

        self.enabled = True

        logger.info(
            "Model-level offloading enabled: %s <-> %s (mutual exclusion)%s",
            ", ".join(modules.dit_names),
            ", ".join(modules.encoder_names),
            f"; resident on GPU: {', '.join(modules.resident_names)}" if modules.resident_names else "",
        )

    def _enable_with_host_weight_session(self, pipeline: nn.Module) -> None:
        try:
            if callable(getattr(pipeline, "enable_omni_model_cpu_offload", None)):
                raise HostWeightModelLevelError(
                    "a generic host-weight session cannot be combined with a pipeline-owned "
                    "model offload implementation"
                )

            # Discovery is part of the prepared-session transaction: even a
            # pipeline protocol failure must release the artifact lease.
            modules = ModuleDiscovery.discover(pipeline)
            if self.config.use_hsdp:
                raise HostWeightModelLevelError(
                    "model-level host-weight session does not support HSDP in the single-node v1 path"
                )
            if not modules.dits:
                raise HostWeightModelLevelError(
                    "host-weight session was supplied, but no DiT/transformer module was discovered"
                )
            if not modules.encoders:
                raise HostWeightModelLevelError(
                    "model-level host-weight offload requires an encoder/DiT swap boundary; "
                    "without encoders the source path would not provide offload"
                )

            prepared = self._prepared_weight_session
            if prepared is None:
                raise HostWeightModelLevelError("prepared host-weight session was already consumed")
            _validate_session_capabilities(prepared, plan_kind=TransferPlanKind.COMPONENT)
            bindings = self._prepare_host_weight_bindings(
                modules.dit_names,
                modules.dits,
                prepared,
            )
            self._host_weight_controller = _HostWeightComponentController(
                bindings,
                device=self.device,
                pin_memory=self.config.pin_cpu_memory,
                fail_closed=self._terminate_host_weight_session,
            )

            # All source/unit validation and staging allocation completed above.
            # From this point any error performs terminal fail-closed cleanup.
            for encoder in modules.encoders:
                encoder.to(self.device)

            for vae in modules.vaes:
                try:
                    vae.to(self.device, non_blocking=True)
                except Exception as exc:
                    logger.debug("Failed to move VAE to GPU: %s", exc)

            for resident, name in zip(modules.resident_modules, modules.resident_names):
                if any(resident is child for dit in modules.dits for child in dit.modules()):
                    logger.debug(
                        "Resident module %s is inside a source-managed DiT and follows its component lifecycle",
                        name,
                    )
                    continue
                try:
                    resident.to(self.device)
                except Exception as exc:
                    logger.warning("Failed to move resident module '%s' to GPU: %s", name, exc)

            self._offload_modules = [*modules.dits, *modules.encoders]
            apply_sequential_offload(
                dit_modules=modules.dits,
                encoder_modules=modules.encoders,
                device=self.device,
                pin_memory=self.config.pin_cpu_memory,
                use_hsdp=False,
                host_weight_controller=self._host_weight_controller,
            )

            # The binder commit is deliberately after complete backend
            # preflight and hook installation.  The prepared transaction owns
            # rollback until this non-failing transition succeeds.
            # Publish the active owner before clearing the prepared owner.  No
            # fallible backend work may occur in that handoff window.
            self._weight_session = prepared.commit()
            self._adopt_committed_weight_session()
            self._host_weight_controller.attach(self._weight_session)
            self.enabled = True
            logger.info(
                "Model-level host-weight offloading enabled: %s <-> %s",
                ", ".join(modules.dit_names),
                ", ".join(modules.encoder_names),
            )
        except BaseException as primary_error:
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
                    f"model-level host-weight {action}",
                    cleanup_error,
                )
            raise

    def _adopt_committed_weight_session(self) -> None:
        prepared = self._prepared_weight_session
        session = self._weight_session
        if prepared is None or session is None:
            return
        prepared.adopt(session)
        if self._prepared_weight_session is prepared:
            self._prepared_weight_session = None

    @classmethod
    def _prepare_host_weight_bindings(
        cls,
        component_paths: Sequence[str],
        components: Sequence[nn.Module],
        prepared: PreparedWeightAccessSession,
    ) -> tuple[_HostWeightComponentBinding, ...]:
        if len(component_paths) != len(components):
            raise HostWeightModelLevelError("DiT component names and modules have different lengths")
        if len(components) != 1:
            raise HostWeightModelLevelError(
                "v1 model-level host-weight offload requires exactly one managed DiT target"
            )

        plan = prepared.transfer_plan
        if len(plan.execution_bindings) != 1:
            raise HostWeightModelLevelError("component transfer plan must contain exactly one execution binding")
        execution = plan.execution_bindings[0]
        if str(execution.module_path) != ".":
            raise HostWeightModelLevelError("component transfer plan must bind the managed target root")
        if plan.unit_ids != (execution.unit_id,):
            raise HostWeightModelLevelError("component transfer plan must contain only its root execution unit")

        unit = prepared.unit_spec(execution.unit_id)
        if _enum_value(unit.unit_kind) != UnitKind.COMPONENT.value:
            raise HostWeightModelLevelError(f"component execution binding selects non-component unit {unit.unit_id!r}")
        if not unit.planes or not unit.bindings:
            raise HostWeightModelLevelError(f"component transfer unit {unit.unit_id!r} is empty")

        return (
            _HostWeightComponentBinding(
                component_path=component_paths[0],
                component=components[0],
                unit=unit,
            ),
        )

    def _rollback_prepared_weight_session(self) -> None:
        prepared = self._prepared_weight_session
        if prepared is None:
            return
        prepared.rollback()
        if self._prepared_weight_session is prepared:
            self._prepared_weight_session = None

    def _terminate_host_weight_session(self) -> None:
        self._host_weight_terminal = True
        if self._host_weight_teardown_phase is _SessionTeardownPhase.CLOSED:
            return

        if self._host_weight_teardown_phase is _SessionTeardownPhase.ACTIVE:
            controller = self._host_weight_controller
            if controller is not None:
                controller.close()
                if self._host_weight_controller is controller:
                    self._host_weight_controller = None

            session = self._weight_session
            if session is not None:
                # Suspension is the transaction boundary.  It is crossed only
                # once; later retries resume from QUIESCED.
                session.suspend()
            self._host_weight_teardown_phase = _SessionTeardownPhase.QUIESCED

        while self._offload_modules:
            module = self._offload_modules[0]
            remove_sequential_offload([module])
            del self._offload_modules[0]

        session = self._weight_session
        if session is not None:
            session.close(DetachMode.TERMINAL)
            if self._weight_session is session:
                self._weight_session = None

        self.enabled = False
        self._host_weight_teardown_phase = _SessionTeardownPhase.CLOSED

    def disable(self) -> None:
        if self._uses_weight_session:
            if self._prepared_weight_session is not None:
                self._rollback_prepared_weight_session()
            if self._host_weight_teardown_phase is _SessionTeardownPhase.CLOSED:
                return
            self._terminate_host_weight_session()
            logger.info("Model-level host-weight offloading disabled (terminal teardown)")
            return

        if not self.enabled:
            return

        if self._custom_pipeline is not None:
            custom_disable = getattr(self._custom_pipeline, "disable_omni_model_cpu_offload", None)
            if callable(custom_disable):
                custom_disable()
            self._custom_pipeline = None
            self.enabled = False
            logger.info("Model-level offloading disabled")
            return

        remove_sequential_offload(self._offload_modules)

        self._offload_modules.clear()
        self.enabled = False
        logger.info("Model-level offloading disabled")

    def host_weight_diagnostics(self) -> dict[str, object]:
        """Return allocation-only diagnostics for the session-backed adapter."""

        controller = self._host_weight_controller
        staging = () if controller is None else tuple(controller.staging.values())
        # Model-level activation synchronizes H2D before binding and retains no
        # asynchronous event handle.
        return {
            "pinned_slot_budget_bytes": _pinned_cpu_storage_bytes(staging),
            "events": 0,
        }

    def host_weight_session_idle_state(self) -> dict[str, object]:
        """Return session state without exposing the ownership-bearing handle."""

        session = self._weight_session
        if session is None:
            return {
                "outstanding_units": 0,
                "bindings": 0,
                "resident_bindings": 0,
                "total_bindings": 0,
            }
        return dict(session.idle_state())
