# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Public CPU-offload configuration helpers."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from enum import Enum
from typing import Any

DIT_COMPONENT = "dit"
TEXT_ENCODER_COMPONENT = "text_encoder"
ALL_COMPONENT = "all"
OFFLOAD_COMPONENTS = frozenset({DIT_COMPONENT, TEXT_ENCODER_COMPONENT})
OFFLOAD_COMPONENT_SELECTORS = OFFLOAD_COMPONENTS | {ALL_COMPONENT}
DEFAULT_OFFLOAD_COMPONENTS = frozenset({DIT_COMPONENT})
# Transfer settings remain DLO-specific even though component selection is not.
DLO_COMPONENTS = OFFLOAD_COMPONENTS


class OffloadStrategy(str, Enum):
    """CPU-offload scheduling policy selected independently of components."""

    NONE = "none"
    MODEL_LEVEL = "model"
    LAYER_WISE = "layerwise"
    DISTRIBUTED_LAYER_WISE = "distributed-layerwise"


_LEGACY_STRATEGY_FIELDS = {
    "enable_cpu_offload": OffloadStrategy.MODEL_LEVEL,
    "enable_layerwise_offload": OffloadStrategy.LAYER_WISE,
    "enable_distributed_layerwise_offload": OffloadStrategy.DISTRIBUTED_LAYER_WISE,
}


def parse_offload_strategy(value: str | OffloadStrategy) -> OffloadStrategy:
    """Normalize the canonical public strategy spelling."""
    if isinstance(value, OffloadStrategy):
        return value
    if not isinstance(value, str):
        raise TypeError(f"offload_strategy must be a string, got {type(value).__name__}")
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "model-level": OffloadStrategy.MODEL_LEVEL.value,
        "layer-wise": OffloadStrategy.LAYER_WISE.value,
        "distributed-layer-wise": OffloadStrategy.DISTRIBUTED_LAYER_WISE.value,
    }
    normalized = aliases.get(normalized, normalized)
    try:
        return OffloadStrategy(normalized)
    except ValueError as exc:
        choices = ", ".join(strategy.value for strategy in OffloadStrategy)
        raise ValueError(f"Unknown offload_strategy {value!r}; choose from: {choices}") from exc


def resolve_offload_strategy(config: Any) -> OffloadStrategy:
    """Resolve the canonical strategy and deprecated boolean aliases.

    The old ``enable_*_offload`` fields remain accepted for compatibility, but
    multiple legacy strategies and conflicts with ``offload_strategy`` fail
    before any hooks mutate the model.
    """
    raw_strategy = getattr(config, "offload_strategy", None)
    strategy = parse_offload_strategy(raw_strategy) if raw_strategy is not None else None
    enabled_aliases = [
        (field, alias_strategy)
        for field, alias_strategy in _LEGACY_STRATEGY_FIELDS.items()
        if bool(getattr(config, field, False))
    ]
    if len(enabled_aliases) > 1:
        fields = ", ".join(field for field, _ in enabled_aliases)
        raise ValueError(f"Conflicting legacy offload strategy flags: {fields}; use offload_strategy instead")
    if enabled_aliases:
        field, alias_strategy = enabled_aliases[0]
        if strategy is not None and strategy is not alias_strategy:
            raise ValueError(
                f"offload_strategy={strategy.value!r} conflicts with legacy {field}=True ({alias_strategy.value})"
            )
        return alias_strategy
    return strategy or OffloadStrategy.NONE


def uses_offload_strategy(config: Any, *strategies: OffloadStrategy) -> bool:
    """Return whether a diffusion-like config resolves to one of ``strategies``."""
    return resolve_offload_strategy(config) in strategies


def materialize_legacy_offload_flags(config: Any) -> OffloadStrategy:
    """Keep legacy boolean readers working with the canonical strategy API."""
    strategy = resolve_offload_strategy(config)
    for field, field_strategy in _LEGACY_STRATEGY_FIELDS.items():
        setattr(config, field, strategy is field_strategy)
    return strategy


def parse_offload_components(value: str | Collection[str] | None) -> frozenset[str]:
    """Normalize the strategy-independent public component selection."""
    if value is None:
        return DEFAULT_OFFLOAD_COMPONENTS
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, Collection):
        values = list(value)
    else:
        raise TypeError(
            f"offload_components must be a comma-separated string or a sequence of strings, got {type(value).__name__}"
        )
    if any(not isinstance(item, str) for item in values):
        raise TypeError("offload_components entries must be strings")
    components = [item.strip().lower().replace("-", "_") for item in values]
    if not components or any(not item for item in components):
        raise ValueError("offload_components must not be empty")
    unknown = sorted(set(components) - OFFLOAD_COMPONENT_SELECTORS)
    if unknown:
        choices = ", ".join(sorted(OFFLOAD_COMPONENT_SELECTORS))
        raise ValueError(f"Unknown offload component(s): {', '.join(unknown)}. Choose from: {choices}")
    if ALL_COMPONENT in components:
        return OFFLOAD_COMPONENTS
    return frozenset(components)


class DLOTransfer(str, Enum):
    """How one component's next block reaches the device."""

    ALLGATHER = "allgather"
    RANK_LOCAL = "rank-local"


def _parse_transfer(value: Any) -> DLOTransfer:
    if not isinstance(value, str):
        raise TypeError(f"DLO transfer values must be strings, got {type(value).__name__}")
    normalized = value.strip().lower().replace("_", "-")
    try:
        return DLOTransfer(normalized)
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in DLOTransfer)
        raise ValueError(f"Unknown DLO transfer {value!r}; choose from: {choices}") from exc


def parse_dlo_transfer(
    value: str | Mapping[str, str] | None,
    *,
    legacy_use_allgather: bool = True,
) -> dict[str, DLOTransfer]:
    """Resolve a scalar or ``component=value`` transfer specification.

    A scalar applies to both supported components. A mapping may be passed as
    either a Python mapping or a comma-separated CLI value. Missing component
    entries inherit the legacy ``dlo_use_allgather`` setting.
    """
    fallback = DLOTransfer.ALLGATHER if legacy_use_allgather else DLOTransfer.RANK_LOCAL
    resolved = {component: fallback for component in DLO_COMPONENTS}
    if value is None:
        return resolved

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ValueError("dlo_transfer must not be empty")
        if "=" not in raw:
            transfer = _parse_transfer(raw)
            return {component: transfer for component in DLO_COMPONENTS}
        entries: list[tuple[str, str]] = []
        for item in raw.split(","):
            key, separator, raw_mode = item.partition("=")
            if not separator or not key.strip() or not raw_mode.strip():
                raise ValueError("dlo_transfer must be a transfer name or a comma-separated component=transfer map")
            entries.append((key, raw_mode))
    elif isinstance(value, Mapping):
        entries = list(value.items())
    else:
        raise TypeError(f"dlo_transfer must be a transfer string or component mapping, got {type(value).__name__}")

    seen: set[str] = set()
    for raw_component, raw_transfer in entries:
        if not isinstance(raw_component, str):
            raise TypeError("dlo_transfer component names must be strings")
        component = raw_component.strip().lower().replace("-", "_")
        if component == "all":
            targets = tuple(DLO_COMPONENTS)
        elif component in DLO_COMPONENTS:
            targets = (component,)
        else:
            choices = ", ".join(sorted((*DLO_COMPONENTS, "all")))
            raise ValueError(f"Unknown DLO transfer component {raw_component!r}; choose from: {choices}")
        transfer = _parse_transfer(raw_transfer)
        for target in targets:
            if target in seen:
                raise ValueError(f"Duplicate dlo_transfer entry for component {target!r}")
            resolved[target] = transfer
            seen.add(target)
    return resolved


def component_uses_allgather(config: Any, component: str = DIT_COMPONENT) -> bool:
    """Read a component transfer from an Omni diffusion-like config object."""
    transfers = parse_dlo_transfer(
        getattr(config, "dlo_transfer", None),
        legacy_use_allgather=bool(getattr(config, "dlo_use_allgather", True)),
    )
    try:
        return transfers[component] is DLOTransfer.ALLGATHER
    except KeyError as exc:
        raise ValueError(f"Unknown DLO component {component!r}") from exc


def selected_offload_components(config: Any) -> frozenset[str]:
    """Resolve the public component selector without importing a backend."""
    return parse_offload_components(getattr(config, "offload_components", None))


def any_selected_component_uses_allgather(config: Any) -> bool:
    """Return whether an enabled DLO component requires collectives."""
    if not uses_offload_strategy(config, OffloadStrategy.DISTRIBUTED_LAYER_WISE):
        return False
    return any(component_uses_allgather(config, component) for component in selected_offload_components(config))
