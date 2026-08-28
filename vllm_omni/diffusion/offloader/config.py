# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Public configuration helpers for distributed layerwise offload."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any

DIT_COMPONENT = "dit"
TEXT_ENCODER_COMPONENT = "text_encoder"
DLO_COMPONENTS = frozenset({DIT_COMPONENT, TEXT_ENCODER_COMPONENT})


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


def selected_dlo_components(config: Any) -> frozenset[str]:
    """Resolve the public component selector without importing the backend."""
    value = getattr(config, "layerwise_offload_components", None)
    if value is None:
        return frozenset({DIT_COMPONENT})
    values = value.split(",") if isinstance(value, str) else list(value)
    components = {str(item).strip().lower().replace("-", "_") for item in values}
    if "all" in components:
        return DLO_COMPONENTS
    return frozenset(components)


def any_selected_component_uses_allgather(config: Any) -> bool:
    """Return whether an enabled DLO component requires collectives."""
    if not getattr(config, "enable_distributed_layerwise_offload", False):
        return False
    return any(component_uses_allgather(config, component) for component in selected_dlo_components(config))
