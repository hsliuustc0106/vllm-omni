# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit built-in composition for the independent Host Weight Runtime."""

from __future__ import annotations

from .backings import (
    BackingProvider,
    BackingProviderRegistry,
    LoadedTensorBackingProvider,
    StaticBackingProviderRegistry,
)
from .runtime import HostWeightRuntime
from .store import LocalArtifactRepository, LocalRuntimeMmapBackingProvider


def create_default_backing_provider_registry(
    repository: LocalArtifactRepository | None = None,
) -> BackingProviderRegistry:
    """Compose the v1 loaded-tensor and optional local-mmap providers."""

    providers: list[BackingProvider] = [LoadedTensorBackingProvider()]
    if repository is not None:
        providers.insert(
            0,
            LocalRuntimeMmapBackingProvider(
                repository,
                # LocalArtifactRepository validates every object and digest
                # during lookup/claim and again during commit. The opener
                # repeats safe-path, type, and size checks but avoids a second
                # multi-GiB digest scan immediately before mmap.
                verify_integrity=False,
            ),
        )
    return StaticBackingProviderRegistry(providers)


def create_default_host_weight_runtime(
    repository: LocalArtifactRepository | None = None,
    *,
    writable: bool = True,
    verify_mmap_integrity: bool = True,
) -> HostWeightRuntime:
    """Create the public built-in single-node composition explicitly.

    ``verify_mmap_integrity=False`` remains rejected in v1. Integrity is
    enforced by the local repository before its provider opens a mapping.
    """

    if not verify_mmap_integrity:
        raise ValueError("v1 requires physical runtime-mmap integrity verification")
    return HostWeightRuntime(
        repository,
        create_default_backing_provider_registry(repository),
        writable=writable,
    )


__all__ = [
    "create_default_backing_provider_registry",
    "create_default_host_weight_runtime",
]
