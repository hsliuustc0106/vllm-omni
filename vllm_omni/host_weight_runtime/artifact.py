# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolved-artifact and synchronous-view lifetime management."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping

import torch

from .backings import BackingClosedError, WeightBacking
from .contracts import ArtifactManifest, TensorSelection, TensorSpec


class ResolvedArtifact:
    """Own one verified backing and hand out strongly-retained views."""

    def __init__(
        self,
        backing: WeightBacking,
        *,
        on_closed: Callable[[], None] | None = None,
    ) -> None:
        self._backing = backing
        self._manifest = backing.manifest
        self._view_registrations: dict[object, bool] = {}
        self._close_requested = False
        self._backing_closed = False
        self._on_closed = on_closed
        self._close_callback_complete = on_closed is None
        self._lock = threading.RLock()

    @property
    def manifest(self) -> ArtifactManifest:
        return self._manifest

    @property
    def close_requested(self) -> bool:
        with self._lock:
            return self._close_requested

    @property
    def active_views(self) -> int:
        with self._lock:
            return len(self._view_registrations)

    def open(self, selection: TensorSelection) -> WeightView:
        known = set(self._manifest.tensor_ids)
        if missing := [item for item in selection.tensor_ids if item not in known]:
            raise KeyError(f"selection contains unknown tensor ids: {missing}")
        with self._lock:
            if self._close_requested:
                raise BackingClosedError("resolved artifact is closing")
            registration = object()

            def release() -> None:
                self._release_view(registration)

            try:
                # The identity reservation makes rollback and an unpublished
                # view destructor idempotent even when other views are live.
                self._view_registrations[registration] = False
                view = WeightView(
                    self,
                    selection,
                    _release_reference=release,
                )
                if registration not in self._view_registrations:
                    raise RuntimeError("host-weight view closed before registration activation")
                self._view_registrations[registration] = True
                return view
            except BaseException:
                release()
                raise

    def _copy_into(self, tensor_id: str, destination: torch.Tensor) -> None:
        with self._lock:
            if self._backing_closed:
                raise BackingClosedError("resolved artifact backing is closed")
            backing = self._backing
        backing.copy_into(tensor_id, destination)

    def _complete_close_locked(self) -> None:
        """Advance backing and callback cleanup without losing retry state."""
        if not self._backing_closed:
            self._backing.close()
            self._backing_closed = True
        if not self._close_callback_complete:
            callback = self._on_closed
            assert callback is not None
            callback()
            self._on_closed = None
            self._close_callback_complete = True

    def _release_view(self, registration: object) -> None:
        with self._lock:
            if registration not in self._view_registrations:
                return
            if self._close_requested and len(self._view_registrations) == 1:
                # Retain the last view reference when either backing or
                # callback cleanup fails so the same view can retry safely.
                self._complete_close_locked()
            del self._view_registrations[registration]

    def close(self) -> None:
        with self._lock:
            self._close_requested = True
            if not self._view_registrations:
                self._complete_close_locked()

    def __enter__(self) -> ResolvedArtifact:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class WeightView:
    """A complete-tensor, synchronous, read-only selection."""

    def __init__(
        self,
        artifact: ResolvedArtifact,
        selection: TensorSelection,
        *,
        _release_reference: Callable[[], None],
    ) -> None:
        self._artifact = artifact
        self.selection = selection
        self._release_reference = _release_reference
        self._closed = False
        self._lock = threading.RLock()

    @property
    def tensors(self) -> tuple[TensorSpec, ...]:
        return tuple(self._artifact.manifest.tensor(tensor_id) for tensor_id in self.selection.tensor_ids)

    @property
    def tensor_specs(self) -> tuple[TensorSpec, ...]:
        """Compatibility spelling for integration code; same immutable tuple."""
        return self.tensors

    def copy_into(self, destinations: Mapping[str, torch.Tensor]) -> None:
        with self._lock:
            if self._closed:
                raise BackingClosedError("host-weight view is closed")
            expected = set(self.selection.tensor_ids)
            if set(destinations) != expected:
                missing = sorted(expected - set(destinations))
                extra = sorted(set(destinations) - expected)
                raise ValueError(f"destination ids do not match selection; missing={missing}, extra={extra}")
            for tensor_id in self.selection.tensor_ids:
                self._artifact._copy_into(tensor_id, destinations[tensor_id])

    def copy_tensor_into(self, tensor_id: str, destination: torch.Tensor) -> None:
        with self._lock:
            if self._closed:
                raise BackingClosedError("host-weight view is closed")
            if tensor_id not in self.selection.tensor_ids:
                raise KeyError(f"tensor {tensor_id!r} is outside this view")
            self._artifact._copy_into(tensor_id, destination)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._release_reference()
            self._closed = True

    def __enter__(self) -> WeightView:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["ResolvedArtifact", "WeightView"]
