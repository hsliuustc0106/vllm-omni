# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace
from weakref import WeakKeyDictionary

import pytest
from torch import nn

from vllm_omni.diffusion.host_weight.formats import fp8_per_tensor as fp8_module


class _LayerwiseInfo:
    def __init__(self) -> None:
        self.kernel_tensors = None
        self.loaded_weights: list[object] = []
        self.load_numel = 0
        self.load_numel_total = 1

    def can_load(self) -> bool:
        return True


class _DeleteThenFailRegistry(WeakKeyDictionary[nn.Module, object]):
    def __init__(self) -> None:
        super().__init__()
        self.fail_after_next_delete = False

    def __delitem__(self, key: nn.Module) -> None:
        super().__delitem__(key)
        if self.fail_after_next_delete:
            self.fail_after_next_delete = False
            raise MemoryError("injected failure after registry deletion")


class _RestoreThenFailRegistry(WeakKeyDictionary[nn.Module, object]):
    def __init__(self) -> None:
        super().__init__()
        self.fail_after_next_restore = False

    def __setitem__(self, key: nn.Module, value: object) -> None:
        super().__setitem__(key, value)
        if self.fail_after_next_restore:
            self.fail_after_next_restore = False
            raise MemoryError("injected failure after registry restoration")


def _target_and_recipe() -> tuple[nn.Module, object]:
    target = nn.Module()
    target.first = nn.Linear(1, 1, bias=False)
    target.second = nn.Linear(1, 1, bias=False)
    recipe = SimpleNamespace(
        layers=(
            SimpleNamespace(module_path="first"),
            SimpleNamespace(module_path="second"),
        )
    )
    return target, recipe


def _capture(
    monkeypatch: pytest.MonkeyPatch,
    registry: WeakKeyDictionary[nn.Module, object],
) -> tuple[object, nn.Module, nn.Module, _LayerwiseInfo, _LayerwiseInfo]:
    target, recipe = _target_and_recipe()
    first = target.first
    second = target.second
    first_info = _LayerwiseInfo()
    second_info = _LayerwiseInfo()
    registry[first] = first_info
    registry[second] = second_info
    monkeypatch.setattr(
        fp8_module,
        "_online_layerwise_api",
        lambda: (registry, _LayerwiseInfo),
    )
    adapter = fp8_module.Fp8PerTensorFormatAdapter.__new__(fp8_module.Fp8PerTensorFormatAdapter)
    retirement = adapter.retire_online_loader_state(target, recipe)
    return retirement, first, second, first_info, second_info


def test_apply_failure_after_delete_keeps_exact_rollback_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _DeleteThenFailRegistry()
    retirement, first, second, first_info, second_info = _capture(monkeypatch, registry)

    # Capture is deliberately mutation-free; the binder can retain the exact
    # transaction before calling apply().
    assert registry[first] is first_info
    assert registry[second] is second_info

    registry.fail_after_next_delete = True
    with pytest.raises(MemoryError, match="after registry deletion"):
        retirement.apply()

    assert first not in registry
    assert registry[second] is second_info
    with pytest.raises(fp8_module.Fp8FormatError, match="not fully applied"):
        retirement.validate_quiesced()
    with pytest.raises(fp8_module.Fp8FormatError, match="not fully applied"):
        retirement.commit()

    retirement.rollback()
    assert registry[first] is first_info
    assert registry[second] is second_info
    retirement.rollback()


def test_failed_restore_retains_progress_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _RestoreThenFailRegistry()
    retirement, first, second, first_info, second_info = _capture(monkeypatch, registry)
    retirement.apply()
    assert first not in registry
    assert second not in registry

    # __setitem__ mutates first and then raises. The transaction must retain
    # that snapshot, making a second idempotent restoration attempt safe.
    registry.fail_after_next_restore = True
    with pytest.raises(MemoryError, match="after registry restoration"):
        retirement.rollback()
    assert registry[second] is second_info
    assert first not in registry

    retirement.rollback()
    assert registry[first] is first_info
    assert registry[second] is second_info
    retirement.rollback()
