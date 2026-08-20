# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc

import pytest
import torch
from conftest import ArtifactRegistrar

from vllm_omni.host_weight_runtime import (
    AccessFeature,
    AccessRequirements,
    BackingClosedError,
    BackingError,
    BackingIntegrityError,
    BackingKind,
    CapabilityGrant,
    LoadedTensorBacking,
    Ready,
    ResolutionPath,
    RuntimeBusy,
    RuntimeClosed,
    TensorSelection,
)
from vllm_omni.host_weight_runtime import (
    create_default_host_weight_runtime as HostWeightRuntime,
)


def _resolve_loaded(runtime, artifact_spec, tensors):
    grant = runtime.negotiate(
        AccessRequirements(
            frozenset({AccessFeature.COMPLETE_TENSOR_READ}),
            frozenset({BackingKind.LOADED_TENSOR}),
        )
    )
    assert isinstance(grant, CapabilityGrant)
    registrar = ArtifactRegistrar()
    outcome = runtime.resolve_loaded(artifact_spec, grant, tensors, registrar)
    if isinstance(outcome, Ready):
        assert registrar.artifact is outcome.artifact
    return outcome


def test_loaded_runtime_copies_a_complete_tensor(artifact_spec) -> None:
    runtime = HostWeightRuntime()
    source = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4))
    outcome = _resolve_loaded(runtime, artifact_spec, {"block.weight": source})

    assert isinstance(outcome, Ready)
    assert outcome.info.path is ResolutionPath.LOADED
    assert outcome.info.backing_kind is BackingKind.LOADED_TENSOR
    destination = torch.empty_like(source)
    with outcome.artifact.open(TensorSelection.one("block.weight")) as view:
        assert view.tensors == view.tensor_specs
        view.copy_into({"block.weight": destination})
    torch.testing.assert_close(destination, source)
    outcome.artifact.close()
    runtime.close()


def test_loaded_backing_detects_mutation_after_resolution(artifact_spec) -> None:
    runtime = HostWeightRuntime()
    source = torch.arange(4, dtype=torch.float32)
    outcome = _resolve_loaded(runtime, artifact_spec, {"weight": source})
    assert isinstance(outcome, Ready)
    source.add_(1)

    with outcome.artifact.open(TensorSelection.one("weight")) as view:
        with pytest.raises(BackingIntegrityError, match="mutated"):
            view.copy_into({"weight": torch.empty_like(source)})
    outcome.artifact.close()
    runtime.close()


def test_loaded_backing_rejects_shared_storage(artifact_spec) -> None:
    storage = torch.arange(8, dtype=torch.float32)

    with pytest.raises(BackingError, match="share storage"):
        LoadedTensorBacking.from_tensors(
            artifact_spec,
            {"first": storage[:4], "second": storage[4:]},
        )


def test_view_retains_backing_and_keeps_runtime_busy(artifact_spec) -> None:
    runtime = HostWeightRuntime()
    outcome = _resolve_loaded(
        runtime,
        artifact_spec,
        {"weight": torch.arange(4, dtype=torch.float32)},
    )
    assert isinstance(outcome, Ready)
    view = outcome.artifact.open(TensorSelection.one("weight"))

    outcome.artifact.close()
    with pytest.raises(RuntimeBusy, match="resolved artifact"):
        runtime.close()
    destination = torch.empty(4)
    view.copy_into({"weight": destination})
    torch.testing.assert_close(destination, torch.arange(4, dtype=torch.float32))
    with pytest.raises(BackingClosedError, match="closing"):
        outcome.artifact.open(TensorSelection.one("weight"))

    view.close()
    runtime.close()
    runtime.close()
    with pytest.raises(RuntimeClosed, match="closed"):
        runtime.negotiate(
            AccessRequirements(
                frozenset({AccessFeature.COMPLETE_TENSOR_READ}),
                frozenset({BackingKind.LOADED_TENSOR}),
            )
        )


def test_runtime_close_rejects_directly_live_artifact(artifact_spec) -> None:
    runtime = HostWeightRuntime()
    outcome = _resolve_loaded(
        runtime,
        artifact_spec,
        {"weight": torch.arange(1, dtype=torch.float32)},
    )
    assert isinstance(outcome, Ready)

    with pytest.raises(RuntimeBusy):
        runtime.close()
    outcome.artifact.close()
    runtime.close()


def test_registrar_owns_artifact_before_ready_return_window_can_be_lost(
    artifact_spec,
) -> None:
    runtime = HostWeightRuntime()
    grant = runtime.negotiate(
        AccessRequirements(
            frozenset({AccessFeature.COMPLETE_TENSOR_READ}),
            frozenset({BackingKind.LOADED_TENSOR}),
        )
    )
    assert isinstance(grant, CapabilityGrant)
    registrar = ArtifactRegistrar()

    outcome = runtime.resolve_loaded(
        artifact_spec,
        grant,
        {"weight": torch.arange(4, dtype=torch.float32)},
        registrar,
    )
    assert isinstance(outcome, Ready)
    borrowed = outcome.artifact
    assert registrar.artifact is borrowed

    # Simulate interruption after the core method returns but before a caller
    # can retain the outcome.  The preinstalled registrar is already the exact
    # owner and the borrowed artifact remains usable.
    del outcome, borrowed
    gc.collect()
    destination = torch.empty(4, dtype=torch.float32)
    with registrar.artifact.open(TensorSelection.one("weight")) as view:
        view.copy_into({"weight": destination})
    torch.testing.assert_close(destination, torch.arange(4, dtype=torch.float32))

    registrar.artifact.close()
    runtime.close()


def test_failed_registrar_handoff_retains_artifact_guard_for_close_retry(
    artifact_spec,
    monkeypatch,
) -> None:
    runtime = HostWeightRuntime()
    backing = LoadedTensorBacking.from_tensors(
        artifact_spec,
        {"weight": torch.arange(1, dtype=torch.float32)},
    )
    original_close = backing.close
    close_calls = 0

    def fail_close_once() -> None:
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise RuntimeError("injected backing close failure")
        original_close()

    class RejectingRegistrar:
        def adopt_artifact(self, _artifact) -> None:
            raise RuntimeError("injected registrar adoption failure")

    monkeypatch.setattr(backing, "close", fail_close_once)

    with pytest.raises(RuntimeError, match="registrar adoption failure") as caught:
        runtime._register(backing, RejectingRegistrar())

    assert close_calls == 1
    assert runtime.pending_artifact_assemblies == 1
    assert runtime.active_artifacts == 1
    assert any("artifact cleanup" in note for note in caught.value.__notes__)
    with pytest.raises(RuntimeBusy, match="artifact cleanup remains pending"):
        _resolve_loaded(
            runtime,
            artifact_spec,
            {"weight": torch.arange(1, dtype=torch.float32)},
        )

    runtime.close()
    assert close_calls == 2
    assert runtime.pending_artifact_assemblies == 0
    assert runtime.active_artifacts == 0


def test_register_construction_base_exception_closes_backing_without_liveness_leak(
    artifact_spec,
    monkeypatch,
) -> None:
    import vllm_omni.host_weight_runtime.runtime as runtime_module

    class ConstructionFailure(BaseException):
        pass

    runtime = HostWeightRuntime()
    backing = LoadedTensorBacking.from_tensors(
        artifact_spec,
        {"weight": torch.arange(1, dtype=torch.float32)},
    )
    original_close = backing.close
    close_calls = 0
    primary = ConstructionFailure()

    def record_close() -> None:
        nonlocal close_calls
        close_calls += 1
        original_close()

    def fail_construction(*_args, **_kwargs):
        raise primary

    monkeypatch.setattr(backing, "close", record_close)
    monkeypatch.setattr(runtime_module, "ResolvedArtifact", fail_construction)

    with pytest.raises(ConstructionFailure) as caught:
        runtime._register(backing, ArtifactRegistrar())

    assert caught.value is primary
    assert close_calls == 1
    assert runtime.active_artifacts == 0
    runtime.close()


def test_unpublished_artifact_destructor_releases_only_its_reservation(
    artifact_spec,
    monkeypatch,
) -> None:
    import vllm_omni.host_weight_runtime.runtime as runtime_module

    class ConstructionBoundaryFailure(BaseException):
        pass

    runtime = HostWeightRuntime()
    backing = LoadedTensorBacking.from_tensors(
        artifact_spec,
        {"weight": torch.arange(1, dtype=torch.float32)},
    )
    original_artifact = runtime_module.ResolvedArtifact
    original_close = backing.close
    primary = ConstructionBoundaryFailure()
    close_calls = 0

    def record_close() -> None:
        nonlocal close_calls
        close_calls += 1
        original_close()

    def construct_destroy_then_fail(*args, **kwargs):
        unpublished = original_artifact(*args, **kwargs)
        del unpublished
        gc.collect()
        raise primary

    monkeypatch.setattr(backing, "close", record_close)
    monkeypatch.setattr(runtime_module, "ResolvedArtifact", construct_destroy_then_fail)

    with pytest.raises(ConstructionBoundaryFailure) as caught:
        runtime._register(backing, ArtifactRegistrar())

    assert caught.value is primary
    assert close_calls == 1
    assert runtime.active_artifacts == 0
    runtime.close()


def test_view_construction_memory_error_does_not_publish_view_reference(
    artifact_spec,
    monkeypatch,
) -> None:
    import vllm_omni.host_weight_runtime.artifact as artifact_module

    runtime = HostWeightRuntime()
    outcome = _resolve_loaded(
        runtime,
        artifact_spec,
        {"weight": torch.arange(1, dtype=torch.float32)},
    )
    assert isinstance(outcome, Ready)
    primary = MemoryError("injected weight-view allocation failure")

    def fail_construction(*_args, **_kwargs):
        raise primary

    monkeypatch.setattr(artifact_module, "WeightView", fail_construction)

    with pytest.raises(MemoryError) as caught:
        outcome.artifact.open(TensorSelection.one("weight"))

    assert caught.value is primary
    assert outcome.artifact.active_views == 0
    outcome.artifact.close()
    assert runtime.active_artifacts == 0
    runtime.close()


def test_unpublished_view_destructor_does_not_release_existing_live_view(
    artifact_spec,
    monkeypatch,
) -> None:
    import vllm_omni.host_weight_runtime.artifact as artifact_module

    runtime = HostWeightRuntime()
    outcome = _resolve_loaded(
        runtime,
        artifact_spec,
        {"weight": torch.arange(1, dtype=torch.float32)},
    )
    assert isinstance(outcome, Ready)
    live_view = outcome.artifact.open(TensorSelection.one("weight"))
    original_view = artifact_module.WeightView
    primary = MemoryError("injected post-view-construction interruption")

    def construct_destroy_then_fail(*args, **kwargs):
        unpublished = original_view(*args, **kwargs)
        del unpublished
        gc.collect()
        raise primary

    monkeypatch.setattr(artifact_module, "WeightView", construct_destroy_then_fail)

    with pytest.raises(MemoryError) as caught:
        outcome.artifact.open(TensorSelection.one("weight"))

    assert caught.value is primary
    assert outcome.artifact.active_views == 1
    destination = torch.empty(1)
    live_view.copy_into({"weight": destination})
    torch.testing.assert_close(destination, torch.arange(1, dtype=torch.float32))

    live_view.close()
    assert outcome.artifact.active_views == 0
    outcome.artifact.close()
    assert runtime.active_artifacts == 0
    runtime.close()


def test_backing_close_failure_is_retryable_before_releasing_runtime_liveness(
    artifact_spec,
    monkeypatch,
) -> None:
    runtime = HostWeightRuntime()
    outcome = _resolve_loaded(
        runtime,
        artifact_spec,
        {"weight": torch.arange(1, dtype=torch.float32)},
    )
    assert isinstance(outcome, Ready)
    close_calls = 0
    original_close = outcome.artifact._backing.close

    def fail_close_once() -> None:
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise RuntimeError("injected backing close failure")
        original_close()

    monkeypatch.setattr(outcome.artifact._backing, "close", fail_close_once)

    with pytest.raises(RuntimeError, match="injected backing close failure"):
        outcome.artifact.close()

    assert close_calls == 1
    assert runtime.active_artifacts == 1
    with pytest.raises(RuntimeBusy):
        runtime.close()

    outcome.artifact.close()
    assert close_calls == 2
    assert runtime.active_artifacts == 0
    outcome.artifact.close()
    assert close_calls == 2
    runtime.close()


def test_last_view_release_retries_backing_close_without_underflow(
    artifact_spec,
    monkeypatch,
) -> None:
    runtime = HostWeightRuntime()
    outcome = _resolve_loaded(
        runtime,
        artifact_spec,
        {"weight": torch.arange(1, dtype=torch.float32)},
    )
    assert isinstance(outcome, Ready)
    view = outcome.artifact.open(TensorSelection.one("weight"))
    original_close = outcome.artifact._backing.close
    close_calls = 0

    def fail_close_once() -> None:
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise RuntimeError("injected last-view close failure")
        original_close()

    monkeypatch.setattr(outcome.artifact._backing, "close", fail_close_once)
    outcome.artifact.close()

    with pytest.raises(RuntimeError, match="injected last-view close failure"):
        view.close()

    assert outcome.artifact.active_views == 1
    assert runtime.active_artifacts == 1
    view.copy_into({"weight": torch.empty(1)})

    view.close()
    assert outcome.artifact.active_views == 0
    assert runtime.active_artifacts == 0
    assert close_calls == 2
    view.close()
    assert close_calls == 2
    runtime.close()


@pytest.mark.parametrize("release_from_last_view", [False, True])
def test_close_retries_liveness_callback_without_reclosing_backing(
    artifact_spec,
    monkeypatch,
    release_from_last_view: bool,
) -> None:
    runtime = HostWeightRuntime()
    outcome = _resolve_loaded(
        runtime,
        artifact_spec,
        {"weight": torch.arange(1, dtype=torch.float32)},
    )
    assert isinstance(outcome, Ready)
    artifact = outcome.artifact
    view = artifact.open(TensorSelection.one("weight")) if release_from_last_view else None
    original_backing_close = artifact._backing.close
    original_callback = artifact._on_closed
    assert original_callback is not None
    backing_close_calls = 0
    callback_attempts = 0
    successful_releases = 0

    def record_backing_close() -> None:
        nonlocal backing_close_calls
        backing_close_calls += 1
        original_backing_close()

    def fail_callback_once() -> None:
        nonlocal callback_attempts, successful_releases
        callback_attempts += 1
        if callback_attempts == 1:
            raise RuntimeError("injected liveness callback failure")
        original_callback()
        successful_releases += 1

    monkeypatch.setattr(artifact._backing, "close", record_backing_close)
    artifact._on_closed = fail_callback_once
    if view is not None:
        artifact.close()
        closer = view.close
    else:
        closer = artifact.close

    with pytest.raises(RuntimeError, match="injected liveness callback failure"):
        closer()

    assert backing_close_calls == 1
    assert callback_attempts == 1
    assert successful_releases == 0
    assert runtime.active_artifacts == 1
    if view is not None:
        assert artifact.active_views == 1

    closer()
    assert backing_close_calls == 1
    assert callback_attempts == 2
    assert successful_releases == 1
    assert runtime.active_artifacts == 0
    if view is not None:
        assert artifact.active_views == 0
    closer()
    assert (backing_close_calls, callback_attempts, successful_releases) == (1, 2, 1)
    runtime.close()
