# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable source-identity tests for the MiniMax-H3 artifact producer."""

import hashlib
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.host_weight import producer as producer_module


class _FakeProducerLoader:
    def __init__(self, *, has_online_quant: bool, marked: int = 0) -> None:
        self.has_online_quant = has_online_quant
        self.marked = marked
        self.requested: list[nn.Module] = []

    def _has_online_quant(self, _model: nn.Module) -> bool:
        return self.has_online_quant

    def _request_offload_after_quant(self, model: nn.Module) -> int:
        self.requested.append(model)
        return self.marked


def test_streaming_fp8_does_not_require_quantizer_offload_hook() -> None:
    model = nn.Module()
    loader = _FakeProducerLoader(has_online_quant=True, marked=0)

    assert producer_module._prepare_streaming_online_quant(loader, model) == 0
    assert loader.requested == [model]


def test_streaming_fp8_still_requires_online_quantized_layers() -> None:
    model = nn.Module()
    loader = _FakeProducerLoader(has_online_quant=False)

    with pytest.raises(
        producer_module.DiffusionArtifactProducerError,
        match="found no online-quantized layers",
    ):
        producer_module._prepare_streaming_online_quant(loader, model)
    assert loader.requested == []


def test_build_session_close_retries_only_unfinished_cleanup_steps(
    monkeypatch,
) -> None:
    session = producer_module._MiniMaxH3ArtifactBuildSession(
        od_config=object(),
        source=object(),
        device=torch.device("cuda:0"),
        quant_config=object(),
        format_adapter=object(),
        spec=object(),
        load_config=object(),
    )
    session._owned.append(object())
    calls = {"gc": 0, "empty_cache": 0}

    def flaky_gc() -> None:
        calls["gc"] += 1
        if calls["gc"] == 1:
            raise RuntimeError("injected gc failure")

    def flaky_empty_cache() -> None:
        calls["empty_cache"] += 1
        if calls["empty_cache"] == 1:
            raise RuntimeError("injected accelerator cache failure")

    monkeypatch.setattr(producer_module.gc, "collect", flaky_gc)
    monkeypatch.setattr(producer_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        producer_module.torch.accelerator,
        "empty_cache",
        flaky_empty_cache,
    )

    with pytest.raises(RuntimeError, match="injected gc failure"):
        session.close()
    assert session._owned == []
    assert calls == {"gc": 1, "empty_cache": 0}
    assert not session._closed

    with pytest.raises(RuntimeError, match="injected accelerator cache failure"):
        session.close()
    assert calls == {"gc": 2, "empty_cache": 1}
    assert not session._closed

    session.close()
    assert calls == {"gc": 2, "empty_cache": 2}
    assert session._closed
    session.close()
    assert calls == {"gc": 2, "empty_cache": 2}


def _link_shard_blob(partition_root, payload: bytes) -> str:
    transformer_root = partition_root / "transformer"
    shard = transformer_root / "model-00001-of-00001.safetensors"
    shard.unlink(missing_ok=True)
    blob_id = hashlib.sha256(payload).hexdigest()
    blob_root = partition_root.parents[2] / "blobs"
    blob_root.mkdir(parents=True, exist_ok=True)
    blob = blob_root / blob_id
    blob.write_bytes(payload)
    shard.symlink_to(os.path.relpath(blob, shard.parent))
    return blob_id


def _write_minimax_source(partition_root, *, link_hf_blob: bool) -> None:
    transformer_root = partition_root / "transformer"
    transformer_root.mkdir(parents=True)
    (partition_root / "model_index.json").write_text("{}", encoding="utf-8")
    (transformer_root / "config.json").write_text(
        '{"hidden_size": 128}',
        encoding="utf-8",
    )
    (transformer_root / "model.safetensors.index.json").write_text(
        '{"weight_map":{"transformer.weight":"model-00001-of-00001.safetensors"}}',
        encoding="utf-8",
    )
    if link_hf_blob:
        _link_shard_blob(partition_root, b"checkpoint")
    else:
        (transformer_root / "model-00001-of-00001.safetensors").write_bytes(b"checkpoint")


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        model="unused",
        revision=None,
        task_type="t2va",
    )


def test_source_identity_accepts_content_addressed_hf_snapshot(
    monkeypatch,
    tmp_path,
) -> None:
    commit = "a" * 40
    partition_root = tmp_path / "models--MiniMaxAI--MiniMax-H3" / "snapshots" / commit / "FL2VA"
    _write_minimax_source(partition_root, link_hf_blob=True)
    monkeypatch.setattr(
        producer_module,
        "_minimax_h3_partition_for_task",
        lambda *_args: "fl2va",
    )
    monkeypatch.setattr(
        producer_module,
        "_resolve_minimax_h3_model_root",
        lambda *_args: partition_root,
    )

    source = producer_module.resolve_minimax_h3_transformer_source(_config())

    assert source.partition_root == partition_root
    assert source.source_identity["repo_id"] == "MiniMaxAI/MiniMax-H3"
    assert source.source_identity["revision"] == commit
    assert source.source_identity["shards"][0]["blob_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    assert "transformer_config_sha256" in source.source_identity


def test_source_identity_changes_for_same_size_blob_replacement(
    monkeypatch,
    tmp_path,
) -> None:
    commit = "a" * 40
    partition_root = tmp_path / "models--MiniMaxAI--MiniMax-H3" / "snapshots" / commit / "FL2VA"
    _write_minimax_source(partition_root, link_hf_blob=True)
    monkeypatch.setattr(producer_module, "_minimax_h3_partition_for_task", lambda *_args: "fl2va")
    monkeypatch.setattr(producer_module, "_resolve_minimax_h3_model_root", lambda *_args: partition_root)

    before = producer_module.resolve_minimax_h3_transformer_source(_config())
    _link_shard_blob(partition_root, b"checkpoinu")
    after = producer_module.resolve_minimax_h3_transformer_source(_config())

    assert len(b"checkpoint") == len(b"checkpoinu")
    assert before.source_fingerprint != after.source_fingerprint


def test_source_identity_rejects_regular_shard_in_snapshot_shape(
    monkeypatch,
    tmp_path,
) -> None:
    commit = "a" * 40
    partition_root = tmp_path / "models--MiniMaxAI--MiniMax-H3" / "snapshots" / commit / "FL2VA"
    _write_minimax_source(partition_root, link_hf_blob=False)
    monkeypatch.setattr(producer_module, "_minimax_h3_partition_for_task", lambda *_args: "fl2va")
    monkeypatch.setattr(producer_module, "_resolve_minimax_h3_model_root", lambda *_args: partition_root)

    with pytest.raises(
        producer_module.DiffusionArtifactProducerError,
        match="immutable Hugging Face blob link",
    ):
        producer_module.resolve_minimax_h3_transformer_source(_config())


def test_source_identity_rejects_mutable_local_revision(
    monkeypatch,
    tmp_path,
) -> None:
    partition_root = tmp_path / "FL2VA"
    _write_minimax_source(partition_root, link_hf_blob=False)
    monkeypatch.setattr(
        producer_module,
        "_minimax_h3_partition_for_task",
        lambda *_args: "fl2va",
    )
    monkeypatch.setattr(
        producer_module,
        "_resolve_minimax_h3_model_root",
        lambda *_args: partition_root,
    )
    config = _config()
    config.revision = "operator-label-that-is-not-content-addressed"

    with pytest.raises(
        producer_module.DiffusionArtifactProducerError,
        match="content-addressed Hugging Face snapshot",
    ):
        producer_module.resolve_minimax_h3_transformer_source(config)
