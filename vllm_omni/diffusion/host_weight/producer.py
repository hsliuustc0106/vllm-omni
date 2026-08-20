# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Elected MiniMax-H3 transformer artifact producer.

Descriptor construction is deliberately separate from ``open_build``.  The
former reads immutable source identity only; the latter is the sole path that
constructs a loaded transformer, iterates checkpoint sources, or quantizes.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from vllm.config.load import LoadConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.diffusers_loader import (
    DiffusersPipelineLoader,
)
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
    MiniMaxH3DiTModel,
)
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    _minimax_h3_partition_for_task,
    _resolve_minimax_h3_model_root,
)
from vllm_omni.host_weight_runtime import (
    ArtifactBuildSession,
    ArtifactManifest,
    ArtifactSink,
    ArtifactSpec,
    BuildSessionCleanupRegistry,
    ProducerDescriptor,
    TensorRole,
    canonical_digest,
)

from .formats.base import ArtifactFormatExporter


class DiffusionArtifactProducerError(RuntimeError):
    pass


def _prepare_streaming_online_quant(
    loader: DiffusersPipelineLoader,
    model: nn.Module,
) -> int:
    """Validate the producer's streaming contract and enable optional hooks.

    vLLM online-FP8 layers advertise ``uses_meta_device`` and are finalized by
    the layerwise weight loader.  ``_stream_online_quant_weights_to_cpu`` then
    moves each completed layer to host memory.  Some other quantizers also
    expose an ``offload_after_quant`` hook, but that hook is an optimization,
    not the capability marker for online FP8.
    """
    if not loader._has_online_quant(model):
        raise DiffusionArtifactProducerError("MiniMax-H3 FP8 producer found no online-quantized layers")
    return loader._request_offload_after_quant(model)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _hex_digest(value: str, *, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value.lower())


def _hf_snapshot_identity(path: Path) -> tuple[str, str, Path] | None:
    resolved = path.resolve()
    parts = resolved.parts
    try:
        snapshot_index = parts.index("snapshots")
    except ValueError:
        return None
    if snapshot_index == 0 or snapshot_index + 1 >= len(parts):
        return None
    repo_dir = parts[snapshot_index - 1]
    commit = parts[snapshot_index + 1]
    if not repo_dir.startswith("models--") or not _hex_digest(commit, length=40):
        return None
    repo_id = repo_dir.removeprefix("models--").replace("--", "/")
    repo_root = Path(*parts[:snapshot_index])
    return repo_id, commit, repo_root


def _hf_lfs_blob_identity(shard: Path, *, blob_root: Path) -> str:
    """Return the immutable LFS object ID behind one HF snapshot shard.

    Hashing every shard would add a 40+ GiB startup read to the first use case.
    Instead v1 accepts only the standard Hugging Face cache representation:
    snapshot entries must be symlinks to a flat ``blobs/<sha256>`` object.
    The cache's content-addressed blob ID, not a mutable path or file size,
    becomes part of the artifact key.
    """

    if not shard.is_symlink():
        raise DiffusionArtifactProducerError(
            f"MiniMax-H3 transformer shard is not an immutable Hugging Face blob link: {shard.name}"
        )
    try:
        resolved_blob_root = blob_root.resolve(strict=True)
        blob = shard.resolve(strict=True)
    except OSError as exc:
        raise DiffusionArtifactProducerError(f"MiniMax-H3 transformer shard link is invalid: {shard.name}") from exc
    if blob.parent != resolved_blob_root or not _hex_digest(blob.name, length=64):
        raise DiffusionArtifactProducerError(
            f"MiniMax-H3 transformer shard does not reference a content-addressed HF LFS blob: {shard.name}"
        )
    if not blob.is_file():
        raise DiffusionArtifactProducerError(f"MiniMax-H3 transformer shard blob is missing: {shard.name}")
    return blob.name.lower()


@dataclass(frozen=True)
class MiniMaxH3TransformerSource:
    model_root: Path
    partition_root: Path
    source_fingerprint: str
    source_identity: dict[str, Any]

    def component_source(
        self,
        *,
        revision: str | None,
    ) -> DiffusersPipelineLoader.ComponentSource:
        return DiffusersPipelineLoader.ComponentSource(
            model_or_path=str(self.partition_root),
            subfolder="transformer",
            revision=revision,
            prefix="transformer.",
            fall_back_to_pt=False,
        )


def resolve_minimax_h3_transformer_source(
    od_config: OmniDiffusionConfig,
) -> MiniMaxH3TransformerSource:
    partition = _minimax_h3_partition_for_task(
        getattr(od_config, "task_type", None),
        str(od_config.model),
    )
    if partition != "fl2va":
        raise DiffusionArtifactProducerError(
            "Host Weight Runtime v1 supports only the MiniMax-H3 FL2VA "
            "transformer; set task_type to t2va/fl2va or pass the FL2VA path"
        )
    model_root = _resolve_minimax_h3_model_root(
        str(od_config.model),
        od_config.revision,
        partition,
    ).resolve()
    partition_root = model_root if model_root.name == "FL2VA" else model_root / "FL2VA"
    model_index = partition_root / "model_index.json"
    transformer_index = partition_root / "transformer" / "model.safetensors.index.json"
    if not model_index.is_file() or not transformer_index.is_file():
        raise DiffusionArtifactProducerError(f"incomplete MiniMax-H3 FL2VA source at {partition_root}")

    snapshot = _hf_snapshot_identity(partition_root)
    if snapshot is None:
        raise DiffusionArtifactProducerError(
            "Host Weight Runtime v1 requires a content-addressed Hugging Face "
            "snapshot path; an operator-supplied local revision cannot prove "
            "that same-sized checkpoint shards are immutable"
        )
    repo_id, commit, repo_root = snapshot

    index_payload = json.loads(transformer_index.read_text(encoding="utf-8"))
    shard_names = sorted(set(index_payload.get("weight_map", {}).values()))
    shard_identities: list[dict[str, int | str]] = []
    for name in shard_names:
        shard = transformer_index.parent / name
        if not shard.is_file():
            raise DiffusionArtifactProducerError(f"MiniMax-H3 transformer shard is missing: {name}")
        shard_identities.append(
            {
                "blob_sha256": _hf_lfs_blob_identity(shard, blob_root=repo_root / "blobs"),
                "name": name,
                "size": shard.stat().st_size,
            }
        )
    source_identity: dict[str, Any] = {
        "schema": 2,
        "repo_id": repo_id,
        "revision": commit,
        "partition": "FL2VA",
        "model_index_sha256": _sha256(model_index),
        "transformer_index_sha256": _sha256(transformer_index),
        "shards": shard_identities,
    }
    transformer_config = transformer_index.parent / "config.json"
    if transformer_config.is_file():
        source_identity["transformer_config_sha256"] = _sha256(transformer_config)
    return MiniMaxH3TransformerSource(
        model_root=model_root,
        partition_root=partition_root,
        source_fingerprint=canonical_digest(source_identity),
        source_identity=source_identity,
    )


class _TransformerBuildModel(nn.Module):
    """Minimal wrapper preserving MiniMax's ordinary pipeline load contract."""

    def __init__(
        self,
        transformer: MiniMaxH3DiTModel,
        source: DiffusersPipelineLoader.ComponentSource,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.weights_sources = (source,)

    def load_weights(
        self,
        weights: Any,
    ) -> set[str]:
        def stripped() -> Any:
            for name, tensor in weights:
                if not name.startswith("transformer."):
                    raise DiffusionArtifactProducerError(f"unexpected transformer build weight {name!r}")
                yield name.removeprefix("transformer."), tensor

        loaded = self.transformer.load_weights(stripped())
        self.transformer.post_load_weights()
        return {f"transformer.{name}" for name in loaded}


class _MiniMaxH3ArtifactBuildSession:
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        source: MiniMaxH3TransformerSource,
        device: torch.device,
        quant_config: Fp8Config,
        format_adapter: ArtifactFormatExporter,
        spec: ArtifactSpec,
        load_config: LoadConfig,
    ) -> None:
        self._od_config = od_config
        self._source = source
        self._device = device
        self._quant_config = quant_config
        self._format_adapter = format_adapter
        self._spec = spec
        self._load_config = load_config
        self._owned: list[Any] = []
        self._owned_released = False
        self._gc_complete = False
        self._accelerator_cache_released = False
        self._closed = False
        self._built = False

    def build(self, sink: ArtifactSink) -> ArtifactManifest:
        if self._closed:
            raise DiffusionArtifactProducerError("artifact build session is closed")
        if self._built:
            raise DiffusionArtifactProducerError("artifact build session is single-use")
        self._built = True
        quant_config = self._quant_config
        if quant_config.is_checkpoint_fp8_serialized:
            raise DiffusionArtifactProducerError("HWR v1 requires transformer online per-tensor FP8")

        with set_default_torch_dtype(self._od_config.dtype):
            with self._device, set_current_diffusion_config(self._od_config):
                transformer = MiniMaxH3DiTModel(
                    self._od_config,
                    quant_config=quant_config,
                )
        source = self._source.component_source(revision=self._od_config.revision)
        wrapper = _TransformerBuildModel(transformer, source)
        loader = DiffusersPipelineLoader(self._load_config, od_config=self._od_config)
        self._owned[:] = [transformer, wrapper, loader]

        _prepare_streaming_online_quant(loader, wrapper)
        loader.load_weights(wrapper, stream_online_quant_to_cpu=True)
        loader._process_weights_after_loading(wrapper, self._device)
        wrapper.to("cpu")

        finalized = self._format_adapter.finalize_for_artifact(transformer)
        self._owned.append(finalized)
        tensor_specs = []
        for item in finalized.tensors:
            # The repository sink computes the content digest while copying.
            # Returning the TensorSpec avoids a second pass over a 40+ GiB FP8
            # transformer.
            tensor_specs.append(
                sink.write_tensor(
                    item.tensor_id,
                    item.tensor,
                    role=TensorRole(item.role),
                )
            )
        create = getattr(ArtifactManifest, "create", None)
        if not callable(create):
            raise DiffusionArtifactProducerError("Host Weight Runtime ArtifactManifest.create is unavailable")
        return create(
            self._spec,
            format_metadata=finalized.binding_recipe.to_dict(),
            tensors=tuple(tensor_specs),
        )

    def close(self) -> None:
        if self._closed:
            return
        if not self._owned_released:
            self._owned.clear()
            self._owned_released = True
        if not self._gc_complete:
            gc.collect()
            self._gc_complete = True
        if not self._accelerator_cache_released:
            if self._device.type == "cuda" and torch.cuda.is_available():
                torch.accelerator.empty_cache()
            self._accelerator_cache_released = True
        self._closed = True


class DiffusionArtifactProducer:
    """Lazy producer capability passed to the independent runtime."""

    def __init__(
        self,
        *,
        spec: ArtifactSpec,
        od_config: OmniDiffusionConfig,
        source: MiniMaxH3TransformerSource,
        device: torch.device,
        quant_config: Fp8Config,
        format_adapter: ArtifactFormatExporter,
        load_config: LoadConfig,
    ) -> None:
        self._spec = spec
        self._od_config = od_config
        self._source = source
        self._device = device
        self._quant_config = quant_config
        self._format_adapter = format_adapter
        self._load_config = load_config

    @property
    def descriptor(self) -> ProducerDescriptor:
        return self._spec.producer

    def open_build(
        self,
        cleanup_registry: BuildSessionCleanupRegistry,
    ) -> ArtifactBuildSession:
        if os.environ.get("VLLM_OMNI_HWR_POISON_PRODUCER") == "1":
            raise DiffusionArtifactProducerError("artifact producer is poisoned for HWR warm-hit verification")
        session = _MiniMaxH3ArtifactBuildSession(
            od_config=self._od_config,
            source=self._source,
            device=self._device,
            quant_config=self._quant_config,
            format_adapter=self._format_adapter,
            spec=self._spec,
            load_config=self._load_config,
        )
        cleanup_registry.register_before_return(session)
        return session


__all__ = [
    "DiffusionArtifactProducer",
    "DiffusionArtifactProducerError",
    "MiniMaxH3TransformerSource",
    "resolve_minimax_h3_transformer_source",
]
