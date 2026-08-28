# Layerwise Offloading

Layerwise, or blockwise, offloading keeps one transformer block on the
accelerator and prefetches the next block while the current block computes.
It is best suited to compute-heavy video DiTs whose block execution can hide
host-to-device transfers.

## Execution flow

Each block has a pre-forward and post-forward hook. Parameters are consolidated
in pinned host tensors and rematerialized for execution on a dedicated copy
stream.

| Block | Pre-forward hook | Forward | Post-forward hook |
| --- | --- | --- | --- |
| block 0 | Prefetch block 1 | Compute block 0 | Free block 0 |
| block 1 | Prefetch block 2 | Compute block 1 | Free block 1 |
| ... | ... | ... | ... |
| last block | Prefetch block 0 | Compute last block | Free last block |

Selected, plan-declared text-encoder blocks can use the same rank-local
streaming mechanism. Image encoders, VAEs, and non-block DiT modules remain
device resident.

## Usage

```python
from vllm_omni import Omni

omni = Omni(
    model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    enable_layerwise_offload=True,
)
```

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --omni --enable-layerwise-offload
```

## Component selection

Use `--layerwise-offload-components` with `dit`, `text_encoder`, or both:

- Omitting the option selects `dit`, preserving the existing DiT-only behavior.
- `all` selects both supported categories.
- An explicit subset such as `dit,text_encoder` controls only those categories.

```bash
# Backward-compatible DiT-only behavior
vllm serve /path/to/model --omni \
  --enable-layerwise-offload

# Stream a model-declared text encoder while keeping the DiT resident
vllm serve /path/to/model --omni \
  --enable-layerwise-offload \
  --layerwise-offload-components text_encoder
```

Encoder categories are resolved from `OffloadPlan.encoder_component_types`
first. A name-based fallback is retained for pipelines that predate
`OffloadPlan`.

For encoder-only gather tables that are too large to justify device residency,
models may declare `encoder_host_resident_table_attrs`. The table lookup then
runs on CPU and only its small output crosses to the target device. This is an
explicit safety contract: do not declare tied embeddings or weights accessed
directly outside the table module's forward.

## Model integration

Transformer classes declare containers of executable blocks:

```python
class WanTransformer3DModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["blocks"]


class Flux2Transformer2DModel(nn.Module):
    _layerwise_offload_blocks_attrs = [
        "transformer_blocks",
        "single_transformer_blocks",
    ]
```

Auxiliary components use declarative pipeline metadata:

```python
from vllm_omni.diffusion.offloader import OffloadPlan


class MyPipeline(nn.Module):
    _encoder_modules = ["prompt_model"]
    _offload_plan = OffloadPlan(
        encoder_component_types={"prompt_model": "text_encoder"},
        encoder_block_attrs={"prompt_model": ("encoder.layers",)},
        encoder_host_resident_table_attrs={"prompt_model": ("shared",)},
    )
```

See the [layerwise design](../../../design/feature/offloader/layerwise_offload.md)
for the discovery and hook invariants. Both ordinary and distributed layerwise
offload consume the same `OffloadPlan` metadata.

## Limitations

- The ordinary backend uses rank-local transfer. For multi-device shard +
  AllGather transport, use distributed layerwise offload.
- Setup consolidates and pins block parameters, increasing cold-start time.
- Performance depends on block compute time and host-to-device bandwidth;
  lightweight blocks may not hide transfers.
