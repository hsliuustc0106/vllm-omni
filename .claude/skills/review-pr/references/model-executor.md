# Model Executor Review

Use for model loading, stage inputs, runners, workers, shared model layers,
device mapping, platform worker overrides, and AR-to-downstream data bridges.

Official docs: [model contribution guides](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/model/)
and [adding an omni model](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/model/adding_omni_model/).

## Boundaries

Start with `vllm_omni/model_executor/`, `vllm_omni/worker/`, and
`vllm_omni/inputs/`. Treat deploy/config authoring as Configuration, shared
diffusion execution as Diffusion, and request orchestration as Serving.

## Rules

- Preserve loader selection, weight-name mapping, dtype/device placement,
  optional-dependency errors, and architecture/model-owned hooks.
- Trace stage input identity, shape, layout, dtype, device, batch rows, and
  metadata through preprocessing, runner state, model call, and stage output.
- Keep shared runner behavior valid for model-specific hooks, chunked-prefill,
  pipeline parallelism, and platform overrides; compare sibling runners when a
  shared call site changes.
- Initialize mixins and per-request state on every production MRO; clear them on
  completion, failure, cancellation, and shutdown.
- Keep worker/device code focused on execution rather than routing or public
  policy, and guard vendor-only imports and capabilities.

Require focused loader/runner/bridge tests plus a representative model consumer.
Use the model-addition and platform references when those overlays apply.
