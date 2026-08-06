# Model Addition Checklist

Read this reference when a PR adds or expands a model architecture, pipeline,
loader, processor, registry entry, or stage configuration.

Official docs: [model contribution guides](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/model/),
[adding an omni model](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/model/adding_omni_model/),
[adding a diffusion model](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/model/adding_diffusion_model/),
[adding a TTS model](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/model/adding_tts_model/),
and [supported models](https://docs.vllm.ai/projects/vllm-omni/en/latest/models/supported_models/).

## Match claims to the diff

- Reconcile every claimed architecture, modality, variant, feature, and example
  with changed files and registry entries.
- Resolve config model types, architecture names, callable paths, exports, and
  dependency extras against live symbols.
- Check sibling generations sharing an architecture name are disambiguated by
  a stable config predicate rather than import order or an accidental default.

## Close the integration chain

Trace:

```text
public model id -> config/pipeline selection -> registry -> loader/processor
  -> stage inputs -> model execution -> stage/public output
```

Verify optional dependencies fail with an actionable message, weight names and
dtypes map correctly, and each advertised serving mode reaches the production
dispatcher. Confirm outputs are non-empty and valid for their modality, shape,
sample rate, or response schema.

## Remove accidental surface

Search bounded call sites for:

- inference-dead training `forward()` paths, unused factories/wrappers, never-set
  branch keys, and immediately discarded parameters;
- duplicate payload strings, validation, or shape coercion across stages;
- compatibility aliases in brand-new code without an existing caller;
- private symbols re-exported as public API or module-level side effects.

Keep code only when it has a distinct live caller, invariant, or compatibility
contract. Prefer one typed producer-consumer schema to repeated string keys.

## Require proportionate evidence

- Run a representative production-path inference and assert output content, not
  only process survival.
- Compare with a known-good upstream implementation when available; fix seeds
  and state numeric tolerances or qualitative limitations.
- Require profiling or A/B tables when the PR makes performance, memory,
  precision, or quality claims, or when a suspected hot-path/device bug needs
  that evidence. Do not impose fixed utilization or regression thresholds that
  the repository does not define.
- Require focused registry/config/loading tests, a representative E2E path, and
  user-facing model/usage documentation.

Use [diffusion-checklist.md](diffusion-checklist.md) for diffusion models and
[perf-verification.md](perf-verification.md) for quantitative claims.
