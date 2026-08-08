# Model Addition Checklist

Read this reference when a PR adds or expands a model architecture, pipeline,
loader, processor, registry entry, or stage configuration.

Official docs: [model contribution guides](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/model/),
[adding an omni model](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/model/adding_omni_model/),
[adding a diffusion model](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/model/adding_diffusion_model/),
[adding a TTS model](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/model/adding_tts_model/),
[supported models](https://docs.vllm.ai/projects/vllm-omni/en/latest/models/supported_models/),
[diffusion features](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion_features/),
and [feature compatibility](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/feature_compatibility/).

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

## Keep support documentation truthful

- Build an evidence-backed inventory of the new model's exact architecture and
  checkpoint identifiers, modalities and tasks, hardware backends, serving
  modes, and supported acceleration, parallelism, cache, offload, and
  quantization features. Treat untested support as unknown, not supported.
- Update `docs/models/supported_models.md` with the architecture, model family,
  example checkpoints, and only the backends validated by the PR.
- For a diffusion model, add it to the applicable ImageGen, VideoGen, or
  AudioGen table in `docs/user_guide/diffusion_features.md`. Fill every feature
  column according to its legend and document partial support or constraints.
- Update `docs/user_guide/feature_compatibility.md` when the model changes valid
  feature combinations, configuration constraints, examples, or limitations,
  and update any other branch-local support table affected by the claims.
- Cross-check every documented support mark against the registry, configuration,
  production-path tests, and PR evidence; do not copy a sibling model's support
  claims without validation.

## Ship the model recipe

- Require every new model PR to add or update one model-family recipe under
  `recipes/<vendor>/` and its row in the `recipes/README.md` Available Recipes
  table. Follow [`recipes/TEMPLATE.md`](../../../../recipes/TEMPLATE.md) and the
  layout rules in [`recipes/README.md`](../../../../recipes/README.md); the
  external recipes repository is a structural reference, not a substitute.
- Include only tested task, serving-mode, and hardware configurations. Keep the
  exact model identifiers, flags, commands, verification, feature support,
  constraints, and limitations consistent with code, tests, examples, and the
  support documentation.
- Cover each validated platform in its own hardware section. Use one recipe per
  model family by default; justify a split when configurations cannot be kept
  clear in one file, and do not add placeholder sections for untested platforms.

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
