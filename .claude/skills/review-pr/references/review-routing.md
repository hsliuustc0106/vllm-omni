# vLLM-Omni Review Routing

Load this reference only after collecting the changed-file census. Select the
narrowest matching rows; do not load every skill for every review.

## Route by changed surface

| Changed surface or claim | Load | Review emphasis |
| --- | --- | --- |
| Any production-code diff | [`precheck-pr` code-quality patterns](../../precheck-pr/references/code-quality.md) | Apply only to added lines: kwargs string plumbing, swallowed exceptions, misleading types, hot-path copies, and event-loop blocking. |
| Tests, missing tests, or CI marker/wiring changes | [`vllm-omni-test`](../../vllm-omni-test/SKILL.md) | Map the runtime path to L1-L4 coverage, markers, Buildkite wiring, and a runnable focused command. |
| Bug fix, public/config change, model addition, or performance claim | [`precheck-pr` evidence checklists](../../precheck-pr/references/checklists.md) | Reuse the relevant evidence requirements, but independently verify the author-facing claims. |
| New or changed diffusion pipeline/model | [`add-diffusion-model`](../../add-diffusion-model/SKILL.md) | Transformer/pipeline contract, latent shapes, parallelism, cache acceleration, offline and online paths, docs, and E2E coverage. |
| Diffusion performance or benchmark claim | [`diffusion-perf-opt`](../../diffusion-perf-opt/SKILL.md) | Measurement protocol, stage timings, parallel strategy, quality gate, peak memory, and comparable A/B evidence. |
| New or changed TTS model | [`add-tts-model`](../../add-tts-model/SKILL.md) | Model/pipeline registration, optional dependencies, AR/stage contracts, serving path, samples, and tests. |
| Quantization, checkpoint dtype, or weight-loading change | [`quantization`](../../quantization/SKILL.md) | Method selection, modality compatibility, weight mapping, scale/dtype/device handling, output quality, and hardware evidence. |
| NPU runner or Ascend-specific upgrade | [`vllm-omni-npu-upgrade`](../../vllm-omni-npu-upgrade/SKILL.md) | GPU-to-NPU semantic translation, runner lifecycle, platform guards, parity, and NPU validation. |

If a linked skill is absent in an older checkout, continue with live source and
tests and state the missing repository guidance. Do not fetch a replacement
skill from an unrelated branch during review.

## Cross-cutting contract map

Use the applicable rows to structure source inspection. A row with no changed
contract is not automatically a required test dimension.

| Risk | Trace or matrix |
| --- | --- |
| Public API or config | Each accepted source/alias/default -> validation -> normalized owner value -> every consumer; include invalid, duplicate, and compatibility inputs. |
| Offline and online parity | Same user field/default -> each dispatcher -> equivalent model/stage consumer; include streaming when exposed. |
| Stage topology | Pipeline config -> stage construction -> rank/group mapping -> connector payload -> receiving stage; include single- and multi-stage layouts that remain supported. |
| Tensor/data contract | Producer shape/dtype/device/layout -> serialization or collective -> consumer expectation; include batch/CFG expansion and empty/optional values. |
| Async/resource lifecycle | Allocation/start -> ownership transfer -> normal completion -> exception -> timeout/cancellation -> shutdown; identify blocking points and concurrency bounds. |
| Cache/state | Key construction -> population -> reuse/invalidation -> cross-request isolation -> cleanup; verify feature-off behavior. |
| Registry/model loading | Config/model identifier -> registry -> import/export -> class/config construction -> dependency/weight load -> first inference. |
| Performance | Exact workload/environment -> baseline -> PR -> quality equivalence -> stage metrics -> latency/throughput/memory claim. |

## Minimum evidence by change type

### Bug fixes

- Reproduce the original failure or identify a checked-in regression test that
  fails on the frozen base and passes on the frozen head.
- Trace the fix to the root owner instead of accepting a downstream symptom
  patch.
- Check analogous paths only when the same contract reaches them.

### Public API, schema, CLI, or config changes

- Check a normal input, invalid input, default/omitted input, and any supported
  alias or legacy form.
- Verify both streaming and non-streaming endpoints when both expose the field.
- Require docs/examples for user-visible additions or behavior changes.
- Treat silent acceptance, overwrite, or fallback as a defect when the contract
  promises strict validation.

### Model and pipeline changes

- Verify registration, exports, pipeline configuration, processor/loader path,
  and terminal output.
- Require at least one test that reaches the production dispatcher, not only a
  helper mock.
- Check representative offline and online paths when both are supported.
- Treat claimed hardware execution as unverified without actual device evidence.

### Distributed, connector, and scheduler changes

- Enumerate affected topology/rank/stage modes instead of testing one happy
  layout.
- Verify payload schema, sender/receiver symmetry, shutdown/error propagation,
  and no event-loop blocking.
- Check feature-disabled and constrained/offload behavior when supported.

### Performance and accuracy changes

- Compare base and head with the same hardware, software, model, inputs, seed,
  precision, parallelism, warmup, and measured repetitions.
- Require both performance and output-quality evidence; a speedup alone does not
  establish correctness.
- Prefer checked-in commands or scripts and report variability, not a best run.
- Do not invent universal regression thresholds when the repository or PR
  contract does not define them; explain the observed tradeoff.

### Documentation-only changes

- Check links, navigation/build rules, commands, identifiers, and version claims.
- Inspect only the bounded live code or config needed to verify described
  behavior.
- Skip pytest unless the docs change executable examples with an existing
  lightweight test path.

## Finding discipline

- Anchor each finding to an added or modified line whenever possible.
- Prove the path reaches a real consumer; helper-only concerns are notes until
  reachability is established.
- Use missing tests as a finding only when they leave a changed high-risk
  contract unprotected.
- Consolidate repeated symptoms under one owner/root-cause comment.
- Record unavailable hardware, incomplete CI, or missing reproduction as a
  validation gap, not automatically as a defect.
