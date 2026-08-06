# vLLM-Omni Review Routing

Use this reference after the diff census. Select only the rows supported by the
changed files and claims; one primary domain plus one cross-cutting domain is
usually enough.

| Signal | Load | Optional repo-local skill |
| --- | --- | --- |
| Config, orchestration, connector, scheduler, cache, or cross-stage change | [architecture.md](architecture.md), then [blocker-patterns.md](blocker-patterns.md) if a risk signal appears | [`precheck-pr` code-quality patterns](../../precheck-pr/references/code-quality.md) |
| New or changed model, registry, processor, loader, or stage config | [model-addition-checklist.md](model-addition-checklist.md) | [`add-tts-model`](../../add-tts-model/SKILL.md) for TTS |
| Diffusion pipeline, model, scheduler, latent, cache, or parallel feature | [diffusion-checklist.md](diffusion-checklist.md) | [`add-diffusion-model`](../../add-diffusion-model/SKILL.md) |
| Latency, throughput, memory, acceleration, or quality claim | [perf-verification.md](perf-verification.md) | [`diffusion-perf-opt`](../../diffusion-perf-opt/SKILL.md) for diffusion |
| Tests changed, missing, or test-only | [test-quality-evaluation.md](test-quality-evaluation.md) and, when docs/CI evidence matters, [tests-docs-checklist.md](tests-docs-checklist.md) | [`vllm-omni-test`](../../vllm-omni-test/SKILL.md) |
| Runnable affected path and suitable hardware/server available | [verification.md](verification.md) | None |
| Quantization, dtype, scales, or weight mapping | [blocker-patterns.md](blocker-patterns.md) | [`quantization`](../../quantization/SKILL.md) |
| Ascend/NPU runner or platform change | [architecture.md](architecture.md) and [verification.md](verification.md) | [`vllm-omni-npu-upgrade`](../../vllm-omni-npu-upgrade/SKILL.md) |
| Findings ready for delivery | [maintainer-style-study.md](maintainer-style-study.md) | None |

For a bug fix or public/config change, also use the relevant
[`precheck-pr` evidence checklist](../../precheck-pr/references/checklists.md).
If a linked skill is absent, continue with live code and tests; do not import a
replacement from another branch.

## Bound context expansion

- Read linked issues only when they define acceptance criteria or reproduction.
- Search sibling implementations only for the same producer-consumer contract.
- Open CI logs only for the first overlapping failure.
- Stop loading references once every changed semantic path has an evidence plan.

## Calibrate findings

- **P0:** security exposure, data corruption, or broad project unusability.
- **P1:** reachable runtime failure, wrong output, compatibility break, or unsafe
  lifecycle in the changed behavior.
- **P2:** real non-blocking defect with a concrete future failure mode.

Treat missing hardware, pending CI, or unsupported claims as validation gaps
unless the repository contract makes that evidence a merge requirement.
