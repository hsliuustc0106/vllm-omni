# vLLM-Omni Review Routing

Use after the diff census. Apply the general checks first, choose one primary
behavior owner, add a second only when the live call chain crosses its boundary,
then select any cross-cutting overlays. Title prefixes are hints, not routes.

Treat the traced live consumer and producer-consumer boundary as authoritative,
then apply this repo-local map. Use titles and changed paths to validate the
route, not to override the live behavior.

Official docs: [design documents](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/),
[model contribution guides](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/model/),
and [feature compatibility](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/feature_compatibility/).

## Primary owner

| Owner | Claim and live-consumer signals | Read |
| --- | --- | --- |
| Configuration | Schema, defaults, deploy/pipeline construction, registry, CLI projection, topology | [configuration.md](configuration.md) |
| Serving | Public request/response, CLI/API, streaming, output assembly, orchestration, engine lifecycle | [serving.md](serving.md) |
| Model Executor | Loader, stage input, worker/runner, device startup, AR model execution or data bridge | [model-executor.md](model-executor.md) |
| Diffusion | Shared pipeline/denoise loop, diffusion scheduler, latent, VAE/DiT, cache, offload or parallelism | [diffusion-checklist.md](diffusion-checklist.md) |
| Distributed | Connector, KV transfer, collective, load balancing, route/port or cross-stage transport | [distributed.md](distributed.md) |
| Scheduler | Request queue/state, token budget, KV/input readiness, scheduling coordination or prefix cache | [scheduler.md](scheduler.md) |

Use changed files to validate the selected owner and find both sides of a
boundary. For tests-, docs-, or CI-only PRs, route to the production behavior
they protect; use only general checks and the delivery overlay when no such
behavior exists.

## Cross-cutting overlays

| Signal | Read | Optional repo-local skill |
| --- | --- | --- |
| New or expanded model, loader, processor, registry, or stage config | [model-addition-checklist.md](model-addition-checklist.md) | [`add-tts-model`](../../add-tts-model/SKILL.md) or [`add-diffusion-model`](../../add-diffusion-model/SKILL.md) |
| Accelerator, kernel, attention backend, quantization, dtype, scales, or vendor path | [platform-checks.md](platform-checks.md) and, if runnable, [verification.md](verification.md) | [`quantization`](../../quantization/SKILL.md) or [`$vllm-omni-npu-model-runner-upgrade`](../../vllm-omni-npu-upgrade/SKILL.md) |
| Latency, throughput, memory, scaling, precision, or quality claim | [perf-verification.md](perf-verification.md) | [`diffusion-perf-opt`](../../diffusion-perf-opt/SKILL.md) for diffusion |
| Tests changed, absent for risky behavior, or test-only | [test-quality-evaluation.md](test-quality-evaluation.md) | [`vllm-omni-test`](../../vllm-omni-test/SKILL.md) |
| CI, examples, docs, public behavior, or contributor evidence | [tests-docs-checklist.md](tests-docs-checklist.md) | None |
| Suitable hardware/server and runnable affected path | [verification.md](verification.md) | None |
| User asks who should review or asks to request/ping reviewers | [review-requests.md](review-requests.md) | None |

For a bug fix, require a reachable reproduction, before/after behavior, and a
regression test that would fail when the defect returns. For a refactor, prove
behavior parity and remove obsolete paths. For a feature, verify the public
contract, compatibility/default behavior, production dispatch, and docs.

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
