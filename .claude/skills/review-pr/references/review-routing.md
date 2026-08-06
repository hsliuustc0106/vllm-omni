# vLLM-Omni Review Routing

Use this reference after the diff census. Select the narrowest matching domain
rows and risk checks; do not load every linked skill.

## Contents

- [Route by changed surface](#route-by-changed-surface)
- [Map architecture ownership](#map-architecture-ownership)
- [Scan recurring blockers](#scan-recurring-blockers)
- [Require change-specific evidence](#require-change-specific-evidence)
- [Calibrate findings](#calibrate-findings)

## Route by changed surface

| Changed surface or claim | Load | Review emphasis |
| --- | --- | --- |
| Any production-code diff | [`precheck-pr` code-quality patterns](../../precheck-pr/references/code-quality.md) | Apply only to added lines: kwargs string plumbing, swallowed exceptions, misleading types, hot-path copies, and event-loop blocking. |
| Tests, missing tests, or CI wiring | [`vllm-omni-test`](../../vllm-omni-test/SKILL.md) | Map the live path to L1-L4 coverage, markers, Buildkite wiring, and a focused command. |
| Bug fix, public/config change, model addition, or performance claim | [`precheck-pr` evidence checklists](../../precheck-pr/references/checklists.md) | Reuse the relevant evidence bar, then verify the claims independently. |
| New or changed diffusion pipeline/model | [`add-diffusion-model`](../../add-diffusion-model/SKILL.md) | Pipeline contracts, latent shapes, parallel/cache integration, offline/online paths, docs, and E2E coverage. |
| Diffusion performance claim | [`diffusion-perf-opt`](../../diffusion-perf-opt/SKILL.md) | Measurement protocol, stage timing, parallel strategy, quality, memory, and comparable A/B evidence. |
| New or changed TTS model | [`add-tts-model`](../../add-tts-model/SKILL.md) | Registration, optional dependencies, AR/stage contracts, serving, outputs, and tests. |
| Quantization, dtype, or weight loading | [`quantization`](../../quantization/SKILL.md) | Method/layer/device compatibility, weight mapping, scale/dtype handling, quality, and hardware evidence. |
| NPU runner or Ascend-specific upgrade | [`vllm-omni-npu-upgrade`](../../vllm-omni-npu-upgrade/SKILL.md) | GPU-to-NPU translation, runner lifecycle, platform guards, parity, and NPU validation. |

If a linked skill is absent in the reviewed checkout, continue with live source
and tests and state the missing repository guidance. Do not import a replacement
from another branch during review.

## Map architecture ownership

| Owner | Primary paths | Contract to preserve |
| --- | --- | --- |
| Entrypoints and I/O | `entrypoints/`, `inputs/`, `outputs/`, `request.py`, `data_entry_keys.py` | Validate public inputs, preserve request identity/modality, serialize responses, and map errors without owning model policy. |
| Configuration | `config/`, `deploy/`, stage configuration | Resolve authoring inputs once, keep transport-safe config separate from process-local runtime objects, and reject invalid topology early. |
| Orchestration | `engine/`, `distributed/omni_coordinator/`, `distributed/ray_utils/` | Own stage routing, ordering, lifecycle, cancellation, startup, shutdown, and failure propagation. |
| Connectors | `distributed/omni_connectors/`, platform connectors | Transfer and synchronize data without choosing stages; preserve sender/receiver schema, completion, timeout, backpressure, and cleanup. |
| AR runtime | `core/`, `worker/` | Preserve upstream scheduler/request-state semantics; workers execute rather than route. |
| Diffusion runtime | `diffusion/sched/`, `executor/`, `worker/`, `diffusion_engine.py`, `ipc.py` | Keep one scheduler-owned request lifecycle and release all request state on terminal paths. |
| Model integration | `model_executor/`, `model_extras/`, `plugins/`, `diffusion/models/`, registries/loaders | Select through registries, keep model policy out of orchestration, and align config, exports, loaders, processors, and tests. |
| Platform layer | `platforms/`, platform attention/quantization | Isolate vendor capability detection and imports; preserve common import paths and supported fallback behavior. |
| Cache and quantization | prefix/KV/latent caches, `quantization/`, diffusion quantization | Define complete identity, validity, eviction/cleanup, disabled/reference paths, and incompatible-input rejection. |
| Metrics and benchmarks | `metrics/`, profilers, benchmarks | Preserve metric meaning and request correlation; keep profiling opt-in and benchmarks reproducible and correctness-aware. |

When ownership crosses rows, trace the producer and final consumer on both sides.
Treat silent ownership movement as a design change that needs explicit evidence.

## Scan recurring blockers

Apply only when the diff introduces or exposes the pattern. Existing backlog is
not a finding by itself.

- **Exceptions:** reject bare or broad catches that hide init, config, weight
  loading, request, or execution failures. Catch expected types and preserve
  actionable context.
- **Public contracts:** check removed parameters/APIs, newly required arguments,
  changed defaults, aliases, and migration behavior across every entrypoint.
- **Validation:** validate at the owning boundary before expensive work; reject
  unknown, duplicate, malformed, or incompatible inputs visibly.
- **Async/concurrency:** reject blocking I/O or sleep on the event loop, locks
  held across `await`, accidental serialization, and unbounded background work.
- **Lifecycle:** cover allocation/start, success, partial failure, timeout,
  cancellation, shutdown, and cleanup. Ensure terminal states are monotonic.
- **Connectors:** verify sender/receiver identity, shape, dtype, device, ordering,
  completion, timeout, error propagation, and resource release.
- **Distributed execution:** cover affected rank/world-size/topology modes and a
  supported single-device path; isolate vendor and collective assumptions.
- **Cache/state:** include every correctness-affecting input in the key; verify
  request isolation, feature-off behavior, invalidation, eviction, and cleanup.
- **Model/config registration:** align registry entries, architecture names,
  exports, YAML/config, processors, loaders, dependencies, and examples.
- **Tensor/media data:** trace shape, dtype, device, layout, batch/CFG expansion,
  empty/optional values, and serialization through the final consumer.
- **Tests:** reject helper-only or over-mocked tests that bypass the production
  dispatcher, use an unrealistic MRO, or cannot fail when behavior regresses.
- **Security:** reject secrets, unsafe deserialization/shell/eval, user payloads
  in logs, unbounded metric labels, and unvalidated user-controlled paths.

## Require change-specific evidence

### Bug fixes

- Reproduce the failure or identify a regression test that fails on the frozen
  base and passes on the frozen head.
- Trace the repair to the root owner and check analogous paths only when the
  same contract reaches them.

### Public API, CLI, schema, or config

- Check normal, invalid, omitted/default, duplicate, alias, and legacy inputs.
- Verify streaming/non-streaming and offline/online paths when both expose the
  contract.
- Require user-facing docs and migration guidance for visible behavior changes.

### Model and pipeline additions

- Verify registry, exports, config, processor/loader, dependencies, and a
  representative production-dispatch inference path.
- Require valid output at the correct modality/shape/rate and comparison with a
  known-good implementation when available.
- For diffusion, check both supported serving modes, latent/parallel/cache
  contracts, model/feature docs, and production-path E2E coverage.
- Request performance, memory, acceleration, or quality evidence only for claims
  and supported requirements that actually apply; do not impose stale matrices.

### Distributed, connector, scheduler, cache, or lifecycle changes

- Build a compact path/topology matrix covering feature off/on, supported rank
  layouts, concurrency, failure, timeout/cancellation, and cleanup.
- Verify payload symmetry and the final owner of request/cache/resource state.

### Performance, memory, or accuracy claims

- Compare base and head on the same hardware, software, model, inputs, seed,
  precision, parallelism, warmup, and measured repetitions.
- Require both the claimed metric and correctness/quality evidence. Include exact
  commands, variability, stage metrics when relevant, and peak memory.
- Explain regressions rather than applying universal thresholds that the current
  repository contract does not define.

### Test-only or documentation-only changes

- For tests, verify assertions pin the intended contract, markers select the
  right CI lane, and the test reaches the relevant production path.
- For docs, check links, navigation/build rules, commands, identifiers, version
  claims, and the bounded live contract being described.

## Calibrate findings

- **P0:** security exposure, data corruption, or broad project unusability.
- **P1:** likely runtime failure, wrong output, compatibility break, unsafe
  lifecycle, or missing evidence for a changed high-risk contract.
- **P2:** a real non-blocking defect or maintainability issue with a concrete
  future failure mode.

Anchor each finding to a current diff line when possible. Prove the trigger and
reachable path, consolidate downstream symptoms under one root cause, and turn
unavailable hardware or incomplete CI into a validation gap rather than an
automatic defect.
