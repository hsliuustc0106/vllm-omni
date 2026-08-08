# Scheduler Review

Use for AR/generation request queues, token budgets, waiting/running state,
cross-stage KV readiness, scheduling coordination, and tensor prefix cache.

Official docs: [AR module design](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/module/ar_module/)
and [architecture overview](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/architecture_overview/).

## Boundaries

Start with `vllm_omni/core/sched/` and `vllm_omni/core/prefix_cache.py`.
Distributed owns transfer execution, Serving owns engine orchestration, and
Diffusion owns denoise/noise schedulers.

## Rules

- Preserve legal request-state transitions and one terminal cleanup path across
  normal completion, preemption, timeout, cancellation, and upstream failure.
- Keep token, sequence, batch, and memory budgets synchronized with admission,
  scheduling, and worker-visible state; prevent starvation and double counting.
- Treat KV/input readiness as a distributed state machine: verify chunk/full
  payload boundaries, rank consensus, retries, and stale completion signals.
- Include every correctness-affecting model, request, stage, layout, and payload
  input in prefix-cache identity; define miss, disabled, eviction, and cleanup.
- Check compatibility with the pinned upstream vLLM scheduler APIs and sibling
  sync/async implementations when shared behavior changes.

Require focused state-transition and regression tests that fail when the changed
condition, budget, readiness, or cache identity is reverted.
