# Distributed Review

Use for connectors, KV transfer, coordinator/load balancing, collectives,
cross-stage routes, ports, and distributed data movement.

Official docs: [disaggregated inference](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/disaggregated_inference/)
and [design documents](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/).

## Boundaries

Start with `vllm_omni/distributed/` and both sides of the changed transfer.
Scheduler owns wait/readiness state; Serving owns stage lifecycle and request
orchestration; Configuration owns backend/topology selection.

## Rules

- Prove sender/receiver agreement on request and stage identity, shape, dtype,
  device, ordering, chunk boundaries, completion, and ownership.
- Validate rank/world-size and replica/topology assumptions for every supported
  path touched by the diff; keep collectives symmetric on success and failure.
- Bound connect, send, receive, retry, and readiness waits with actionable error
  propagation, backpressure, cancellation, and terminal cleanup.
- Keep backend selection and fallback explicit; never silently reinterpret or
  drop payloads across SHM, network, or vendor connectors.
- Prevent route/port collisions and stale state across repeated startup,
  partial failure, replica replacement, and shutdown.

Require focused producer-consumer tests and, when runnable, the smallest
multi-process or multi-rank topology that exercises the changed contract.
