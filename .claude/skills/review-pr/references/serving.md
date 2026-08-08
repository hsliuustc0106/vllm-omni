# Serving Review

Use for CLI/offline entrypoints, OpenAI-compatible APIs, request/response
schemas, streaming, engine orchestration, output processing, and lifecycle.

Official docs: [Chat Completions API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/chat_completions_api/)
and [API reference](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/).

## Boundaries

Start with `vllm_omni/entrypoints/` and `vllm_omni/engine/`. Route config
construction to Configuration, worker/model execution to Model Executor, and
connector data movement to Distributed.

## Rules

- Preserve request identity, modality, ordering, validation, defaults, and error
  semantics from public ingress through the selected engine path.
- Verify sync/async and offline/online parity only for advertised modes; check
  streaming chunk order, terminal events, disconnects, and partial outputs.
- Make startup, readiness, admission, cancellation, failure propagation, and
  shutdown explicit across every owned task, process, stage, and replica.
- Keep routing and output assembly in serving owners rather than model-specific
  code; do not silently change public response or metrics schemas.
- Bound queues, retries, timeouts, metric cardinality, and user-controlled media
  or URL handling.

Require focused protocol/lifecycle tests and one representative production-path
request for changed public behavior. Document API, CLI, default, and error
contract changes.
