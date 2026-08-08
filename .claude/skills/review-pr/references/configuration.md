# Configuration Review

Use for config construction, deploy YAML, pipeline/endpoint registries, schema,
defaults, aliases, CLI projection, and topology changes.

Official docs: [configuration](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/)
and [architecture overview](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/architecture_overview/).

## Boundaries

Start with `vllm_omni/config/` and `vllm_omni/deploy/`; inspect affected
entrypoint and engine consumers. Keep HTTP request policy in Serving and final
stage/device startup in Model Executor.

## Rules

- Resolve precedence once across structured config, legacy adapters, CLI/direct
  kwargs, defaults, aliases, and stage overrides.
- Reject unknown or owner-mismatched fields explicitly; preserve transport-safe
  control-plane data and keep process-local runtime objects out of shared config.
- Verify default, explicit, and feature-off values reach every live factory and
  consumer without silent reinterpretation.
- Treat deploy stage count, placement, connector, parallelism, and memory fields
  as one topology contract; validate impossible combinations before launch.
- Preserve parity between supported structured, legacy, and default construction
  paths until a documented migration removes one.

Require focused normalization/schema tests, a live consumer assertion, and docs
for public keys, defaults, constraints, or migration behavior.
