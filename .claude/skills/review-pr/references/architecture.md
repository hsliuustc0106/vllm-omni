# vLLM-Omni Architecture

Read this reference when a change crosses module boundaries or alters
configuration, orchestration, runtime ownership, connectors, or shared state.

## Source of truth

Prefer matching documents under `docs/design/module/**` on the reviewed branch.
Treat normative invariants as requirements, draft invariants as guidance to
verify against code and tests, and undocumented ownership changes as design
gaps. Do not turn this map into a second architecture contract.

## Ownership map

| Owner | Primary paths | Contract |
| --- | --- | --- |
| Entrypoints and I/O | `entrypoints/`, `inputs/`, `outputs/`, `request.py`, `data_entry_keys.py` | Validate public data, preserve identity/modality, and adapt errors without owning model policy. |
| Configuration | `config/`, `deploy/`, stage configuration | Resolve authoring inputs once; keep control-plane config transport-safe and runtime objects process-local. |
| Orchestration | `engine/`, `distributed/omni_coordinator/`, `distributed/ray_utils/` | Own routing, ordering, startup, cancellation, failure propagation, and shutdown. |
| Connectors | `distributed/omni_connectors/`, platform connectors | Transfer data without choosing stages; preserve schema, completion, timeout, backpressure, and cleanup. |
| AR runtime | `core/`, `worker/` | Preserve vLLM scheduler/request-state semantics; workers execute rather than route. |
| Diffusion runtime | `diffusion/sched/`, `executor/`, `worker/`, `diffusion_engine.py`, `ipc.py` | Keep one scheduler-owned request lifecycle and release state on every terminal path. |
| Model integration | `model_executor/`, `model_extras/`, `plugins/`, `diffusion/models/`, registries/loaders | Select through registries and keep model policy out of orchestration. |
| Platform layer | `platforms/`, platform attention/quantization | Isolate vendor imports and capability checks; preserve portable imports and supported fallbacks. |
| Shared state | KV/prefix/latent caches, quantization, metrics | Define identity, ownership, validity, disabled behavior, eviction, and cleanup. |

## Configuration boundary

Trace the live construction flow on the reviewed branch. On current `main`, the
primary path is:

```text
model/deploy inputs -> OmniConfigFactory -> VllmOmniConfig.from_pipeline_config()
  -> stage configs or the transitional legacy adapter -> StageRuntime/engine
```

- Load deploy/model metadata once and resolve precedence, stage overrides,
  connectors, requests, and placement before launch.
- Keep process-local device handles, workers, clients, and launch resources out
  of shared structured configuration.
- Let `StageRuntime` allocate replicas and devices, not reinterpret raw inputs.
- Treat `create_legacy_stage_configs_from_model()` as a transitional consumer:
  verify parity with `VllmOmniConfig` and do not add an independent resolution
  path or assume the migration is already complete.

## Boundary review

Trace the changed contract through:

```text
entrypoint -> config/I/O -> orchestrator -> scheduler -> worker/model
  -> connector or output conversion -> terminal cleanup
```

Inspect both producer and consumer. Verify identity, ordering, shape, dtype,
device, ownership, error propagation, cancellation, and cleanup. Treat silent
ownership movement as an architecture change requiring design evidence and
synchronized module documentation.
