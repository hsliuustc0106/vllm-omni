# General Review Checks

Apply these checks to every PR after freezing the review surface. A pattern is
not a finding until the diff introduces or exposes a reachable trigger, impact,
and smallest safe fix. Do not report unrelated backlog.

Official docs: [contributing guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/)
and [feature compatibility](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/feature_compatibility/).

## Contract and scope

- Match the title/body claims to every changed production path, test, config,
  dependency, generated artifact, and user-visible document.
- Trace each changed value through public ingress, validation/defaulting,
  producer, transformations, process or device boundaries, final consumer, and
  terminal cleanup.
- Check every applicable sync/async, offline/online, streaming/non-streaming,
  feature-off, topology, and compatibility path without demanding unsupported
  modes merely for symmetry.

## Module, API, and class design

Apply these rules to new or materially changed production code; do not require
unrelated cleanup from untouched files.

1. Start each source module with a concise module docstring that defines its
   responsibility and boundaries and identifies its primary classes and key
   functions. Place it after any required license header and before imports.
2. Give every added or materially changed class and function a clear docstring
   covering its responsibility, parameters, return value, raised exceptions,
   and non-obvious invariants. Use explicit typed parameters at owned API
   boundaries; avoid opaque `**kwargs` and `Any`. Allow them only when an
   upstream or compatibility interface requires them, and document and validate
   the accepted keys and concrete types.
3. Move stable helpers shared by multiple features into an appropriately scoped
   utility module. Do not let feature modules accumulate large collections of
   unrelated private utility functions; keep a private helper local only when
   it is single-purpose and tightly coupled to that feature.
4. Keep every method on a key class within that class's stated responsibility,
   and document non-obvious design decisions and invariants. Before adding a
   method, prefer reuse, consolidation, or extension of an existing abstraction;
   split the class when new behavior creates a separate responsibility or makes
   the class unnecessarily long.

## Blocking risk scan

| Risk | Prove before reporting |
| --- | --- |
| Correctness | A live input reaches a wrong output, exception, silent drop, shape/dtype/device mismatch, or partially initialized state. |
| Compatibility | A supported API, config key, default, serialization schema, model identifier, or caller breaks without validation, migration, or deprecation. |
| Lifecycle | Allocation or startup lacks cleanup on partial failure, timeout, cancellation, shutdown, or repeated requests. |
| Concurrency/distribution | Ordering, identity, rank, world size, completion, timeout, backpressure, or collective assumptions fail on a supported path. |
| Cache/shared state | Identity omits a correctness-affecting input, or isolation, invalidation, disabled behavior, eviction, or cleanup is incomplete. |
| Async behavior | Blocking I/O, sleep, a lock across `await`, or unnecessary serialization harms a reachable async path. |
| Security/data handling | Secrets, user payloads, unsafe deserialization/eval/shell, unvalidated paths, or unbounded metric labels are reachable. |
| Validation evidence | Tests bypass the production dispatcher, mock away the change, use unrealistic types/MRO, or cannot fail when the defect returns. |
| User contract | User-visible model, feature, API, CLI, config, default, or compatibility behavior changes without accurate docs or examples. |

Search bounded callers and sibling implementations before reporting dead code,
duplication, or a missed consumer. Keep new abstractions only when they have a
distinct live caller, invariant, or compatibility need.

## Finding bar

Anchor each finding to a changed `path:line`; name the trigger or call path,
current behavior, impact, and smallest fix direction. Treat pending CI, missing
hardware, or unsupported measurements as validation gaps unless the repository
contract makes that evidence a merge requirement.
