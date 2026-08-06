# Blocker Patterns

Use this catalog only for risks introduced or exposed by the diff. A pattern is
not a finding until a reachable trigger, impact, and smallest safe fix are
proved; existing backlog is not review scope.

| Signal | Prove before reporting |
| --- | --- |
| Bare or broad exception handling | An expected failure is hidden, converted to a false success, or loses actionable context on init, config, loading, request, or execution paths. |
| Unchecked optional state | A live path dereferences or consumes `None` or a partially initialized value. |
| Stringly typed `**kwargs` or payload keys | Producer and consumer can drift, misspell, silently drop, or reinterpret a field; prefer explicit types or one validated accessor. |
| Misleading `Any` or annotation | The declared contract permits an invalid value or hides a real producer-consumer mismatch. |
| Public signature/default removal or alias | A supported caller breaks without validation, compatibility, deprecation, or migration guidance. |
| Blocking or serialized async work | Blocking I/O, sleep, a lock across `await`, or sequential awaits reduce correctness or concurrency on a reachable async path. |
| Incomplete lifecycle | Allocation/start lacks cleanup for partial failure, timeout, cancellation, shutdown, or repeated requests. |
| Connector mismatch | Sender and receiver disagree on identity, shape, dtype, device, ordering, completion, timeout, or ownership. |
| Distributed/platform assumption | Rank, world size, device, collective, or vendor behavior fails on a supported topology or common import path. |
| Cache/state collision | A correctness-affecting input is absent from identity, or invalidation, isolation, disabled behavior, eviction, or cleanup is incomplete. |
| Tensor/media mismatch | Layout, batch/CFG expansion, sampling rate, serialization, dtype, device, empty value, or final modality changes before consumption. |
| Mixin/MRO initialization | Production inheritance skips required mixin state while a simplified test double hides the failure. |
| Unsafe input or data handling | Secrets, user payloads, unsafe deserialization/eval/shell, unbounded metric labels, or unvalidated paths are reachable. |
| Weak regression evidence | Tests bypass the production dispatcher, over-mock the changed behavior, use unrealistic types/MRO, or cannot fail when the bug returns. |

For suspected duplication or dead code, search bounded call sites and sibling
implementations. Request deletion or consolidation only when the behavior has no
live caller, distinct invariant, compatibility need, or owner.

Anchor the finding to the changed line, name the trigger/call path, describe the
observed failure, and recommend the smallest fix direction. Turn unsupported
hardware or missing measurements into a validation gap unless the current
repository contract explicitly requires them.
