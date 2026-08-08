---
name: review-pr
description: Review pull requests and local branches for vllm-project/vllm-omni with a frozen snapshot, repo-local owner and domain routing, targeted validation, and concise evidence-backed findings. Use for default, detailed, or repeat maintainer reviews; checking correctness, compatibility, tests, benchmarks, model additions, distributed changes, or breaking behavior; and identifying or explicitly requesting the most relevant code-owner reviewers. Use precheck-pr instead for an author's pre-submit self-check.
---

# Review vLLM-Omni Pull Requests

Review like a maintainer: direct, selective, and focused on issues that CI does
not prove. Prefer a few high-confidence findings over exhaustive commentary.
Zero findings is a valid result.

## Quality contract

Make every finding:

- **Correct:** prove a reachable failure, not a suspicion.
- **Prioritized:** lead with merge blockers and high-impact defects.
- **Actionable:** identify the smallest safe fix direction.
- **Evidence-based:** cite code, tests, docs, CI, or measurements.
- **Concise:** avoid review templates and repeated summaries.
- **Calibrated:** match severity to user and maintainer impact.

Do not report unrelated backlog, style already enforced by pre-commit, or a
missing test that would not protect changed behavior.

## Select the input and depth

Use `vllm-project/vllm-omni` as the base repository. Accept its forks and local
checkouts; use another skill for unrelated repositories.

| Input | Review surface |
| --- | --- |
| PR number or URL | Frozen PR metadata, full diff, and relevant threads. |
| Local branch/worktree | Frozen target-base SHA through committed, staged, unstaged, and in-scope untracked changes. |
| Pre-filled context | Reuse supplied metadata; fetch only missing facts and the full diff. |

Default to maintainer brevity. A detailed or audit request expands coverage and
lists `path:line` findings, but keeps the same confidence and severity bar.

## Reference guide

Read [review-execution.md](references/review-execution.md) and
[general-checks.md](references/general-checks.md) for every review. After the
diff census, read [review-routing.md](references/review-routing.md), select one
primary owner, and load a second owner only for a real cross-boundary call path.

Each concise reference links to the maintained
[vLLM-Omni documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/).
For branch-specific behavior, inspect the matching `docs/` file in the reviewed
checkout first; use the published latest docs for current guidance and discovery.
If docs and live code disagree, verify the code/tests and report the drift.

### Primary owners

| Reference | Read when |
| --- | --- |
| [configuration.md](references/configuration.md) | Config construction, deploy YAML, schema, defaults, registry, or topology changes. |
| [serving.md](references/serving.md) | Entrypoints, request/response behavior, streaming, orchestration, or engine lifecycle changes. |
| [model-executor.md](references/model-executor.md) | Model loading, stage inputs, runners, workers, or device startup changes. |
| [diffusion-checklist.md](references/diffusion-checklist.md) | A diffusion pipeline, model, scheduler, cache, or feature changes. |
| [distributed.md](references/distributed.md) | Connectors, KV transfer, collectives, routing, or cross-stage communication changes. |
| [scheduler.md](references/scheduler.md) | Request state, token budgets, KV readiness, or prefix-cache behavior changes. |

### Cross-cutting references

| Reference | Read when |
| --- | --- |
| [model-addition-checklist.md](references/model-addition-checklist.md) | A model, architecture, loader, processor, registry, or stage config is added. |
| [platform-checks.md](references/platform-checks.md) | Hardware, kernel, attention backend, quantization, or vendor code changes. |
| [perf-verification.md](references/perf-verification.md) | The PR makes a latency, throughput, memory, or quality claim. |
| [test-quality-evaluation.md](references/test-quality-evaluation.md) | Tests change, are absent for risky code, or may not exercise production behavior. |
| [tests-docs-checklist.md](references/tests-docs-checklist.md) | Coverage, CI markers, examples, user docs, or PR evidence need review. |
| [verification.md](references/verification.md) | Hardware, a server, or a runnable affected path is available for active verification. |
| [maintainer-style-study.md](references/maintainer-style-study.md) | Findings are ready for concise maintainer-style delivery. |
| [review-requests.md](references/review-requests.md) | The user asks to identify, suggest, request, or ping code-owner reviewers. |

## Workflow

### 1. Freeze and report the snapshot

Pin the base and head before reading source or running validation. Within 60 seconds,
report the pinned head, CI, mergeability, and preliminary findings in the host
conversation. Do not wait for CI or post this update to GitHub.

If the target changes while fetching, discard the evidence and retry once. If
it changes again, report the churn and wait for a stable target.

For a PR, materialize the pinned head in an isolated detached worktree and run
all source inspection and validation there. For a local review, freeze the
committed, index, worktree, and NUL-safe in-scope untracked contents. Follow
[review-execution.md](references/review-execution.md) for the required identity
assertions and final byte-for-byte staleness check.

### 2. Build the diff census

Group files into production code, tests, docs, configuration, build/CI, and
generated artifacts. Map each changed production file and test group to the PR
goal. Compare the title/body claims with the actual diff; use linked issues only
when they define the contract or reproduction.

Mark unrelated scope and unexplained generated artifacts. Do not infer behavior
from the PR description without tracing the live code.

### 3. Route from the live behavior

Trace each claimed behavior through the changed producer to its live consumer,
then use [review-routing.md](references/review-routing.md) to select one primary
owner, a second owner only for a real cross-boundary call path, and the smallest
applicable cross-cutting overlays. Treat titles and changed paths as hints; the
live producer-consumer contract and repo-local routing map are authoritative.
For docs-, tests-, or CI-only changes with no production behavior, use the
general checks and applicable cross-cutting references.

### 4. Run the blocker scan

Apply every category in [general-checks.md](references/general-checks.md) before
lower-priority comments.

For each changed value or behavior, trace:

```text
public ingress -> validation/defaulting -> producer -> transformations
  -> stage/worker/connector boundary -> final consumer -> terminal cleanup
```

Cover every applicable offline/online, streaming/non-streaming, sync/async,
feature-on/off, topology, and compatibility path. Search bounded callers and
sibling implementations rather than assuming the changed hunk is the only path.

### 5. Apply the selected domain checks

Apply the reference set selected in step 3 and any matching repo-local domain
skill. Inspect both sides of any config, registry, serialization, connector,
cache, or stage boundary.

When a diff adds or expands a helper, class, fallback, compatibility branch, or
public behavior, run a subtraction pass: remove out-of-scope behavior and check
whether each new abstraction can be deleted, merged, moved, or inlined.

### 6. Verify the changed path

Run an import/version preflight, then the narrowest relevant tests and low-cost
static checks. Bind every result to the head SHA and environment fingerprint.

- Treat CI as status evidence; inspect only the first overlapping failure.
- For docs-only changes, use diff hygiene, links/build checks, and bounded live
  contract verification instead of dependency setup or pytest.
- For hardware-dependent paths, run available static/CPU checks and name the
  exact GPU/NPU gap. Never simulate device evidence.
- For performance or accuracy claims, require comparable base/head runs with
  the same environment, workload, warmup, repetitions, and quality criteria.

Stop when each changed semantic path has a supported finding or an explicit
no-issue conclusion. Do not search further only to increase confidence.

### 7. Consolidate and deliver

Verify each finding against the current diff, deduplicate by root cause, and
order by severity.

Re-read the remote head immediately before delivery. If it changed, mark the
review stale and restart from the new snapshot.

Return findings first. Use
[maintainer-style-study.md](references/maintainer-style-study.md) to keep them
direct and brief. Each finding must include an exact `path:line`, trigger or
call path, current behavior, impact, and smallest fix direction. If there are
no findings, say so briefly and name material validation gaps.

Keep the review read-only unless the user explicitly authorizes posting. Do not
submit `APPROVE`, `COMMENT`, or `REQUEST_CHANGES`, add labels, edit code, or push
commits as an implied part of review.

### 8. Optionally request focused owner reviews

Only when the user asks to identify or request reviewers, read
[review-requests.md](references/review-requests.md). Rank path-matched
CODEOWNERS with documented domain expertise and propose one to three focused
reviewers with an explicit rationale.

Identifying or suggesting reviewers is read-only. Requesting reviewers or
posting `@mention` comments changes external state and requires explicit user
authorization. When authorized, recheck the head, deduplicate existing
requests, and post at most one consolidated comment. Do not infer this
permission from a request to review the code.
