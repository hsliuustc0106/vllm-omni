---
name: review-pr
description: Review pull requests and local branches for vllm-project/vllm-omni with a frozen diff, InferMatrixCopilot routing, a vLLM-Omni blocker scan, targeted validation, and concise evidence-backed findings. Use for default, detailed, or repeat maintainer reviews and for checking correctness, compatibility, tests, benchmarks, model additions, distributed changes, or breaking behavior. Use precheck-pr instead for an author's pre-submit self-check.
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
InferMatrix Direct remains the default; use Strict only when explicitly asked.

## Reference guide

Read [review-execution.md](references/review-execution.md) for every review.
After the diff census, read [review-routing.md](references/review-routing.md),
then load only the references selected for the changed behavior:

| Reference | Read when |
| --- | --- |
| [architecture.md](references/architecture.md) | Ownership, configuration, orchestration, connector, or cross-stage boundaries change. |
| [blocker-patterns.md](references/blocker-patterns.md) | Production code adds a known correctness, lifecycle, compatibility, or security risk. |
| [diffusion-checklist.md](references/diffusion-checklist.md) | A diffusion pipeline, model, scheduler, cache, or feature changes. |
| [maintainer-style-study.md](references/maintainer-style-study.md) | Findings are ready for concise maintainer-style delivery. |
| [model-addition-checklist.md](references/model-addition-checklist.md) | A model, architecture, loader, processor, registry, or stage config is added. |
| [perf-verification.md](references/perf-verification.md) | The PR makes a latency, throughput, memory, or quality claim. |
| [test-quality-evaluation.md](references/test-quality-evaluation.md) | Tests change, are absent for risky code, or may not exercise production behavior. |
| [tests-docs-checklist.md](references/tests-docs-checklist.md) | Coverage, CI markers, examples, user docs, or PR evidence need review. |
| [verification.md](references/verification.md) | Hardware, a server, or a runnable affected path is available for active verification. |

## Workflow

### 1. Freeze and report the snapshot

Pin the base and head before reading knowledge or source. Within 60 seconds,
report the pinned head, CI, mergeability, and preliminary findings in the host
conversation. Do not wait for CI or post this update to GitHub.

If the target changes while fetching, discard the evidence and retry once. If
it changes again, report the churn and wait for a stable target.

### 2. Route once with InferMatrixCopilot

After the progress update, call InferMatrixCopilot `review` once with the frozen
target, title, body, changed files, `repo="vllm-project/vllm-omni"`, and
`post=false`.

- Use `mode="direct"` unless the user explicitly requests Strict.
- In Direct mode, use only returned `quick_map` routes. Open a full route only
  when a concrete ambiguity blocks source review.
- Do not open a fallback index when routing returns no match; continue with the
  repo-local routing reference and record the knowledge gap.
- Treat the returned execution budget as a hard ceiling. Use its one extension
  only for a stated unresolved P1 or other high-risk contract.
- In Strict mode, poll the returned run to terminal; do not also run Direct.

If InferMatrixCopilot is unavailable, continue locally and state the missing
maintainer-knowledge routing. Do not install or reconfigure tools during review.

### 3. Build the diff census

Group files into production code, tests, docs, configuration, build/CI, and
generated artifacts. Map each changed production file and test group to the PR
goal. Compare the title/body claims with the actual diff; use linked issues only
when they define the contract or reproduction.

Mark unrelated scope and unexplained generated artifacts. Do not infer behavior
from the PR description without tracing the live code.

### 4. Run the blocker scan

Audit these categories internally before lower-priority comments:

1. correctness;
2. reliability and lifecycle safety;
3. breaking API, config, or default behavior;
4. test and validation evidence;
5. user-facing documentation;
6. security and data exposure.

For each changed value or behavior, trace:

```text
public ingress -> validation/defaulting -> producer -> transformations
  -> stage/worker/connector boundary -> final consumer -> terminal cleanup
```

Cover every applicable offline/online, streaming/non-streaming, sync/async,
feature-on/off, topology, and compatibility path. Search bounded callers and
sibling implementations rather than assuming the changed hunk is the only path.

### 5. Apply the selected domain checks

Use [review-routing.md](references/review-routing.md) to choose the smallest
applicable reference set and any existing repo-local domain skill. Inspect both
sides of any config, registry, serialization, connector, cache, or stage
boundary.

When a diff adds or expands a helper, class, fallback, compatibility branch, or
public behavior, run a subtraction pass: remove out-of-scope behavior and check
whether each new abstraction can be deleted, merged, moved, or inlined. Record
`subtraction_signal="none"` for ordinary changes without those signals.

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
order by severity. Before finalizing Direct mode, call
`validate_direct_review` with the actual subtraction signal and exactly one
consolidated final result.

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
