---
name: review-pr
description: Review pull requests and local branches for vllm-project/vllm-omni with a frozen diff, InferMatrixCopilot Direct routing, repository-specific contract checks, targeted validation, and evidence-backed P0-P2 findings. Use for maintainer review, re-review after updates, detailed or default PR review, and checks of correctness, compatibility, tests, benchmarks, or breaking changes. Do not use for an author's pre-submit self-check; use precheck-pr instead.
---

# Review vLLM-Omni Pull Requests

Review as a read-only maintainer. Find the smallest set of high-confidence,
actionable defects that matter to users or maintainers. Let CI and pre-commit
own formatting and already-reported failures.

Use InferMatrixCopilot Direct mode as the knowledge router when it is available.
Keep source inspection, judgment, and final wording in the host agent.

## Safety and review contract

- Freeze one base/head snapshot and bind every finding and validation result to
  that head SHA.
- Treat PR descriptions as navigation, never as proof that behavior works.
- Keep the review read-only. Do not post comments, submit a GitHub review, add
  labels, edit code, or push commits unless the user explicitly asks for that
  separate action.
- Never pass `post=true` to InferMatrixCopilot while investigating. When posting
  is explicitly authorized, validate locally first and publish one consolidated
  final review after confirming the head SHA is unchanged.
- Review only the changed behavior and pre-existing code needed to prove its
  runtime path. Do not report unrelated backlog.
- Do not stop merely because CI is pending or mergeability is unknown. Inspect a
  failing CI log only when its first failure overlaps the frozen diff or blocks
  the verdict.
- If the PR is a draft or marked WIP, report that state. Continue a local review
  when the user explicitly requested one, but do not post without separate
  authorization.

## Select the review surface

Use `vllm-project/vllm-omni` as the PR base repository. A fork or local checkout
is valid when it targets this project. For an unrelated repository, stop using
this skill and select that repository's review workflow.

For a GitHub PR, collect the snapshot before reading knowledge or source:

```bash
gh api "repos/vllm-project/vllm-omni/pulls/<PR>" \
  --jq '{base_sha: .base.sha, head_sha: .head.sha}'
REVIEW_FIELDS="number,url,title,body,isDraft,baseRefName,headRefName,mergeable,mergeStateStatus,statusCheckRollup,files"
gh pr view <PR> --repo vllm-project/vllm-omni \
  --json "${REVIEW_FIELDS}"
gh pr diff <PR> --repo vllm-project/vllm-omni
gh api "repos/vllm-project/vllm-omni/pulls/<PR>" \
  --jq '{base_sha: .base.sha, head_sha: .head.sha}'
```

Compare the base and head SHAs returned before and after fetching the diff. If
either changed, discard the metadata and diff and repeat against the new
snapshot. If either changes again, report the churn and wait for a stable review
target.

For a local branch or worktree, determine the target ref from the user, current
PR, or configured upstream. Do not infer it from the branch name. Resolve it
once, then collect committed, staged, unstaged, and in-scope untracked files:

```bash
git status --short
git rev-parse HEAD
git rev-parse <target-ref>
git merge-base <target-base-sha> HEAD
git diff --stat <comparison-commit>
git diff --name-status <comparison-commit>
git diff <comparison-commit>
git ls-files --others --exclude-standard
```

If no PR, branch, or usable current worktree is identifiable, ask for a PR URL,
PR number, or explicit base and head. Do not guess.

Default to a focused maintainer review. If the user asks for a detailed,
line-by-line, or audit review, expand coverage but keep the same evidence and
severity thresholds.

## Report the frozen snapshot first

Within 60 seconds of starting, and before reading InferMatrix knowledge,
searching source, or running tests, update the host conversation with:

```text
Pinned head: <SHA>
Base/comparison: <ref and SHA>
CI: <pass/fail/pending/not applicable>
Mergeability: <state/not applicable>
Preliminary findings: <brief finding or none yet>
```

Mark early findings as preliminary and continue the same review. This update is
not a GitHub comment.

## Route once with InferMatrixCopilot

After the snapshot update, call InferMatrixCopilot `review` exactly once with:

- `mode="direct"`
- `post=false`
- `repo="vllm-project/vllm-omni"`
- the frozen target plus `title`, `body`, and `changed_files`

For local changes without PR metadata, use the user request or commit subject as
the title and the task plus commit messages as the body. Mark that metadata as
synthetic and do not add validation claims that the author did not make.

If the user explicitly requests InferMatrix Strict mode, use `mode="strict"`
instead, keep `post=false`, pass the relevant local checkout through `repo_path`
when available, and poll `get_review_status`/`get_review_result` until terminal.
Do not run Direct and Strict for the same review. A request for a detailed review
alone does not imply Strict mode.

Use the embedded `quick_map` from each returned `knowledge_routes` entry. Open a
full routed page only when a concrete ambiguity blocks source review. Do not
restart from indexes or load broad rule catalogs. Treat the returned
`execution_budget` as a hard ceiling; use its one-time extension only for a
stated unresolved P1 or other high-risk contract.

If Direct returns no knowledge route, do not open a fallback index. Continue
with the repo-local routing reference and record the unrouted knowledge result
as a validation gap.

If InferMatrixCopilot is unavailable, say so once, continue with the repo-local
workflow below, and list the missing maintainer-knowledge routing as a review
gap. Do not install tools or change user configuration as part of a review.

## Review in evidence passes

Reuse one evidence packet containing the frozen SHAs, diff census, files read,
bounded searches, callers, tests, CI evidence, and findings. Run independent
read-only source and validation checks concurrently when the host supports it.

### 1. Establish intent and scope

- Group changed files into production code, tests, docs, configuration, build,
  and generated artifacts.
- Map each production file and test group to a claim in the PR or user request.
- Flag unrelated behavior or unexplained generated artifacts only when the diff
  proves they are in scope accidentally.
- Compare title/body claims with the actual diff. Verify linked issues only when
  they define the contract or reproduction.

### 2. Trace changed contracts end to end

For each changed value or behavior, trace:

```text
public ingress -> validation/defaulting -> producer -> transformations
  -> stage/worker/connector boundary -> final consumer -> error/cleanup owner
```

Check every applicable entry path, including offline/online,
streaming/non-streaming, sync/async, feature enabled/disabled, and compatibility
entrypoints. Search bounded callers and sibling implementations; do not assume
the changed hunk is the only path.

Pay particular attention to:

- stage and connector ownership, rank/world-size assumptions, and topology;
- tensor shape, dtype, device, batch expansion, and latent/KV/cache contracts;
- async cancellation, timeouts, locks, background work, and resource cleanup;
- public API/config defaults, aliases, serialization, validation, and backward
  compatibility;
- registries, exports, dependency declarations, docs, and examples;
- silent fallbacks, broad exception handling, and failure paths that turn
  invalid state into plausible output.

Read [references/review-routing.md](references/review-routing.md) after the diff
census and load only the repo-local skills and checks selected by that table.

### 3. Validate the changed path

Run a short import/version preflight before pytest. Then run the narrowest
relevant tests and low-cost static checks. Record each command, result, head SHA,
Python/platform, and dependency or lock fingerprint.

- Use existing CI as status evidence; do not claim local verification from CI.
- For docs-only changes, skip dependency setup and pytest. Check diff hygiene,
  links/build rules, and any live contract the docs describe.
- For hardware-dependent paths, run available CPU/static tests and state the
  exact GPU/NPU validation gap. Do not simulate hardware evidence.
- For performance or accuracy claims, require comparable baseline and PR runs on
  the same workload and environment. Verify warmup, repetitions, configuration,
  quality, latency/throughput, and peak memory as applicable.
- Stop when every changed semantic path has a supported finding or an explicit
  no-issue conclusion. Do not add searches only to increase confidence.

### 4. Run the subtraction check when triggered

Set `subtraction_signal="triggered"` when the diff adds or expands a helper,
class, fallback, compatibility branch, or public behavior. Check whether every
new behavior belongs to the stated goal, whether an existing owner can absorb
it, and whether any helper/branch can be deleted, merged, moved, or inlined.

Set `subtraction_signal="none"` for ordinary changes without those signals. Do
not force a minimality proof for a small fix.

## Calibrate findings

Report only defects with a concrete trigger, reachable path, impact, and
smallest safe fix direction:

- **P0:** security exposure, data corruption, or a change that makes the project
  broadly unusable.
- **P1:** likely runtime failure, wrong result, compatibility break, unsafe
  lifecycle, or missing validation/test evidence for a high-risk changed
  contract.
- **P2:** a real non-blocking defect or maintainability problem with a concrete
  future failure mode.

Drop speculative concerns, pure preferences, formatting nits, and requests for
tests that do not protect changed behavior. Prefer one root-cause finding over
several downstream symptoms.

Format each finding like an inline GitHub review comment:

```text
[P1] Short imperative title — path/to/file.py:<line>
<triggering input or call path>. <Observed behavior and impact>.
<Smallest safe fix direction>.
```

Use exact current diff lines. Keep internal rule IDs, coverage tables, and audit
matrices out of the user-facing review unless the user explicitly asks for the
full audit artifact.

## Complete and deliver

Before finalizing a Direct review, call InferMatrixCopilot
`validate_direct_review` with the actual `subtraction_signal`. For a triggered
signal, supply concrete subtraction items or a minimality proof. Use
`final_comment_count=1` for the single consolidated user-facing review result;
this count does not authorize or imply a GitHub post. A `partial_review` result
is not a complete review; name the missing evidence instead of claiming the
diff is clean.

Re-read the remote head SHA immediately before delivery or posting. If it
changed, mark prior validation stale and restart from the new snapshot.

Deliver findings first, ordered by severity. If there are no actionable
findings, say so briefly and name material validation gaps. Summarize the frozen
head, reviewed surface, checks run, and local verdict. Do not submit
`APPROVE`, `COMMENT`, or `REQUEST_CHANGES` unless the user explicitly asks to
publish that event.
