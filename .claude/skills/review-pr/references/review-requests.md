# Focused Code-Owner Review Requests

Read only when the user asks to identify, suggest, request, add, or ping
reviewers. Finding candidates is read-only; GitHub review requests and
`@mention` comments are external writes.

Owner sources: the reviewed head's
[CODEOWNERS](https://github.com/vllm-project/vllm-omni/blob/main/.github/CODEOWNERS)
and [governance expertise](https://docs.vllm.ai/projects/vllm-omni/en/latest/community/governance/).
Use the reviewed head first because ownership changes over time; do not copy a
static owner list into review output.

## Rank candidates

1. Freeze the PR head and collect changed files, author, requested reviewers,
   submitted reviews, and existing comments.
2. Resolve the **last matching CODEOWNERS rule** for each changed file. Do not
   union an earlier broad rule with a later specific rule.
3. Group files by the selected primary owner and the contract needing review.
   Weight production owners and live consumers above tests/docs/CI owners and
   the repository-wide fallback.
4. Use documented governance specialties only to choose among matching owners:
   for example serving versus configuration, diffusion model versus cache or
   parallelism, TTS versus generic model execution, or CI versus tests.
5. Exclude the PR author, bots, users who already completed the needed review,
   and duplicate pending requests. Do not infer availability or invent
   expertise from a username.
6. Select one reviewer for a single-owner change and at most three for a real
   cross-domain change. If more are plausible, rank them but do not ping a
   broad group; ask the user before exceeding three.

If every specific owner is excluded, consider one repository-wide fallback
owner and label that choice as a fallback rather than a path match. Never fan
out to the full fallback list. Prefer a qualified owner who covers multiple
critical path groups over separate reviewers for each file. If CODEOWNERS is
missing or invalid, state that gap and use governance only as a labeled fallback.

Recent file history may break a tie only when CODEOWNERS and governance remain
ambiguous. Never use commit count alone to override an explicit current owner.

## Preview before writing

Show the user the exact candidate set and rationale:

```text
@owner — <owned paths or domain>; please verify <specific changed contract>.
```

Draft one consolidated comment tied to the frozen head:

```text
Requesting focused review for `<short-head-sha>`:

- @owner1: could you check <contract> in `<path/group>`?
- @owner2: could you check <cross-boundary contract> in `<path/group>`?

Relevant validation: <one concise result or exact remaining gap>.
```

Mention only owners being requested. Keep the reason specific enough that each
person knows what decision or invariant needs attention; do not post a generic
“please review” or a generated ownership audit.

## External-write gate

- “Who should review?”, “suggest reviewers”, or “draft a request” authorizes
  only the preview.
- “Request/add these reviewers” authorizes the GitHub review-request mutation.
- “Ping/comment/tag the owners” authorizes the consolidated comment.
- When the user explicitly asks to request reviews **with an owner comment**,
  both writes are authorized. Obey narrower wording when only one action is
  requested.

Immediately before an authorized write, verify the PR is still open, the head
SHA is unchanged, and the candidate is not the author or already satisfied.
Report draft/WIP state and avoid notifications unless the user explicitly wants
reviewers requested before ready-for-review.

Use the GitHub connector for reviewer requests and the single top-level comment;
use `gh` only when connector coverage is unavailable. Do not submit an approval,
request-changes event, label, assignment, or multiple comments as part of this
action. After writing, reread the PR and report the reviewers requested and the
comment URL. If the head changed, discard the draft and rerank before posting.
