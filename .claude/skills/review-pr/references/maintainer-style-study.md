# Maintainer Review Style

Use this reference only when turning proved findings into comments. The baseline
maintainer samples favor short, direct comments and reserve long explanations
for architecture or non-obvious failure paths.

## Write the finding

- Lead with the issue or decisive question; omit a review preamble.
- Name the concrete trigger and impact, then the smallest fix direction.
- Keep obvious fixes to one sentence. Use more detail only when the call path or
  ownership boundary is otherwise unclear.
- Prefer one root-cause comment over repeated symptom comments.
- Hedge only when evidence is genuinely incomplete; mark that item as a
  question or validation gap rather than overstating severity.
- Say briefly that no findings were found when the review is clean; zero
  comments is a valid outcome when posting is authorized.

Useful forms:

```text
This leaves <state> allocated when <failure/cancellation path>. Release it in
the shared terminal cleanup path.
```

```text
Does this producer also update <consumer>? It still reads the old field here.
```

Avoid generic praise, “Nit:” prefixes, dramatic labels, audit templates, rule
IDs, comment-count narration, and “I left a few comments.” Keep approvals,
review events, and GitHub replies subject to the user's explicit authorization.
