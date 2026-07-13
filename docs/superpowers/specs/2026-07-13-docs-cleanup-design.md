# Docs Cleanup & De-duplication — Design

**Date:** 2026-07-13
**Status:** Proposed
**Scope:** Fix broken/stale docs + reduce duplication. Explicitly excludes IA restructure, build-time-mutation fixes, and a docs CI gate.

---

## 1. Motivation

The vLLM-Omni docs surface (~145 tracked Markdown files under `docs/`, 286 across the repo) has accumulated three classes of rot:

- **Nav bugs** — broken references, duplicate entries, and collisions that break or confuse `mkdocs build --strict`.
- **Orphans** — valuable pages that exist on disk but are reachable by no navigation path, plus one stale file pointing at a deleted example source.
- **Duplication signals** — design-doc ↔ user-guide pairs that cover the same feature without cross-linking, a filename mismatch (`vae_parallel` vs `vae_parallelism`), and three `metrics.md` files that are not three-of-a-kind.
- **Truth drift** — the FAQ claims features are "planned" that already ship; Python-version support is stated four different ways; the recipe ownership rule contradicts itself between two pages; `DOCS_GUIDE.md` describes a directory layout and deployment workflow that no longer exist.

An audit by a second reviewer (codex) surfaced additional issues — most notably that `mkdocs build` mutates tracked source via hooks (`generate_examples.py`, `generate_api_readme.py`, `generate_argparse.py`), and that an isolated strict build dirties ~26 tracked files. **That stabilization work is explicitly out of scope here** and tracked as a follow-up (see §10). This design accepts the mutation as a known condition and scopes the cleanup so each fix is robust to it.

## 2. Goals & non-goals

**Goals**
1. Make navigation correct: no broken references, no duplicate entries, no glob-vs-explicit collisions.
2. Make every valuable page reachable; delete the one true stale orphan.
3. Resolve the design↔user-guide duplication by adding bidirectional cross-links (not merging).
4. Fix filename inconsistencies that make pairs hard to find (`vae_parallel` / `vae_parallelism`; `metrics.md` ×3).
5. Correct factual drift on high-traffic pages (FAQ, Python versions, recipe ownership).
6. Rewrite `DOCS_GUIDE.md` to match reality.
7. Remove stray root-level files that don't belong to the project.

**Non-goals**
- Fixing build-time mutation (hooks writing tracked source). Tracked in §10.
- Adding a `scripts/docs-check` CI gate. Tracked in §10.
- Restructuring the information architecture or nav taxonomy.
- Splitting/rewriting the contributor model-onboarding docs.
- Golden-path journey consolidation, versioned docs, governance changes.

## 3. Decisions already confirmed with the user

- **Goals:** fix broken/stale docs + reduce duplication (selected).
- **Design↔user-guide split policy:** keep the split, add bidirectional "See also" cross-links. Do **not** consolidate or merge. This mirrors the pattern already established by the `step_execution` and `request_batching` pairs, which explicitly cross-reference each other.
- **Execution:** plan only. Implementation happens in a later session.
- **Stabilization scope:** de-dupe only; accept the hook-mutation problem as a known follow-up rather than absorbing codex's Phase 1–2 work.

## 4. Inventory of work

Each item below is tagged with its source (`[mine]` = found in my audit, `[codex]` = found in codex's audit, `[both]` = found by both). All findings were re-verified against `origin/main` (local was 61 commits behind at audit time; the nav bugs and orphans persist upstream).

### 4.1 Nav correctness (Batch 1 — mechanical)

| # | File | Problem | Fix |
|---|------|---------|-----|
| 1.1 | `docs/contributing/ci/.nav.yaml` `[both]` | References `test_markers.md` and `test_style.md` (singular); files on disk are `tests_markers.md` / `tests_style.md` (plural). Two broken nav entries. | Rename nav entries to `tests_markers.md` / `tests_style.md`. |
| 1.2 | `docs/.nav.yml` `[both]` | `design/feature/vae_parallel.md` listed twice (lines ~115 and ~120). | Remove the duplicate line. |
| 1.3 | `docs/.nav.yml` `[mine]` | `contributing/README.md` and `contributing/DOCS_GUIDE.md` are each reachable twice — once via explicit entry and once via the `glob: contributing/*`. | Make the glob exclude `README.md` and `DOCS_GUIDE.md`, or drop the explicit entries and rely on the glob + awesome-nav `flatten_single_child_sections`. Prefer the exclusion approach (keeps explicit titles). |

### 4.2 Orphan recovery (Batch 2)

| # | File | Status | Action |
|---|------|--------|--------|
| 2.1 | `docs/design/metrics.md` `[mine]` | Prometheus metrics design doc; unreferenced in `.nav.yml`. | Add under Design Documents → top level (alongside `architecture_overview.md`). |
| 2.2 | `docs/design/feature/pipeline_parallel.md` `[mine]` | Design doc for pipeline parallelism; every sibling design doc is listed except this one. | Add to the Feature Design section of `.nav.yml`. |
| 2.3 | `docs/user_guide/diffusion/parallelism/pipeline_parallel.md` `[mine]` | User-guide counterpart; the parallelism nav lists overview/cfg/expert/hsdp/sequence/tensor/vae but omits pipeline. | Add to the Parallelism subsection of `.nav.yml`. |
| 2.4 | `docs/user_guide/diffusion/mot_config.md` `[mine]` | Auto-tuned Triton kernel configs for MoT; no H1, unreferenced. | Add an H1 heading and wire into the Diffusion Features section. If the content is too thin to publish, move to `design/` as internal reference instead — decide at implementation time. |
| 2.5 | `docs/design/qwen3_omni_tts_performance_optimization.md` `[mine]` | Performance optimization writeup; unreferenced. | Add under Design Documents. |
| 2.6 | `docs/discussion/world_action_model_analysis.md` `[mine]` | Analysis doc; entire `discussion/` dir unreferenced. Has a stale `.html` twin on disk. | Wire the `.md` into nav under a new "Discussion" subsection of Developer Guide, OR relocate into `design/`. Delete the `.html` twin (the `.md` is the source of truth). Preference: relocate to `design/` to avoid inventing a new top-level section. |
| 2.7 | `docs/user_guide/examples/online_serving/text_to_audio.md` `[mine]` | **True stale orphan.** At `origin/main` it points to `examples/online_serving/stable_audio`, which does not exist. The `generate_examples.py` hook cannot regenerate it because the source dir is gone. | Delete the file. (Note: `examples/offline_inference/text_to_audio/` still exists and is unrelated.) |

**Hook-mutation caveat for 2.1–2.6:** the `generate_examples.py` hook rewrites the Examples section of `.nav.yml` on every build (line 315: `open(NAV_FILE, "w")`). The orphan fixes above are all *outside* the Examples section, so the hook will preserve them. The `text_to_audio.md` deletion (2.7) is inside the hook-managed area — deleting the file is safe because the hook will simply not re-add it (no source dir to generate from). This is confirmed by reading the hook's category-scan logic (`generate_examples.py:333-345`).

### 4.3 De-duplication via cross-linking (Batch 2)

The design-doc ↔ user-guide pairs are an **intentional author-vs-user split**, not redundant copies. The fix is bidirectional "See also" pointers, not merging. Two pairs (`step_execution`, `request_batching`) already do this correctly and serve as the template.

Template to add at the top of each design doc (after the H1):

```markdown
> 📖 **End-user guide:** For enablement, configuration parameters, and Quick Start,
> see [`<relative path to user guide>`](<path>).
```

Template to add at the top of each user guide (after the H1):

```markdown
> 🔧 **Design doc:** For implementation details, internal APIs, and architecture,
> see [`<relative path to design doc>`](<path>).
```

Pairs to cross-link (the 8 that currently lack mutual links):

| Design doc | User guide |
|---|---|
| `design/feature/cfg_parallel.md` | `user_guide/diffusion/parallelism/cfg_parallel.md` |
| `design/feature/expert_parallel.md` | `user_guide/diffusion/parallelism/expert_parallel.md` |
| `design/feature/hsdp.md` | `user_guide/diffusion/parallelism/hsdp.md` |
| `design/feature/sequence_parallel.md` | `user_guide/diffusion/parallelism/sequence_parallel.md` |
| `design/feature/tensor_parallel.md` | `user_guide/diffusion/parallelism/tensor_parallel.md` |
| `design/feature/vae_parallel.md` | `user_guide/diffusion/parallelism/vae_parallelism.md` |
| `design/feature/cache_dit.md` | `user_guide/diffusion/cache_acceleration/cache_dit.md` |
| `design/feature/teacache.md` | `user_guide/diffusion/cache_acceleration/teacache.md` |

Plus `pipeline_parallel` once it's wired into nav (§4.2 items 2.2 + 2.3).

### 4.4 Filename inconsistencies (Batch 2)

| # | Problem | Fix |
|---|---------|-----|
| 4.4a | `design/feature/vae_parallel.md` vs `user_guide/diffusion/parallelism/vae_parallelism.md` — singular vs `-ism` suffix. `[mine]` | Rename the user-guide file to `vae_parallel.md` for consistency with its design counterpart and the other parallelism files (`cfg_parallel`, `tensor_parallel`, etc.). Update the `.nav.yml` entry and any inbound links. |
| 4.4b | Three `metrics.md` files are not three-of-a-kind: `design/metrics.md` + `usage/metrics.md` are a Prometheus pair (intentionally split); `contributing/metrics.md` is about `--log-stats` console log tables — a different topic. `[mine]` | Rename `contributing/metrics.md` → `contributing/log_stats.md`. Add a hat-note at the top of both Prometheus docs pointing to each other, and a hat-note on the renamed `log_stats.md` distinguishing it from the Prometheus docs. Update `.nav.yml` and inbound links. |

### 4.5 Truth corrections (Batch 3)

| # | File | Problem | Fix |
|---|------|---------|-----|
| 5.1 | `docs/usage/faq.md` `[codex]` | Says quantization is "planned to introduce ... in version 0.16.0" and streaming is "Not yet." Both ship today (extensive `user_guide/quantization/` docs; streaming APIs in `serving/`). | Rewrite both Q&As to reflect current state. Link to the quantization overview and the streaming endpoints. |
| 5.2 | Python version drift `[codex]` | `pyproject.toml` says `>=3.10,<3.14`. Docs claim 3.9+ (DOCS_GUIDE), 3.10–3.12 (contributing/README), 3.12+ (comfyui, fp8 note). | Align all docs to the canonical `>=3.10,<3.14`. Where a page recommends a specific dev version (e.g. "we recommend 3.12 for development"), keep that but make the support range canonical. |
| 5.3 | Recipe ownership contradiction `[codex]` | `recipes/README.md` says "Add recipes for this repository under this in-repo `recipes/` directory." `docs/contributing/model/adding_omni_model.md:570` says "please add a model recipe to the [vllm-project/recipes] repository." | Decide the canonical location (the in-repo `recipes/` appears to be the active one based on the README's detail). Update `adding_omni_model.md` to point contributors to the in-repo `recipes/` dir, matching `recipes/README.md`. If the external repo is still a valid alternative, state both explicitly rather than contradicting. |

### 4.6 DOCS_GUIDE rewrite (Batch 3)

`docs/contributing/DOCS_GUIDE.md` `[both]` is stale: it documents a `docs/` structure (`architecture/`, `index.md`, `examples/`, `stylesheets/`) and a GitHub Pages deployment workflow that no longer exist. Actual structure uses `.nav.yml` + awesome-nav, `design/` (not `architecture/`), and Read the Docs (`.readthedocs.yml`, `fail_on_warning: true`).

**Rewrite to cover:**
- Actual directory structure (referencing the real subdirs: `getting_started/`, `serving/`, `user_guide/`, `design/`, `contributing/`, `configuration/`, `features/`, `community/`, `api/`, `cli/`).
- Navigation model: `.nav.yml` (awesome-nav plugin), per-subdir `.nav.yml`/`.nav.yaml` files, glob patterns, `exclude_docs` for `*.inc.md`.
- Build & deploy: `mkdocs serve`, `mkdocs build --strict`, Read the Docs config, the `fail_on_warning` gate.
- Hooks: what `generate_examples.py`, `generate_api_readme.py`, `generate_argparse.py` do and the **known caveat** that they rewrite tracked source (so contributors should not hand-edit generated files under `docs/user_guide/examples/`, `docs/api/README.md`, or CLI doc paths).
- API docs: `api-autonav` + `mkdocstrings`, cross-reference syntax.
- Snippet includes: `--8<-- "path"` syntax and `*.inc.md` convention.

### 4.7 Stray root files (Batch 1)

| # | File | What it is | Action |
|---|------|-----------|--------|
| 7.1 | `num_layers` `[mine]` | 0-byte file, untracked, not referenced anywhere. | Delete. |
| 7.2 | `rfc-2280-summary.md` `[mine]` | Untracked scratch file at repo root. | Delete (or move into `docs/discussion/` / `docs/design/` if it has lasting value — review content at implementation time). |
| 7.3 | `wan-streamer-explorer.html` `[mine]` | Untracked 24KB HTML scratch file at repo root. | Delete. |

All three are untracked (`git ls-files` returns empty for them) and not gitignored, so deletion is a plain `rm` with no git history impact.

## 5. Work breakdown & sequencing

Three batches, ordered by risk and dependency. Each batch is one PR.

### Batch 1 — Mechanical fixes (low risk, no content decisions)
- §4.1 nav correctness (items 1.1, 1.2, 1.3)
- §4.2 item 2.7 (delete stale `text_to_audio.md` orphan)
- §4.7 stray root files (items 7.1, 7.2, 7.3)

**Verify:** `mkdocs build --strict` passes (acknowledging it will dirty tracked files via hooks — that's expected and out of scope to fix). Grep confirms no remaining `test_markers.md` / `test_style.md` singular references and no duplicate `vae_parallel` nav line.

### Batch 2 — Orphan recovery + de-duplication (medium risk, per-file)
- §4.2 items 2.1–2.6 (wire 6 orphans into nav; relocate or delete discussion `.html`)
- §4.3 cross-links on 8 (+1 pipeline) design↔user-guide pairs
- §4.4 filename fixes (4.4a `vae_parallel` rename; 4.4b `metrics.md` → `log_stats.md` rename + hat-notes)

**Verify:** `mkdocs build --strict` passes. A follow-up grep confirms every design-doc path in §4.3 has a "See also" link and every user-guide path reciprocates. `find docs -name "metrics.md"` returns only the two Prometheus docs. Inbound links to renamed files are all updated (grep for old paths returns 0).

### Batch 3 — Truth corrections + DOCS_GUIDE (content risk, needs review)
- §4.5 truth corrections (FAQ, Python versions, recipe ownership)
- §4.6 DOCS_GUIDE rewrite

**Verify:** `mkdocs build --strict` passes. FAQ no longer contains "0.16.0" or "Not yet" for streaming. `grep -rn "python 3.9"` in `docs/` returns 0. Recipe guidance in `adding_omni_model.md` and `recipes/README.md` agrees. DOCS_GUIDE no longer references `architecture/`, `index.md`, or GitHub Pages.

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Hook rewrites `.nav.yml` on build, potentially dropping hand-added nav entries. | All orphan-recovery entries (§4.2) are outside the Examples section that the hook rewrites. Verified by reading `generate_examples.py:275-315` — it preserves non-example items. The `text_to_audio.md` deletion is inside the hook-managed area but safe because the hook won't regenerate a file whose source dir is gone. |
| Renaming `vae_parallelism.md` → `vae_parallel.md` and `metrics.md` → `log_stats.md` breaks inbound links. | Each rename PR includes a repo-wide grep-and-replace for the old path. `mkdocs build --strict` catches broken internal links. |
| Truth corrections (FAQ, Python, recipes) require judgment calls about canonical values. | Batch 3 is isolated so reviewers can focus on content. Canonical Python range is anchored to `pyproject.toml` (`>=3.10,<3.14`), not a judgment call. Recipe-location decision is the one genuine judgment call — flagged in §4.5 item 5.3. |
| Local checkout is 61 commits behind `origin/main`. | All findings were re-verified against `origin/main` via `git show`. Implementation must start from a fresh `origin/main` checkout. |
| `mkdocs build --strict` dirties the tree, making "clean build" verification ambiguous. | Accept as known condition (§3). Verification uses `--strict` for *error* detection, not for `git status` cleanliness. The mutation problem is tracked in §10. |

## 7. Testing & verification strategy

Per-batch verification commands (run from a fresh `origin/main` checkout):

```bash
# 1. Strict build — must exit 0 (warnings about generated files are expected)
mkdocs build --strict

# 2. Nav correctness (Batch 1)
grep -rn "test_markers\.md\|test_style\.md" docs/contributing/ci/.nav.yaml  # expect 0 (should be tests_*)
grep -c "design/feature/vae_parallel.md" docs/.nav.yml                       # expect 1

# 3. Orphan recovery (Batch 2) — every previously-orphaned file now in nav
for f in design/metrics design/feature/pipeline_parallel \
         user_guide/diffusion/parallelism/pipeline_parallel \
         design/qwen3_omni_tts_performance_optimization; do
  grep -c "$f" docs/.nav.yml  # expect >=1 for each
done

# 4. Cross-links (Batch 2) — bidirectional
# (manual: open each pair and confirm the "See also" block exists in both directions)

# 5. Renames (Batch 2)
find docs -name "metrics.md"            # expect design/metrics.md, usage/metrics.md only
find docs -name "vae_parallelism.md"    # expect 0
find docs -name "vae_parallel.md"       # expect 2 (design/feature + user_guide/diffusion/parallelism)

# 6. Truth corrections (Batch 3)
grep -n "0\.16\.0\|Not yet" docs/usage/faq.md          # expect 0
grep -rn "python 3\.9" docs/                            # expect 0
grep -n "vllm-project/recipes" docs/contributing/model/adding_omni_model.md  # expect consistency with recipes/README.md

# 7. DOCS_GUIDE (Batch 3)
grep -n "architecture/\|index\.md\|GitHub Pages" docs/contributing/DOCS_GUIDE.md  # expect 0
```

No new test infrastructure is added (that's out of scope per §2).

## 8. Rollback

Each batch is a separate PR. If a batch causes build failures or regressions:
- Batch 1: revert the PR — all changes are mechanical and independent.
- Batch 2: revert per-file if a specific cross-link or rename is problematic; the orphan-wiring and cross-links are independent of each other.
- Batch 3: revert per-page; truth corrections and DOCS_GUIDE are independent.

## 9. Open questions for implementation

These are flagged for the implementer (or a future planning session) and do not block this design:

1. **`mot_config.md` (§4.2 item 2.4):** publish in `user_guide/` or relocate to `design/` as internal reference? Decide based on content depth at implementation time.
2. **`discussion/world_action_model_analysis.md` (§4.2 item 2.6):** relocate to `design/` (preferred, avoids a new top-level nav section) or create a "Discussion" subsection? Preference stated in §4.2; confirm at implementation.
3. **`rfc-2280-summary.md` (§4.7 item 7.2):** delete outright, or does it have lasting value worth moving into `docs/`? Review content at implementation time.
4. **Recipe ownership (§4.5 item 5.3):** is the in-repo `recipes/` the sole canonical location, or is `vllm-project/recipes` still a valid alternative? The `recipes/README.md` treats in-repo as canonical; confirm with maintainers.

## 10. Explicitly deferred follow-ups

Tracked here so they are not lost, but **out of scope** for this design:

- **Deterministic generation (codex Phase 1):** stop MkDocs hooks from writing tracked source. Generate examples/API/CLI pages virtually or into gitignored output. Add explicit `generate` and `--check` modes. *Why deferred:* user chose "de-dupe only, accept mutation."
- **Docs CI gate (codex Phase 2):** a single `scripts/docs-check` entry point covering strict build, internal link/anchor checks, nav duplicate/orphan detection, generated-drift detection, and critical-command checks. *Why deferred:* depends on deterministic generation landing first.
- **IA reorganization (codex Phase 3):** task-first home page, grouping operational material, completing the design index. *Why deferred:* user did not select "restructure IA."
- **Golden paths, contributor doc split, governance, versioned docs (codex Phases 4–6):** *Why deferred:* beyond the two selected goals.

The hook-mutation problem is the most important deferred item: until it lands, nav and example-page fixes are correct as source but will be rewritten on every local `mkdocs build`. This design's fixes are scoped to survive the hook (see §6), but the underlying instability remains.

---

## Appendix: audit provenance

- **My audit:** surveyed all 286 `.md` files; used two Explore subagents to map orphans/broken-refs/duplicates and to compare 13 candidate duplicated pairs. Re-verified findings against `origin/main` via `git show` after discovering local was 61 commits behind.
- **Codex audit:** independently surveyed the docs; surfaced build-time mutation, truth drift (FAQ, Python, NPU, recipes), and the 17 open docs issues. All substantive codex claims were verified against the repo: hook write-operations confirmed at `generate_examples.py:315,350`, `generate_api_readme.py:278`, `generate_argparse.py:254`; FAQ staleness confirmed; Python drift confirmed against `pyproject.toml`; recipe contradiction confirmed between `recipes/README.md` and `adding_omni_model.md:570`.
- **Findings that hold at `origin/main`:** CI sub-nav singular/plural mismatch, `vae_parallel.md` duplicate nav entry, `design/metrics.md` orphan (0 nav refs), `pipeline_parallel` nav omission, stale `text_to_audio.md` orphan.
