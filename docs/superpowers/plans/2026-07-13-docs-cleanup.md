# Docs Cleanup & De-duplication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix broken/stale navigation, recover orphaned docs, cross-link duplicated design↔user-guide pairs, correct factual drift, and rewrite the stale DOCS_GUIDE — without restructuring the IA or fixing build-time mutation.

**Architecture:** Three sequential PRs (Batches 1–3) ordered by risk. Batch 1 is mechanical nav/file fixes. Batch 2 wires orphans into nav and adds bidirectional cross-links plus two filename renames. Batch 3 corrects factual content (FAQ, Python versions, recipe ownership) and rewrites DOCS_GUIDE. All work starts from a fresh `origin/main` checkout.

**Tech Stack:** MkDocs Material, awesome-nav (`.nav.yml`), Read the Docs (`fail_on_warning: true`), Markdown.

**Spec:** `docs/superpowers/specs/2026-07-13-docs-cleanup-design.md`

**Known condition (do NOT try to fix):** `mkdocs build` rewrites tracked files via hooks (`generate_examples.py`, `generate_api_readme.py`, `generate_argparse.py`). A clean build dirties the tree. This is expected. Verification uses `mkdocs build --strict` for *error* detection only — `git status` cleanliness after a build is not a success criterion. The mutation fix is out of scope (see spec §10).

---

## Pre-flight (before any task)

- [ ] **Step 0.1: Rebaseline to origin/main**

```bash
git fetch origin
git checkout main
git reset --hard origin/main
```

- [ ] **Step 0.2: Confirm baseline build works**

```bash
mkdocs build --strict 2>&1 | tail -5
```

Expected: exit 0 (build succeeds; tree will be dirty afterward — that's fine). If the baseline build already fails, stop and report — the plan assumes a green baseline.

- [ ] **Step 0.3: Create a working branch**

```bash
git checkout -b docs-cleanup-batch1
```

---

## Batch 1 — Mechanical fixes (one PR)

### Task 1: Fix CI sub-nav singular/plural mismatch

The file `docs/contributing/ci/.nav.yaml` references `test_markers.md` and `test_style.md`, but the files on disk are `tests_markers.md` and `tests_style.md` (plural).

**Files:**
- Modify: `docs/contributing/ci/.nav.yaml`

- [ ] **Step 1.1: Read the current content**

```bash
cat docs/contributing/ci/.nav.yaml
```

Expected output:
```yaml
nav:
  - CI_5levels.md
  - failures.md
  - test_guide.md
  - test_markers.md
  - test_style.md
```

- [ ] **Step 1.2: Fix the two broken references**

In `docs/contributing/ci/.nav.yaml`, replace:
```yaml
  - test_markers.md
  - test_style.md
```
with:
```yaml
  - tests_markers.md
  - tests_style.md
```

- [ ] **Step 1.3: Verify the files referenced now exist**

```bash
ls docs/contributing/ci/tests_markers.md docs/contributing/ci/tests_style.md
```

Expected: both files listed (no "No such file" error).

- [ ] **Step 1.4: Commit**

```bash
git add docs/contributing/ci/.nav.yaml
git commit -m "docs: fix CI sub-nav singular/plural filename mismatch

.nav.yaml referenced test_markers.md and test_style.md (singular), but the
files on disk are tests_markers.md and tests_style.md (plural). Two nav
entries were broken."
```

### Task 2: Remove duplicate vae_parallel nav entry

`docs/.nav.yml` lists `design/feature/vae_parallel.md` twice in the Feature Design section (currently lines 115 and 120).

**Files:**
- Modify: `docs/.nav.yml`

- [ ] **Step 2.1: Confirm the duplicate**

```bash
grep -n "design/feature/vae_parallel.md" docs/.nav.yml
```

Expected: two lines (e.g. `115:...` and `120:...`).

- [ ] **Step 2.2: Remove the second occurrence**

In `docs/.nav.yml`, find this block inside the `Feature Design:` section:
```yaml
      - design/feature/vae_parallel.md
      - design/feature/hsdp.md
      - design/feature/cache_dit.md
      - design/feature/teacache.md
      - design/feature/async_chunk.md
      - design/feature/vae_parallel.md
      - design/feature/diffusion_step_execution.md
```
Replace with (removing the second `vae_parallel.md` line):
```yaml
      - design/feature/vae_parallel.md
      - design/feature/hsdp.md
      - design/feature/cache_dit.md
      - design/feature/teacache.md
      - design/feature/async_chunk.md
      - design/feature/diffusion_step_execution.md
```

- [ ] **Step 2.3: Verify only one occurrence remains**

```bash
grep -c "design/feature/vae_parallel.md" docs/.nav.yml
```

Expected: `1`

- [ ] **Step 2.4: Commit**

```bash
git add docs/.nav.yml
git commit -m "docs: remove duplicate vae_parallel nav entry

design/feature/vae_parallel.md was listed twice in the Feature Design
section of .nav.yml."
```

### Task 3: Resolve contributing/* glob-vs-explicit collisions

`contributing/README.md` and `contributing/DOCS_GUIDE.md` are each reachable twice — once via an explicit nav entry and once via the `glob: contributing/*`. The glob expands to all four direct children (`README.md`, `DOCS_GUIDE.md`, `metrics.md`, `profiling.md`), colliding with the two explicit entries.

**Files:**
- Modify: `docs/.nav.yml`

- [ ] **Step 3.1: Read the current contributing nav block**

```bash
sed -n '/  - General:/,/  - Model Implementation:/p' docs/.nav.yml
```

Expected:
```yaml
  - General:
    - contributing/README.md
    - glob: contributing/*
      flatten_single_child_sections: true
```

- [ ] **Step 3.2: Exclude the explicitly-referenced files from the glob**

In `docs/.nav.yml`, replace:
```yaml
  - General:
    - contributing/README.md
    - glob: contributing/*
      flatten_single_child_sections: true
```
with:
```yaml
  - General:
    - contributing/README.md
    - glob: contributing/*
      flatten_single_child_sections: true
      exclude:
        - README.md
        - DOCS_GUIDE.md
```

This keeps the explicit `README.md` entry (with its title) and the `DOCS_GUIDE.md` entry under "Docs Guide" later in the file, while letting the glob pick up `metrics.md` and `profiling.md` only.

- [ ] **Step 3.3: Verify the awesome-nav exclude syntax is valid**

Check the awesome-nav docs for `exclude` under `glob`. If `exclude` is not a supported key for awesome-nav globs, use this alternative instead — replace the block with:
```yaml
  - General:
    - contributing/README.md
    - contributing/metrics.md
    - contributing/profiling.md
```
(i.e. drop the glob entirely and list the two remaining files explicitly.)

- [ ] **Step 3.4: Commit**

```bash
git add docs/.nav.yml
git commit -m "docs: resolve contributing/* glob-vs-explicit nav collisions

contributing/README.md and contributing/DOCS_GUIDE.md were each reachable
twice (explicit entry + glob expansion). Exclude them from the glob."
```

### Task 4: Delete stale text_to_audio online-serving orphan

`docs/user_guide/examples/online_serving/text_to_audio.md` is a true stale orphan: at `origin/main` it points to `examples/online_serving/stable_audio`, which does not exist. The `generate_examples.py` hook cannot regenerate it because the source dir is gone.

**Files:**
- Delete: `docs/user_guide/examples/online_serving/text_to_audio.md`

- [ ] **Step 4.1: Confirm the source dir is gone**

```bash
ls examples/online_serving/text_to_audio 2>&1
ls examples/online_serving/stable_audio 2>&1
```

Expected: both report "No such file or directory" (confirming the hook won't regenerate the page).

- [ ] **Step 4.2: Confirm the .md file is tracked**

```bash
git ls-files docs/user_guide/examples/online_serving/text_to_audio.md
```

Expected: the path is printed (file is tracked).

- [ ] **Step 4.3: Delete the file**

```bash
git rm docs/user_guide/examples/online_serving/text_to_audio.md
```

- [ ] **Step 4.4: Commit**

```bash
git commit -m "docs: delete stale text_to_audio online-serving example page

The example source examples/online_serving/stable_audio no longer exists, so
the generate_examples.py hook cannot regenerate this page. It pointed at a
deleted directory. The offline_inference/text_to_audio example is unrelated
and remains."
```

### Task 5: Remove stray root-level files

Three untracked files at the repo root are not part of the project: `num_layers` (0-byte), `rfc-2280-summary.md` (scratch summary), `wan-streamer-explorer.html` (scratch HTML).

**Files:**
- Delete: `num_layers`, `rfc-2280-summary.md`, `wan-streamer-explorer.html`

- [ ] **Step 5.1: Confirm all three are untracked**

```bash
git ls-files num_layers rfc-2280-summary.md wan-streamer-explorer.html
```

Expected: empty output (none are tracked). If any IS tracked, stop — `git rm` it instead and note this in the commit.

- [ ] **Step 5.2: Review rfc-2280-summary.md content briefly**

```bash
head -20 rfc-2280-summary.md
```

Decision point: if the content has lasting value relevant to vllm-omni, move it to `docs/design/rfc-2280-summary.md` instead of deleting. If it's scratch/personal notes, delete it. Default action: delete.

- [ ] **Step 5.3: Delete the files**

```bash
rm num_layers rfc-2280-summary.md wan-streamer-explorer.html
```

- [ ] **Step 5.4: Verify they're gone**

```bash
ls num_layers rfc-2280-summary.md wan-streamer-explorer.html 2>&1
```

Expected: "No such file or directory" for all three.

- [ ] **Step 5.5: Commit**

Since these are untracked, there's nothing to commit to git — they were never in the repo. The deletion is purely local filesystem cleanup. No commit needed. (If Step 5.1 found any were tracked, commit the `git rm` here instead.)

### Task 6: Batch 1 verification

- [ ] **Step 6.1: Strict build passes**

```bash
mkdocs build --strict 2>&1 | tail -10
```

Expected: exit 0. (Tree will be dirty from hooks — that's fine, not a failure signal. Discard hook-generated changes with `git checkout -- docs/.nav.yml docs/user_guide/examples/ docs/api/README.md` if you want a clean tree before the next step.)

- [ ] **Step 6.2: Nav correctness checks**

```bash
# CI sub-nav now references the plural filenames
grep -E "test_markers\.md|test_style\.md" docs/contributing/ci/.nav.yaml
# Expected: no output (the singular forms are gone)

grep -E "tests_markers\.md|tests_style\.md" docs/contributing/ci/.nav.yaml
# Expected: both plural filenames present

# vae_parallel listed exactly once
grep -c "design/feature/vae_parallel.md" docs/.nav.yml
# Expected: 1
```

- [ ] **Step 6.3: Push the branch and open PR 1**

```bash
git push -u origin docs-cleanup-batch1
gh pr create --title "docs: fix broken nav, remove stale orphan and stray files" \
  --body "Batch 1 of the docs cleanup plan (see docs/superpowers/specs/2026-07-13-docs-cleanup-design.md).

- Fix CI sub-nav singular/plural mismatch (test_markers → tests_markers, test_style → tests_style)
- Remove duplicate vae_parallel nav entry
- Resolve contributing/* glob-vs-explicit collisions
- Delete stale text_to_audio online-serving orphan (source dir gone)
- Remove stray root files (num_layers, rfc-2280-summary.md, wan-streamer-explorer.html)

Verification: mkdocs build --strict passes."
```

---

## Batch 2 — Orphan recovery + de-duplication (one PR)

Start a new branch from origin/main after Batch 1 merges.

- [ ] **Step 7.0: Rebaseline and branch**

```bash
git fetch origin
git checkout main
git reset --hard origin/main
git checkout -b docs-cleanup-batch2
```

### Task 8: Wire pipeline_parallel design doc into nav

`docs/design/feature/pipeline_parallel.md` exists but every sibling design doc is listed in `.nav.yml` except this one.

**Files:**
- Modify: `docs/.nav.yml`

- [ ] **Step 8.1: Add pipeline_parallel to the Feature Design section**

In `docs/.nav.yml`, inside the `Feature Design:` section, find:
```yaml
      - design/feature/diffusion_continuous_batching.md
```
That is the last entry before `Module Design:`. Add `pipeline_parallel.md` before it, keeping alphabetical-ish grouping with the other parallelism entries. Insert after `async_chunk.md`:
```yaml
      - design/feature/async_chunk.md
      - design/feature/pipeline_parallel.md
      - design/feature/diffusion_step_execution.md
```

- [ ] **Step 8.2: Verify**

```bash
grep -c "design/feature/pipeline_parallel.md" docs/.nav.yml
```

Expected: `1`

- [ ] **Step 8.3: Commit**

```bash
git add docs/.nav.yml
git commit -m "docs: wire pipeline_parallel design doc into nav"
```

### Task 9: Wire pipeline_parallel user guide into nav

`docs/user_guide/diffusion/parallelism/pipeline_parallel.md` exists but the parallelism nav lists overview/cfg/expert/hsdp/sequence/tensor/vae and omits pipeline.

**Files:**
- Modify: `docs/.nav.yml`

- [ ] **Step 9.1: Add pipeline_parallelism to the Parallelism subsection**

In `docs/.nav.yml`, inside the `Parallelism:` subsection (under `Diffusion Features:`), find:
```yaml
      - Parallelism:
        - Overview: user_guide/diffusion/parallelism/overview.md
        - CFG Parallel: user_guide/diffusion/parallelism/cfg_parallel.md
        - Expert Parallel: user_guide/diffusion/parallelism/expert_parallel.md
        - Hybrid Sharded Data Parallel: user_guide/diffusion/parallelism/hsdp.md
        - Sequence Parallel: user_guide/diffusion/parallelism/sequence_parallel.md
        - Tensor Parallel: user_guide/diffusion/parallelism/tensor_parallel.md
        - VAE Parallelism: user_guide/diffusion/parallelism/vae_parallelism.md
```
Replace the last line and add pipeline before VAE:
```yaml
        - Tensor Parallel: user_guide/diffusion/parallelism/tensor_parallel.md
        - Pipeline Parallel: user_guide/diffusion/parallelism/pipeline_parallel.md
        - VAE Parallelism: user_guide/diffusion/parallelism/vae_parallelism.md
```

- [ ] **Step 9.2: Verify**

```bash
grep -c "user_guide/diffusion/parallelism/pipeline_parallel.md" docs/.nav.yml
```

Expected: `1`

- [ ] **Step 9.3: Commit**

```bash
git add docs/.nav.yml
git commit -m "docs: wire pipeline_parallel user guide into nav"
```

### Task 10: Wire design/metrics.md into nav

`docs/design/metrics.md` (Prometheus metrics design) exists but is unreferenced.

**Files:**
- Modify: `docs/.nav.yml`

- [ ] **Step 10.1: Add metrics.md to the Design Documents top level**

In `docs/.nav.yml`, in the `Design Documents:` section, find:
```yaml
    - design/index.md
    - design/architecture_overview.md
    - Feature Design:
```
Insert `design/metrics.md` before `Feature Design:`:
```yaml
    - design/index.md
    - design/architecture_overview.md
    - design/metrics.md
    - Feature Design:
```

- [ ] **Step 10.2: Verify**

```bash
grep -c "design/metrics.md" docs/.nav.yml
```

Expected: `1`

- [ ] **Step 10.3: Commit**

```bash
git add docs/.nav.yml
git commit -m "docs: wire Prometheus metrics design doc into nav"
```

### Task 11: Wire qwen3_omni_tts_performance_optimization.md into nav

`docs/design/qwen3_omni_tts_performance_optimization.md` exists but is unreferenced.

**Files:**
- Modify: `docs/.nav.yml`

- [ ] **Step 11.1: Add to Design Documents top level**

In `docs/.nav.yml`, in the `Design Documents:` section, after the `design/metrics.md` line added in Task 10, add:
```yaml
    - design/metrics.md
    - design/qwen3_omni_tts_performance_optimization.md
    - Feature Design:
```

- [ ] **Step 11.2: Verify**

```bash
grep -c "design/qwen3_omni_tts_performance_optimization.md" docs/.nav.yml
```

Expected: `1`

- [ ] **Step 11.3: Commit**

```bash
git add docs/.nav.yml
git commit -m "docs: wire Qwen3-Omni TTS performance optimization doc into nav"
```

### Task 12: Handle mot_config.md orphan

`docs/user_guide/diffusion/mot_config.md` has no H1 heading and is unreferenced. It documents auto-tuned Triton kernel configs for MoT.

**Files:**
- Modify: `docs/user_guide/diffusion/mot_config.md`
- Modify: `docs/.nav.yml`

- [ ] **Step 12.1: Read the current content**

```bash
cat docs/user_guide/diffusion/mot_config.md
```

- [ ] **Step 12.2: Add an H1 heading and intro**

At the top of `docs/user_guide/diffusion/mot_config.md`, before the existing first line (`This directory contains auto-tuned Triton kernel configurations...`), add:
```markdown
# MoT Kernel Configs

This page documents the auto-tuned Triton kernel configurations for the
MoT (Mixture-of-Tokens) GEMM and RMSNorm operators used by BAGEL and other
MoT-architecture diffusion models.

```

- [ ] **Step 12.3: Wire into nav under Diffusion Features**

In `docs/.nav.yml`, in the `Diffusion Features:` section (under `Features:`), find:
```yaml
      - Step Execution: user_guide/diffusion/step_execution.md
```
Add after it:
```yaml
      - Step Execution: user_guide/diffusion/step_execution.md
      - MoT Kernel Configs: user_guide/diffusion/mot_config.md
```

- [ ] **Step 12.4: Verify**

```bash
head -5 docs/user_guide/diffusion/mot_config.md
grep -c "user_guide/diffusion/mot_config.md" docs/.nav.yml
```

Expected: H1 present; nav count `1`.

- [ ] **Step 12.5: Commit**

```bash
git add docs/user_guide/diffusion/mot_config.md docs/.nav.yml
git commit -m "docs: add H1 and wire MoT kernel configs doc into nav"
```

### Task 13: Relocate discussion/ into design/ and delete html twin

`docs/discussion/world_action_model_analysis.md` is unreferenced (the entire `discussion/` dir is). It has a stale `.html` twin. Decision (per spec §4.2 item 2.6): relocate the `.md` into `design/` and delete the `.html`.

**Files:**
- Move: `docs/discussion/world_action_model_analysis.md` → `docs/design/world_action_model_analysis.md`
- Delete: `docs/discussion/world_action_model_analysis.html`
- Modify: `docs/.nav.yml`

- [ ] **Step 13.1: Move the markdown file**

```bash
git mv docs/discussion/world_action_model_analysis.md docs/design/world_action_model_analysis.md
```

- [ ] **Step 13.2: Delete the html twin**

```bash
git rm docs/discussion/world_action_model_analysis.html
```

- [ ] **Step 13.3: Remove the now-empty discussion directory**

```bash
rmdir docs/discussion 2>/dev/null || true
```

- [ ] **Step 13.4: Wire into nav under Design Documents**

In `docs/.nav.yml`, in the `Design Documents:` section, after the `design/qwen3_omni_tts_performance_optimization.md` line added in Task 11, add:
```yaml
    - design/qwen3_omni_tts_performance_optimization.md
    - design/world_action_model_analysis.md
    - Feature Design:
```

- [ ] **Step 13.5: Verify**

```bash
ls docs/design/world_action_model_analysis.md
ls docs/discussion 2>&1
grep -c "design/world_action_model_analysis.md" docs/.nav.yml
```

Expected: file exists; `docs/discussion` gone; nav count `1`.

- [ ] **Step 13.6: Commit**

```bash
git add -A
git commit -m "docs: relocate world_action_model_analysis into design/, delete html twin

The discussion/ directory was unreferenced in nav. Move the .md into design/
and wire it in. Delete the stale .html twin (.md is the source of truth)."
```

### Task 14: Rename vae_parallelism.md → vae_parallel.md

The user-guide file is `vae_parallelism.md` while its design counterpart and all sibling parallelism files use the `*_parallel.md` form.

**Files:**
- Rename: `docs/user_guide/diffusion/parallelism/vae_parallelism.md` → `docs/user_guide/diffusion/parallelism/vae_parallel.md`
- Modify: `docs/.nav.yml`

- [ ] **Step 14.1: Rename the file**

```bash
git mv docs/user_guide/diffusion/parallelism/vae_parallelism.md docs/user_guide/diffusion/parallelism/vae_parallel.md
```

- [ ] **Step 14.2: Update the nav entry**

In `docs/.nav.yml`, replace:
```yaml
        - VAE Parallelism: user_guide/diffusion/parallelism/vae_parallelism.md
```
with:
```yaml
        - VAE Parallelism: user_guide/diffusion/parallelism/vae_parallel.md
```

- [ ] **Step 14.3: Find and update all inbound links**

```bash
grep -rn "vae_parallelism" docs/ --include="*.md"
```

For each result, replace `vae_parallelism` with `vae_parallel` in the link path (keep display text as "VAE Parallelism" where it appears as link text). Common locations: design/feature/vae_parallel.md (the cross-link you'll add in Task 16), other parallelism docs that link to it.

- [ ] **Step 14.4: Verify no stale references remain**

```bash
grep -rn "vae_parallelism" docs/ --include="*.md"
```

Expected: no output (all references updated). Note: the word "parallelism" may appear in prose — only fix path references like `vae_parallelism.md`.

- [ ] **Step 14.5: Commit**

```bash
git add -A
git commit -m "docs: rename vae_parallelism.md to vae_parallel.md

Align with the design counterpart (design/feature/vae_parallel.md) and the
naming convention of the other parallelism files (cfg_parallel, tensor_parallel,
etc.). Update nav and inbound links."
```

### Task 15: Rename contributing/metrics.md → log_stats.md + hat-notes

`contributing/metrics.md` is about `--log-stats` console output, NOT Prometheus metrics. Its identical filename is confused with `design/metrics.md` and `usage/metrics.md` (the Prometheus pair). Rename it and add disambiguating hat-notes.

**Files:**
- Rename: `docs/contributing/metrics.md` → `docs/contributing/log_stats.md`
- Modify: `docs/design/metrics.md` (add hat-note)
- Modify: `docs/usage/metrics.md` (add hat-note)
- Modify: `docs/contributing/log_stats.md` (add hat-note + update heading)
- Modify: `docs/.nav.yml` (glob picks up the new name automatically — verify)

- [ ] **Step 15.1: Rename the file**

```bash
git mv docs/contributing/metrics.md docs/contributing/log_stats.md
```

- [ ] **Step 15.2: Update the H1 in the renamed file**

In `docs/contributing/log_stats.md`, the current first heading is `# Metrics` (with a leading blank line). Replace:
```markdown

# Metrics
```
with:
```markdown
# Console Log Stats (`--log-stats`)

> ℹ️ **Not the Prometheus metrics page.** This page documents the `--log-stats`
> console output (Overall Summary, RequestE2EStats, etc.). For the Prometheus
> `/metrics` endpoint, see [Production Metrics](../usage/metrics.md) and the
> [Prometheus Metrics Design](../design/metrics.md).
```

- [ ] **Step 15.3: Add hat-note to design/metrics.md**

In `docs/design/metrics.md`, after the H1 line `# Prometheus Metrics Design`, add a hat-note block. Find the current top:
```markdown
# Prometheus Metrics Design

This document describes how vLLM-Omni exposes Prometheus metrics...
```
Insert between the H1 and the first paragraph:
```markdown
# Prometheus Metrics Design

> ℹ️ **Design doc.** For the end-user Prometheus metrics reference, see
> [Production Metrics](../../usage/metrics.md). For the `--log-stats` console
> output (a different feature), see [Console Log Stats](../../contributing/log_stats.md).

This document describes how vLLM-Omni exposes Prometheus metrics...
```

- [ ] **Step 15.4: Add hat-note to usage/metrics.md**

In `docs/usage/metrics.md`, after the H1 line `# Production Metrics`, add a hat-note. Find:
```markdown
# Production Metrics

vLLM-Omni exposes Prometheus metrics via the `/metrics` endpoint...
```
Insert:
```markdown
# Production Metrics

> ℹ️ **User guide.** For the design rationale, see
> [Prometheus Metrics Design](../design/metrics.md). For the `--log-stats`
> console output (a different feature), see [Console Log Stats](../contributing/log_stats.md).

vLLM-Omni exposes Prometheus metrics via the `/metrics` endpoint...
```

- [ ] **Step 15.5: Update any inbound links to contributing/metrics.md**

```bash
grep -rn "contributing/metrics" docs/ --include="*.md"
```

Replace each `contributing/metrics` with `contributing/log_stats` in link paths.

- [ ] **Step 15.6: Verify the glob picks up the new name**

The `glob: contributing/*` in `.nav.yml` should auto-include `log_stats.md`. Verify with a build (Task 22). For now:
```bash
grep -rn "contributing/metrics" docs/ --include="*.md"
```
Expected: no output (all references updated to `log_stats`).

- [ ] **Step 15.7: Commit**

```bash
git add -A
git commit -m "docs: rename contributing/metrics.md to log_stats.md, add hat-notes

contributing/metrics.md documented the --log-stats console output, not the
Prometheus /metrics endpoint — but its filename collided with the Prometheus
pair (design/metrics.md, usage/metrics.md). Rename to log_stats.md and add
disambiguating hat-notes on all three."
```

### Task 16: Add bidirectional cross-links to design↔user-guide pairs

Add "See also" cross-links to the 9 design-doc ↔ user-guide pairs that lack them. Mirror the existing pattern from `step_execution` (design doc links to user guide; user guide links to design doc).

The pattern for a **design doc** (insert immediately after the H1 line, before the first paragraph):
```markdown
> 🔧 **Design doc.** For end-user enablement, configuration, and Quick Start,
> see [`<User Guide Title>`](<relative/path/to/user_guide.md>).
```

The pattern for a **user guide** (insert immediately after the H1 line, before the next content):
```markdown
> 📖 **End-user guide.** For implementation details, internal APIs, and architecture,
> see [`<Design Doc Title>`](<relative/path/to/design.md>).
```

**Files (9 pairs = 18 files):**

| Design doc | User guide |
|---|---|
| `docs/design/feature/cfg_parallel.md` | `docs/user_guide/diffusion/parallelism/cfg_parallel.md` |
| `docs/design/feature/expert_parallel.md` | `docs/user_guide/diffusion/parallelism/expert_parallel.md` |
| `docs/design/feature/hsdp.md` | `docs/user_guide/diffusion/parallelism/hsdp.md` |
| `docs/design/feature/sequence_parallel.md` | `docs/user_guide/diffusion/parallelism/sequence_parallel.md` |
| `docs/design/feature/tensor_parallel.md` | `docs/user_guide/diffusion/parallelism/tensor_parallel.md` |
| `docs/design/feature/vae_parallel.md` | `docs/user_guide/diffusion/parallelism/vae_parallel.md` |
| `docs/design/feature/cache_dit.md` | `docs/user_guide/diffusion/cache_acceleration/cache_dit.md` |
| `docs/design/feature/teacache.md` | `docs/user_guide/diffusion/cache_acceleration/teacache.md` |
| `docs/design/feature/pipeline_parallel.md` | `docs/user_guide/diffusion/parallelism/pipeline_parallel.md` |

- [ ] **Step 16.1: Add cross-link to design/feature/cfg_parallel.md**

In `docs/design/feature/cfg_parallel.md`, after the H1 `# CFG-Parallel`, insert:
```markdown

> 🔧 **Design doc.** For end-user enablement, configuration, and Quick Start,
> see [CFG-Parallel Guide](../../user_guide/diffusion/parallelism/cfg_parallel.md).
```

- [ ] **Step 16.2: Add cross-link to user_guide/.../cfg_parallel.md**

In `docs/user_guide/diffusion/parallelism/cfg_parallel.md`, after the H1 `# CFG-Parallel Guide`, insert:
```markdown

> 📖 **End-user guide.** For implementation details, internal APIs, and architecture,
> see [CFG-Parallel](../../../design/feature/cfg_parallel.md).
```

- [ ] **Step 16.3: Add cross-link to design/feature/expert_parallel.md**

In `docs/design/feature/expert_parallel.md`, after `# Expert Parallel`, insert:
```markdown

> 🔧 **Design doc.** For end-user enablement, configuration, and Quick Start,
> see [Expert Parallelism Guide](../../user_guide/diffusion/parallelism/expert_parallel.md).
```

- [ ] **Step 16.4: Add cross-link to user_guide/.../expert_parallel.md**

In `docs/user_guide/diffusion/parallelism/expert_parallel.md`, after `# Expert Parallelism Guide`, insert:
```markdown

> 📖 **End-user guide.** For implementation details, internal APIs, and architecture,
> see [Expert Parallel](../../../design/feature/expert_parallel.md).
```

- [ ] **Step 16.5: Add cross-link to design/feature/hsdp.md**

In `docs/design/feature/hsdp.md`, after `# HSDP`, insert:
```markdown

> 🔧 **Design doc.** For end-user enablement, configuration, and Quick Start,
> see [HSDP Guide](../../user_guide/diffusion/parallelism/hsdp.md).
```

- [ ] **Step 16.6: Add cross-link to user_guide/.../hsdp.md**

In `docs/user_guide/diffusion/parallelism/hsdp.md`, after `# HSDP Guide`, insert:
```markdown

> 📖 **End-user guide.** For implementation details, internal APIs, and architecture,
> see [HSDP](../../../design/feature/hsdp.md).
```

- [ ] **Step 16.7: Add cross-link to design/feature/sequence_parallel.md**

In `docs/design/feature/sequence_parallel.md`, after `# Sequence Parallel`, insert:
```markdown

> 🔧 **Design doc.** For end-user enablement, configuration, and Quick Start,
> see [Sequence Parallelism Guide](../../user_guide/diffusion/parallelism/sequence_parallel.md).
```

- [ ] **Step 16.8: Add cross-link to user_guide/.../sequence_parallel.md**

In `docs/user_guide/diffusion/parallelism/sequence_parallel.md`, after `# Sequence Parallelism Guide`, insert:
```markdown

> 📖 **End-user guide.** For implementation details, internal APIs, and architecture,
> see [Sequence Parallel](../../../design/feature/sequence_parallel.md).
```

- [ ] **Step 16.9: Add cross-link to design/feature/tensor_parallel.md**

In `docs/design/feature/tensor_parallel.md`, after `# Tensor Parallel`, insert:
```markdown

> 🔧 **Design doc.** For end-user enablement, configuration, and Quick Start,
> see [Tensor Parallelism Guide](../../user_guide/diffusion/parallelism/tensor_parallel.md).
```

- [ ] **Step 16.10: Add cross-link to user_guide/.../tensor_parallel.md**

In `docs/user_guide/diffusion/parallelism/tensor_parallel.md`, after `# Tensor Parallelism Guide`, insert:
```markdown

> 📖 **End-user guide.** For implementation details, internal APIs, and architecture,
> see [Tensor Parallel](../../../design/feature/tensor_parallel.md).
```

- [ ] **Step 16.11: Add cross-link to design/feature/vae_parallel.md**

In `docs/design/feature/vae_parallel.md`, after `# VAE Patch Parallelism`, insert:
```markdown

> 🔧 **Design doc.** For end-user enablement, configuration, and Quick Start,
> see [VAE Parallelism Guide](../../user_guide/diffusion/parallelism/vae_parallel.md).
```

- [ ] **Step 16.12: Add cross-link to user_guide/.../vae_parallel.md**

In `docs/user_guide/diffusion/parallelism/vae_parallel.md` (renamed in Task 14), after `# VAE Parallelism Guide`, insert:
```markdown

> 📖 **End-user guide.** For implementation details, internal APIs, and architecture,
> see [VAE Patch Parallelism](../../../design/feature/vae_parallel.md).
```

- [ ] **Step 16.13: Add cross-link to design/feature/cache_dit.md**

In `docs/design/feature/cache_dit.md`, after `# Cache-DiT`, insert:
```markdown

> 🔧 **Design doc.** For end-user enablement, configuration, and Quick Start,
> see [Cache-DiT Guide](../../user_guide/diffusion/cache_acceleration/cache_dit.md).
```

- [ ] **Step 16.14: Add cross-link to user_guide/.../cache_dit.md**

In `docs/user_guide/diffusion/cache_acceleration/cache_dit.md`, after `# Cache-DiT Guide`, insert:
```markdown

> 📖 **End-user guide.** For implementation details, internal APIs, and architecture,
> see [Cache-DiT](../../../design/feature/cache_dit.md).
```

- [ ] **Step 16.15: Add cross-link to design/feature/teacache.md**

In `docs/design/feature/teacache.md`, after `# TeaCache`, insert:
```markdown

> 🔧 **Design doc.** For end-user enablement, configuration, and Quick Start,
> see [TeaCache Guide](../../user_guide/diffusion/cache_acceleration/teacache.md).
```

- [ ] **Step 16.16: Add cross-link to user_guide/.../teacache.md**

In `docs/user_guide/diffusion/cache_acceleration/teacache.md`, after `# TeaCache Guide`, insert:
```markdown

> 📖 **End-user guide.** For implementation details, internal APIs, and architecture,
> see [TeaCache](../../../design/feature/teacache.md).
```

- [ ] **Step 16.17: Add cross-link to design/feature/pipeline_parallel.md**

In `docs/design/feature/pipeline_parallel.md`, after `# Pipeline Parallel`, insert:
```markdown

> 🔧 **Design doc.** For end-user enablement, configuration, and Quick Start,
> see [Pipeline Parallelism Guide](../../user_guide/diffusion/parallelism/pipeline_parallel.md).
```

- [ ] **Step 16.18: Add cross-link to user_guide/.../pipeline_parallel.md**

In `docs/user_guide/diffusion/parallelism/pipeline_parallel.md`, after `# Pipeline Parallelism Guide`, insert:
```markdown

> 📖 **End-user guide.** For implementation details, internal APIs, and architecture,
> see [Pipeline Parallel](../../../design/feature/pipeline_parallel.md).
```

- [ ] **Step 16.19: Commit all cross-links**

```bash
git add docs/design/feature/ docs/user_guide/diffusion/
git commit -m "docs: add bidirectional cross-links to design↔user-guide pairs

9 feature pairs (cfg_parallel, expert_parallel, hsdp, sequence_parallel,
tensor_parallel, vae_parallel, cache_dit, teacache, pipeline_parallel) now
have 'See also' pointers in both directions, mirroring the pattern already
used by step_execution and request_batching."
```

### Task 17: Batch 2 verification

- [ ] **Step 17.1: Strict build passes**

```bash
git checkout -- docs/.nav.yml docs/user_guide/examples/ docs/api/README.md 2>/dev/null
mkdocs build --strict 2>&1 | tail -10
```

Expected: exit 0. (Discard hook-generated changes first so the build tests your source, not hook output. If `mkdocs build --strict` fails on a broken link, the relative paths in the cross-links are the likely culprit — verify path depth.)

- [ ] **Step 17.2: Orphan recovery checks**

```bash
for f in design/metrics design/feature/pipeline_parallel \
         user_guide/diffusion/parallelism/pipeline_parallel \
         design/qwen3_omni_tts_performance_optimization \
         design/world_action_model_analysis \
         user_guide/diffusion/mot_config; do
  echo -n "$f: "; grep -c "$f" docs/.nav.yml
done
```

Expected: each prints `1`.

- [ ] **Step 17.3: Rename checks**

```bash
find docs -name "vae_parallelism.md"
# Expected: no output

find docs -name "vae_parallel.md"
# Expected: docs/design/feature/vae_parallel.md and docs/user_guide/diffusion/parallelism/vae_parallel.md

find docs -name "metrics.md"
# Expected: docs/design/metrics.md and docs/usage/metrics.md only (NOT contributing/)

find docs -name "log_stats.md"
# Expected: docs/contributing/log_stats.md

ls docs/discussion 2>&1
# Expected: "No such file or directory"
```

- [ ] **Step 17.4: Cross-link spot check**

```bash
# Each design doc should have a cross-link to its user guide
grep -l "End-user guide" docs/design/feature/cfg_parallel.md docs/design/feature/expert_parallel.md docs/design/feature/hsdp.md docs/design/feature/sequence_parallel.md docs/design/feature/tensor_parallel.md docs/design/feature/vae_parallel.md docs/design/feature/cache_dit.md docs/design/feature/teacache.md docs/design/feature/pipeline_parallel.md
# Expected: all 9 files listed

# Each user guide should have a cross-link to its design doc
grep -l "Design doc" docs/user_guide/diffusion/parallelism/cfg_parallel.md docs/user_guide/diffusion/parallelism/expert_parallel.md docs/user_guide/diffusion/parallelism/hsdp.md docs/user_guide/diffusion/parallelism/sequence_parallel.md docs/user_guide/diffusion/parallelism/tensor_parallel.md docs/user_guide/diffusion/parallelism/vae_parallel.md docs/user_guide/diffusion/cache_acceleration/cache_dit.md docs/user_guide/diffusion/cache_acceleration/teacache.md docs/user_guide/diffusion/parallelism/pipeline_parallel.md
# Expected: all 9 files listed
```

- [ ] **Step 17.5: Push and open PR 2**

```bash
git push -u origin docs-cleanup-batch2
gh pr create --title "docs: wire orphans into nav, cross-link design↔user-guide pairs, fix filenames" \
  --body "Batch 2 of the docs cleanup plan (see docs/superpowers/specs/2026-07-13-docs-cleanup-design.md).

- Wire 6 orphaned docs into nav (design/metrics, pipeline_parallel design+user guide, qwen3_omni_tts_perf, mot_config, world_action_model_analysis)
- Relocate discussion/ into design/, delete stale .html twin
- Add bidirectional 'See also' cross-links to 9 design↔user-guide pairs
- Rename vae_parallelism.md → vae_parallel.md (align with design counterpart)
- Rename contributing/metrics.md → log_stats.md (disambiguate from Prometheus pair), add hat-notes

Verification: mkdocs build --strict passes; all cross-links resolve."
```

---

## Batch 3 — Truth corrections + DOCS_GUIDE rewrite (one PR)

- [ ] **Step 18.0: Rebaseline and branch**

```bash
git fetch origin
git checkout main
git reset --hard origin/main
git checkout -b docs-cleanup-batch3
```

### Task 19: Fix stale FAQ entries

`docs/usage/faq.md` says quantization is "planned to introduce ... in version 0.16.0" and streaming is "Not yet." Both ship today.

**Files:**
- Modify: `docs/usage/faq.md`

- [ ] **Step 19.1: Rewrite the quantization Q&A**

In `docs/usage/faq.md`, replace:
```markdown
> Q: Does vLLM-Omni support AWQ or any other quantization?

A: We plan to introduce GGUF FP8 prequantized models and online FP8 quantization in version 0.16.0. Support for other quantization types will follow in future releases. For details, please see our [Q1 quantization roadmap](https://github.com/vllm-project/vllm-omni/issues/1057).
```
with:
```markdown
> Q: Does vLLM-Omni support AWQ or any other quantization?

A: Yes. vLLM-Omni supports a range of quantization methods including FP8 (W8A8), Int8 (W8A8), MXFP8, MXFP4 (W4A4), GGUF, AutoRound, msModelSlim, and ModelOpt. Online quantization and quantized KV cache are also supported. For the full list and per-method details, see the [Quantization overview](../user_guide/quantization/overview.md).
```

- [ ] **Step 19.2: Rewrite the streaming Q&A**

In `docs/usage/faq.md`, replace:
```markdown
> Q: Does vLLM-Omni support multimodal streaming input and output?

A: Not yet. We already put it on the [Roadmap](https://github.com/vllm-project/vllm-omni/issues/165). Please stay tuned!
```
with:
```markdown
> Q: Does vLLM-Omni support multimodal streaming input and output?

A: Yes. vLLM-Omni supports streaming outputs across modalities, including text, speech, image, and video. See the [Streaming Video Input API](../serving/video_stream_api.md) and the [Chat Completions API](../serving/chat_completions_api.md) (which supports `stream=True`) for details.
```

- [ ] **Step 19.3: Verify**

```bash
grep -n "0\.16\.0\|Not yet\|plan to introduce" docs/usage/faq.md
```

Expected: no output.

- [ ] **Step 19.4: Commit**

```bash
git add docs/usage/faq.md
git commit -m "docs: fix stale FAQ entries (quantization + streaming)

The FAQ claimed quantization was 'planned for 0.16.0' and streaming was
'Not yet' — both features ship today. Update to reflect current state and
link to the relevant docs."
```

### Task 20: Align Python version claims

`pyproject.toml` says `requires-python = ">=3.10,<3.14"`. Docs disagree: DOCS_GUIDE says 3.9+, contributing/README says 3.10–3.12, comfyui says 3.12+.

**Files:**
- Modify: `docs/contributing/DOCS_GUIDE.md` (line 137: "requires 3.9+")
- Modify: `docs/contributing/README.md` (line 22: "3.10 to 3.12")
- Modify: `docs/features/comfyui.md` (line 8: "Python 3.12 or above")

- [ ] **Step 20.1: Fix DOCS_GUIDE Python version**

Note: DOCS_GUIDE will be substantially rewritten in Task 21. If you're doing Task 20 before Task 21, fix the line now; if doing Task 21 first, the rewrite will incorporate the correct version. In `docs/contributing/DOCS_GUIDE.md`, replace:
```markdown
- Check Python version (requires 3.9+)
```
with:
```markdown
- Check Python version (requires 3.10–3.13, per `pyproject.toml`)
```

- [ ] **Step 20.2: Fix contributing/README.md Python version**

In `docs/contributing/README.md`, replace:
```markdown
    vLLM-Omni is compatible with Python versions 3.10 to 3.12. However, we recommend developing with Python 3.12 to minimize the chance of your local environment clashing with our CI environment.
```
with:
```markdown
    vLLM-Omni is compatible with Python 3.10–3.13 (see `pyproject.toml`). We recommend developing with Python 3.12 to minimize the chance of your local environment clashing with our CI environment.
```

- [ ] **Step 20.3: Fix comfyui.md Python version**

In `docs/features/comfyui.md`, replace:
```markdown
- Python 3.12 or above
```
with:
```markdown
- Python 3.10–3.13 (vLLM-Omni requirement; ComfyUI itself may have its own constraints)
```

- [ ] **Step 20.4: Scan for any other Python version claims**

```bash
grep -rn "python 3\.9\|python 3\.10 to 3\.12\|3\.12 or above\|requires 3\.9" docs/ --include="*.md"
```

Expected: no output (all fixed). If other instances appear, fix them to state "3.10–3.13" per `pyproject.toml`.

- [ ] **Step 20.5: Commit**

```bash
git add docs/contributing/DOCS_GUIDE.md docs/contributing/README.md docs/features/comfyui.md
git commit -m "docs: align Python version claims to pyproject.toml (>=3.10,<3.14)

DOCS_GUIDE said 3.9+, contributing/README said 3.10–3.12, comfyui said
3.12+. All now state 3.10–3.13, matching the canonical requires-python in
pyproject.toml. Dev recommendation of 3.12 preserved."
```

### Task 21: Resolve recipe ownership contradiction

`recipes/README.md` says add recipes in-repo; `docs/contributing/model/adding_omni_model.md` says add them to `vllm-project/recipes`. The in-repo `recipes/` is the active one (per `recipes/README.md`'s detail and the 30+ recipes present).

**Files:**
- Modify: `docs/contributing/model/adding_omni_model.md`

- [ ] **Step 21.1: Fix the "Adding a Model Recipe" section**

In `docs/contributing/model/adding_omni_model.md`, replace:
```markdown
## Adding a Model Recipe

After implementing and testing your model, please add a model recipe to the [vllm-project/recipes](https://github.com/vllm-project/recipes) repository. This helps other users understand how to use your model with vLLM-Omni.
```
with:
```markdown
## Adding a Model Recipe

After implementing and testing your model, please add a model recipe to the in-repo [`recipes/`](https://github.com/vllm-project/vllm-omni/tree/main/recipes) directory. This helps other users understand how to use your model with vLLM-Omni. See [`recipes/README.md`](https://github.com/vllm-project/vllm-omni/tree/main/recipes/README.md) for the layout convention (organize by model vendor, one Markdown file per model family).
```

- [ ] **Step 21.2: Fix the "Example" reference**

In `docs/contributing/model/adding_omni_model.md`, replace:
```markdown
### Example

For reference, see the [LongCat recipe example](https://github.com/vllm-project/recipes/pull/179) which demonstrates the expected format and structure.
```
with:
```markdown
### Example

For reference, see the [Qwen3-Omni recipe](https://github.com/vllm-project/vllm-omni/tree/main/recipes/Qwen/Qwen3-Omni.md) in this repository, which demonstrates the expected format and structure.
```

- [ ] **Step 21.3: Fix the "Recipe Location" section**

In `docs/contributing/model/adding_omni_model.md`, replace:
```markdown
### Recipe Location

Create your recipe file in the appropriate directory structure:
- For organization-specific models: `OrganizationName/ModelName.md`
- For general models: `ModelName.md`

The recipe should be a Markdown file that provides clear, reproducible instructions for users to get started with your model.
```
with:
```markdown
### Recipe Location

Create your recipe file in the in-repo `recipes/` directory, organized by vendor:
- For organization-specific models: `recipes/OrganizationName/ModelName.md`
- For general models: `recipes/ModelName.md`

The recipe should be a Markdown file that provides clear, reproducible instructions for users to get started with your model. See [`recipes/TEMPLATE.md`](https://github.com/vllm-project/vllm-omni/tree/main/recipes/TEMPLATE.md) for the recommended format.
```

- [ ] **Step 21.4: Fix the Summary checklist item**

In `docs/contributing/model/adding_omni_model.md`, replace:
```markdown
7. **Add model recipe** to the [vllm-project/recipes](https://github.com/vllm-project/recipes) repository (see [Adding a Model Recipe](#adding-a-model-recipe) section)
```
with:
```markdown
7. **Add model recipe** to the in-repo [`recipes/`](https://github.com/vllm-project/vllm-omni/tree/main/recipes) directory (see [Adding a Model Recipe](#adding-a-model-recipe) section)
```

- [ ] **Step 21.5: Verify consistency**

```bash
grep -n "vllm-project/recipes" docs/contributing/model/adding_omni_model.md
```

Expected: no output (all references now point to the in-repo `recipes/` dir). Note: `recipes/README.md` mentions `vllm-project/recipes` as a structural reference — that's fine, it's not telling contributors to add files there.

- [ ] **Step 21.6: Commit**

```bash
git add docs/contributing/model/adding_omni_model.md
git commit -m "docs: fix recipe ownership contradiction in adding_omni_model

adding_omni_model.md told contributors to add recipes to vllm-project/recipes,
but recipes/README.md says to add them in-repo. The in-repo recipes/ dir is
active (30+ recipes present). Point contributors to the in-repo location."
```

### Task 22: Rewrite DOCS_GUIDE.md

`docs/contributing/DOCS_GUIDE.md` documents a structure (`architecture/`, `index.md`, `examples/`, `stylesheets/`) and a GitHub Pages workflow that no longer exist. Rewrite it to reflect reality: `.nav.yml`/awesome-nav, `design/`, Read the Docs, hooks that mutate tracked source.

**Files:**
- Modify: `docs/contributing/DOCS_GUIDE.md`

- [ ] **Step 22.1: Read the current (stale) content**

```bash
cat docs/contributing/DOCS_GUIDE.md
```

- [ ] **Step 22.2: Replace the entire file with the rewritten version**

Overwrite `docs/contributing/DOCS_GUIDE.md` with:

````markdown
# Documentation Guide

This guide explains how vLLM-Omni documentation is structured, built, and
deployed. It is intended for contributors who want to add or edit docs.

## Build & Preview Locally

### Prerequisites

Install documentation dependencies:

```bash
uv pip install -e ".[docs]"
```

### Serve (live reload)

```bash
mkdocs serve
```

The site is available at `http://127.0.0.1:8000` and reloads on file changes.

### Strict build (catches broken links)

```bash
mkdocs build --strict
```

This is the same build Read the Docs runs. It fails on warnings, including
broken internal links and missing nav references. **Always run this before
pushing doc changes.**

> ⚠️ **Hooks rewrite tracked files on build.** The `generate_examples.py`,
> `generate_api_readme.py`, and `generate_argparse.py` hooks write to tracked
> source files (e.g. `docs/.nav.yml`, `docs/user_guide/examples/`,
> `docs/api/README.md`) during `mkdocs build`. A clean build will dirty the
> tree. This is expected. Do **not** hand-edit generated files under those
> paths — your changes will be overwritten on the next build. Edit the source
> under `examples/` instead, and let the hooks regenerate the docs.

## Directory Structure

```
docs/
├── .nav.yml                      # Main navigation (awesome-nav plugin)
├── README.md                     # Landing page (home)
├── getting_started/              # Install + quickstart
│   └── installation/
│       └── .nav.yml              # Sub-directory nav
├── serving/                      # API reference (OpenAI-compatible endpoints)
├── user_guide/                   # Feature guides for end users
│   ├── diffusion/                # Diffusion features (parallelism, cache, etc.)
│   ├── quantization/             # Quantization methods
│   └── examples/                 # AUTO-GENERATED from examples/ (do not edit)
├── design/                       # Architecture & feature design docs (contributors)
│   ├── feature/
│   └── module/
├── configuration/                # Runtime config reference
├── features/                     # Standalone feature pages (sleep, comfyui, etc.)
├── contributing/                 # Contributor guides (this file, CI, model onboarding)
│   └── ci/
│       └── .nav.yaml             # Sub-directory nav
├── community/                    # Governance, meetups, contact
├── api/                          # API reference (mkdocstrings, auto-generated README)
├── cli/                          # CLI reference (auto-generated via generate_argparse.py)
└── mkdocs/                       # Theme overrides, hooks, JS, CSS
    └── hooks/
```

## Navigation Model

vLLM-Omni uses the [awesome-nav](https://github.com/lukasgeiter/mkdocs-awesome-nav)
plugin for navigation, configured via `.nav.yml` files.

- **`docs/.nav.yml`** — the main navigation tree. Defines top-level sections
  (User Guide, Developer Guide, API Reference, etc.) and explicit ordering.
- **Sub-directory `.nav.yml` / `.nav.yaml`** — override nav for a sub-tree
  (e.g. `docs/contributing/ci/.nav.yaml`, `docs/getting_started/installation/.nav.yml`).
- **Glob entries** — `glob: contributing/*` auto-includes all markdown files
  in a directory. Use `exclude:` to skip files that have explicit entries
  elsewhere.
- **`exclude_docs`** in `mkdocs.yml` — excludes files from the build entirely
  (currently `**/*.inc.md` snippet files).

### Adding a new page

1. Create the `.md` file in the appropriate `docs/` subdirectory.
2. Add an entry to the relevant `.nav.yml` (either an explicit
   `- Title: path/to/file.md` line or rely on a glob).
3. Run `mkdocs build --strict` to verify it renders and has no broken links.

## Auto-generated Content

Three MkDocs hooks generate content at build time:

| Hook | Generates | Source |
|---|---|---|
| `generate_examples.py` | `docs/user_guide/examples/**` + the Examples section of `.nav.yml` | `examples/` directory |
| `generate_api_readme.py` | `docs/api/README.md` | `vllm_omni/` package (mkdocstrings) |
| `generate_argparse.py` | `docs/cli/**` | CLI argparse definitions |

**Never hand-edit generated files.** Edit the source (e.g. add an
`examples/online_serving/my_model/` directory with an `README.md`) and the
hook will generate the corresponding doc page on the next build.

## API Documentation

API reference is auto-generated using `mkdocs-api-autonav` and `mkdocstrings`:

- Public modules in `vllm_omni/` are auto-discovered (see the `api-autonav`
  config in `mkdocs.yml` for exclusions).
- Use `[module.name.ClassName][]` syntax for cross-references in markdown.
- Add Google- or NumPy-style docstrings to public classes/functions/methods.

## Snippet Includes

Use the `pymdownx.snippets` extension to include shared content:

```markdown
--8<-- "docs/getting_started/installation/python_env_setup.inc.md"
```

Files with the `.inc.md` suffix are excluded from the nav (see
`exclude_docs` in `mkdocs.yml`) — they are include-only fragments.

## Deployment

Documentation is deployed via [Read the Docs](https://readthedocs.org),
configured in [`.readthedocs.yml`](../../.readthedocs.yml).

- **Build trigger:** every push to `main`.
- **Strict mode:** `fail_on_warning: true` — broken links or missing
  references fail the build.
- **Live URL:** https://docs.vllm.ai/projects/vllm-omni/

GitHub Pages is **not** used. Do not add GitHub Pages workflow references.

## Markdown Extensions

The site uses MkDocs Material with these extensions (see `mkdocs.yml`):

- `admonition` + `pymdownx.details` — callouts (`!!! note`, `!!! warning`)
- `pymdownx.superfences` — code blocks + Mermaid diagrams
- `pymdownx.tabbed` — content tabs
- `pymdownx.highlight` — syntax highlighting
- `pymdownx.arithmatex` — math rendering (MathJax)
- `pymdownx.emoji` — emoji/icons
- `toc` — table of contents with permalinks

## Troubleshooting

### Build fails with "broken link"

Run `mkdocs build --strict` and look for the offending path. Common causes:
relative path depth is wrong (count `../` carefully), or the target file was
renamed/deleted without updating inbound links.

### Nav entry doesn't appear

Check the relevant `.nav.yml` — is the file listed? If using a glob, is the
file in the globbed directory and not excluded? Run `mkdocs build --strict`
for diagnostics.

### Generated file keeps reverting my edit

You're editing a hook-generated file (see "Auto-generated Content" above).
Edit the source instead.
````

- [ ] **Step 22.3: Verify no stale references remain**

```bash
grep -n "architecture/\|index\.md\|GitHub Pages\|stylesheets/" docs/contributing/DOCS_GUIDE.md
```

Expected: no output. (The word "architecture" may appear in prose describing `design/` — that's fine. Only flag literal `architecture/` path references.)

- [ ] **Step 22.4: Commit**

```bash
git add docs/contributing/DOCS_GUIDE.md
git commit -m "docs: rewrite DOCS_GUIDE to reflect actual structure

The old DOCS_GUIDE documented nonexistent directories (architecture/, index.md,
examples/, stylesheets/) and a GitHub Pages workflow. Actual deployment uses
Read the Docs with fail_on_warning. Rewrite to cover the real .nav.yml/awesome-nav
model, auto-generated content hooks, and the known hook-mutation caveat."
```

### Task 23: Batch 3 verification

- [ ] **Step 23.1: Strict build passes**

```bash
git checkout -- docs/.nav.yml docs/user_guide/examples/ docs/api/README.md 2>/dev/null
mkdocs build --strict 2>&1 | tail -10
```

Expected: exit 0.

- [ ] **Step 23.2: Truth correction checks**

```bash
# FAQ no longer claims quantization is planned or streaming is unavailable
grep -n "0\.16\.0\|Not yet\|plan to introduce" docs/usage/faq.md
# Expected: no output

# Python versions aligned
grep -rn "python 3\.9\|3\.10 to 3\.12\|3\.12 or above\|requires 3\.9" docs/ --include="*.md"
# Expected: no output

# Recipe guidance consistent
grep -n "vllm-project/recipes" docs/contributing/model/adding_omni_model.md
# Expected: no output (contributors now pointed to in-repo recipes/)

# DOCS_GUIDE no longer references stale structure
grep -n "architecture/\|index\.md\|GitHub Pages" docs/contributing/DOCS_GUIDE.md
# Expected: no output
```

- [ ] **Step 23.3: Push and open PR 3**

```bash
git push -u origin docs-cleanup-batch3
gh pr create --title "docs: fix FAQ/Python/recipe truth drift, rewrite DOCS_GUIDE" \
  --body "Batch 3 of the docs cleanup plan (see docs/superpowers/specs/2026-07-13-docs-cleanup-design.md).

- Fix stale FAQ: quantization and streaming both ship today (were 'planned'/'Not yet')
- Align Python version claims to pyproject.toml (>=3.10,<3.14) across DOCS_GUIDE, contributing/README, comfyui
- Resolve recipe ownership contradiction: adding_omni_model now points to in-repo recipes/ (matching recipes/README.md)
- Rewrite DOCS_GUIDE to reflect actual structure (.nav.yml/awesome-nav, Read the Docs, hooks, real directory layout)

Verification: mkdocs build --strict passes."
```

---

## Post-merge: final verification

- [ ] **Step 24.1: After all three PRs merge, run a full verification**

```bash
git fetch origin
git checkout main
git reset --hard origin/main
mkdocs build --strict 2>&1 | tail -10
```

Expected: exit 0.

- [ ] **Step 24.2: Confirm the deferred follow-ups are tracked**

The spec (§10) records the deferred items: deterministic generation (codex Phase 1), docs CI gate (codex Phase 2), IA reorg, golden paths, governance, versioned docs. These are NOT part of this plan. If the team wants to pursue them, a new spec + plan should be created from codex's audit.
