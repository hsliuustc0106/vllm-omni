# Code-Quality Patterns

Three fragility patterns to sweep on **every** PR, in both quick and full mode. These are pervasive in the existing codebase (hundreds of pre-existing sites) — so the checks are **diff-scoped**: only lines the PR *adds* count. Whole-repo greps light up every PR with the backlog and are useless.

All commands assume:
```bash
BASE=$(git merge-base HEAD origin/main)
ADDED='git diff ${BASE}...HEAD -- "*.py" | grep "^+[^+]"'
```

**Severity:** ⚠ for any *new* instance introduced by the diff; ✗ for the acute sub-cases defined per pattern. Acute ✗ findings belong in the report's blocking count.

---

## 1. `**kwargs` string-lookup passthrough

### Anti-pattern
```python
def forward(self, **kwargs):
    # BLOCKER-prone: raw dict + string keys duplicated across files
    if "runtime_additional_information" in kwargs and \
       "model_intermediate_buffer" not in kwargs:
        ...
```
Real instances of this exact guard are copy-pasted across `cosyvoice3.py`, `indextts2_talker.py` (×2), `indextts2_s2mel_decoder.py`, `qwen3_tts_talker.py`, `qwen3_omni.py`. A typo in any string key silently no-ops; there is no type check, no IDE completion, and unknown keys are silently dropped.

### Why dangerous
Raw `dict` + `**kwargs` + lists of param-name strings is **fragile by construction**: the compiler can't see a missing or misspelled key, so failures are silent. Each duplicated copy is a new place for the strings to drift.

### Severity
- ⚠ any *new* `**kwargs` signature whose body does `kwargs["..."]` / `kwargs.get("...")` / `"..." in kwargs`.
- ✗ when the diff introduces a **new** string key that is duplicated across ≥2 files without a shared module constant, or when `**kwargs` plumbing **silently drops unknown keys** on a fail-fast path (init, config validation, weight loading).

### Detect (diff-scoped)
```bash
git diff ${BASE}...HEAD -- "*.py" | grep "^+[^+]" \
  | grep -E '\*\*kwargs|kwargs(\.get|\[)|"[a-z_]+"\s+in\s+kwargs'
```

### Fix
Prefer explicit typed params. The repo already has the right primitive: `msgspec.Struct` with `forbid_unknown_fields=True` rejects unknown keys instead of silently dropping them — see `_StructBase` in `vllm_omni/data_entry_keys.py:105` and `MoriPullRequest` in `vllm_omni/distributed/omni_connectors/connectors/mori_transfer_engine_connector.py:81`. A `@dataclass` or `TypedDict` works too.

If `**kwargs` is unavoidable (vLLM base-class compat), centralize the key strings as module-level constants behind **one** typed accessor, instead of repeating the literal across files.

---

## 2. Broad exception swallow

### Anti-pattern
```python
# BLOCKER-prone: catches everything, returns None on any failure
try:
    evt = build_event(...)
    return evt
except Exception:
    return None
```
Real clusters: `vllm_omni/metrics/stats.py:240,274,367,450` and `vllm_omni/metrics/{modality,utils}.py` — all `except Exception: return None`.

### Why dangerous
A generic `except Exception:` is **too flexible**. If we catch an exception type we don't expect, it is almost always an unhandled edge case that needs to be *fixed* — not swallowed into `None`/`pass`/`continue`. Broad catches turn fixable bugs into silent wrong behavior and make debugging impossible.

The repo backs this with ruff `BLE001` (broad-exception catch); it is currently suppressed via `# noqa: BLE001` in `vllm_omni/patch.py:413`. Prefer fixing over suppressing.

### Severity
- ⚠ any *new* `except Exception:` / `except BaseException:` / bare `except:`.
- ✗ for `except:` / `except Exception: pass|return None|continue` on a **fail-fast path** (init, config validation, weight loading, request handling, connector setup). This extends the Bug-Fix "No silent failure risk" ✗ in [checklists.md](checklists.md).

### Detect (diff-scoped)
```bash
# Any new broad catch (⚠ baseline):
git diff ${BASE}...HEAD -- "*.py" | grep "^+[^+]" \
  | grep -E 'except\s*(Exception|BaseException)?\s*:'
# Note: this matches only the immediate-colon swallow forms (`except :`,
# `except Exception:`) — it deliberately excludes specific catches like
# `except ValueError:` and the `except Exception as e:` variant. Eyeball
# the latter; `as e` usually means it is logged, but still verify.
```

### Fix
Catch the **specific** types you actually expect (`ValueError`, `KeyError`, `AttributeError`, `OSError`, `torch.cuda.OutOfMemoryError`, `msgspec.ValidationError`, …). At a genuine top-level / best-effort boundary (metrics, signal handlers), at minimum **log** the exception; never swallow into a bare `pass`/`return None` on a path that should fail loudly.

---

## 3. `Any` / wrong type annotations

### Anti-pattern
```python
# ⚠ Any param + Any return — no contract, no narrowing
def _extract_mm_output(engine_outputs: Any) -> dict[str, Any]:
    ...
```

### Why dangerous (root cause)
`Any` disables the type checker for that position, so typos and wrong shapes pass silently. The repo has 636 `: Any` params, 244 `-> Any` returns, and **1,173 `SimpleNamespace` in `tests/`**. The `Any` leak is largely test-driven: vibe-coded unit tests fake objects with `SimpleNamespace`, which forces the production code consuming them to be typed `Any`, and the `Any` then propagates outward into public signatures.

### Severity
- ⚠ any *new* `: Any` / `-> Any` / fully-untyped signature in production code.
- ✗ for a **wrong** annotation (actively misleading — e.g. `-> bool` that returns `Optional[bool]`), or a *new* `SimpleNamespace` in a test that mimics an object which already has a real typed stub / `@dataclass` / `TypedDict` / `Protocol`.

### Detect (diff-scoped)
```bash
# Any leaking into new signatures:
git diff ${BASE}...HEAD -- "*.py" | grep "^+[^+]" \
  | grep -E ':\s*Any\b|->\s*Any\b'
# SimpleNamespace newly added in tests (the Any-leak source):
git diff ${BASE}...HEAD -- "tests/" | grep "^+[^+]" | grep "SimpleNamespace"
```

### Fix
Replace `Any` with the concrete type, a `Protocol`, or a `Union`. If a type is genuinely dynamic, prefer `object` + `isinstance` narrowing over `Any` — `object` keeps the checker engaged. In tests, replace `SimpleNamespace` with the real class, a small `@dataclass`/`TypedDict`, or a `Protocol` + fake, so the system under test stays fully typed.

---

## Running the sweep

Run all three detections against `${BASE}...HEAD`, then for each hit decide ⚠ vs ✗ using the severity rules above. Roll the results into the report as a single **Code quality** dimension row (count of ⚠ and ✗), alongside the type-specific checklist from [checklists.md](checklists.md).
