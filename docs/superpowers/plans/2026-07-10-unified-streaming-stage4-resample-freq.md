# Unified Streaming Stage 4: `resample` `freq=` re-signature

> **STATUS: DRAFT - awaiting user sign-off before execution.** This changes `resample`'s public signature (a user-facing breaking change) and has design decisions that need confirmation (see "Decisions needed" below). Do NOT start execution until the user greenlights and resolves the open decisions.

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Collapse `resample`'s two badly-named bucket arguments `every=` / `count=` into a single `freq=` argument whose meaning is read from the index (following pandas `date_range`), per the settled design spec.

**Architecture:** The C++ core stays pure integer-index space (unchanged). All of `freq=`'s contextual interpretation lives in the thin Python layer: it inspects the index (absent / integer / datetime64), validates the `freq` type against it, and translates to the existing engine call (count mode vs index-span mode, both already implemented in the resample node). No C++ changes.

**Tech Stack:** Python 3 / numpy; existing pybind11 engine (unchanged); pytest; Sphinx/myst-nb docs.

---

## Decisions needed before execution (for the user)

**D1. Count-mode with a provided index is dropped.** Per the spec, index presence selects the mode: no index -> `freq` is a count (every N events); index present -> `freq` is an index-span. So the current ability to do "every N *events* while carrying an index" (`resample(values, index, count=N)`) has no `freq=` spelling. Confirm this is intended. (If some users need "every N ticks" with a timestamp index, we would need a separate opt-in; the spec says no.)

**D2. Multi-column dict form (`resample(t, agg={...})`) mode mapping.** The dict form always takes a clock/timeline `t` (an index), so under D1 `freq=int` is a span. The multi-column node currently ALSO supports count-mode over distinct ticks (MCB Task 6a). Options: (a) drop count-mode for the dict form (span only, consistent with D1); (b) keep a count escape hatch for the dict form only. Recommend (a) for consistency.

**D3. Migration strategy.** ~158 `every=` + 36 `count=` call sites across source, 14 test files, 4 notebooks, 2 doc pages. Options:
- **(a) Hard cutover (recommended):** remove `every=`/`count=`, add `freq=`, migrate every call site in the same change set. Clean, matches the spec's "collapse into one argument"; pre-1.0 so the break is acceptable. Larger diff.
- **(b) Additive + deprecate:** add `freq=`, keep `every=`/`count=` emitting a deprecation warning for one release, migrate later. Smaller blast radius now, but leaves three ways to call resample temporarily (the user dislikes needless variants).

**D4. Scope of this stage.** This plan covers ONLY the `freq=` re-signature. The spec's other Stage-4 items - `(index, NaN)` heartbeats and retiring `advance()`/`dag.live()` - are deliberately deferred to a Stage 4b, because they are a separate, more invasive change to the live/clock path. Confirm this split.

The task breakdown below assumes **D1 = yes, D2 = (a), D3 = (a) hard cutover, D4 = freq-only**. If any decision changes, the affected tasks change.

## Global Constraints

- Causality unchanged; `batch == lazy` preserved (the freq= layer only renames/translates arguments; the engine and its outputs are identical).
- The C++ core stays pure integer-index space; all `freq` interpretation is Python-layer.
- No version-file edits. No em-dashes / no ` -- ` in code, comments, docstrings, or notebooks.
- `origin=`, `label=`, `fill=`, `agg=` carry over unchanged.
- Valid `freq` per index type (from the spec):

  | index type | valid `freq` | meaning |
  |---|---|---|
  | none | `int` | bar every N events (count) |
  | integer | `int` | span of N index units |
  | datetime64 / timestamp | offset str (`"1min"`) or `timedelta` | wall-clock bar, converted to int units |

- The 2 pre-existing `TestBOP` failures are gone; the suite baseline is now 3959 passed, 2 skipped. Zero new failures.

---

### Task 1: `freq=` parsing + validation layer (single-column resample)

**Files:**
- Modify: `screamer/streams.py` (`resample` signature; new `_resample_freq_to_engine(freq, index)` helper; `_resample_validate`)
- Test: `tests/test_streams_resample.py` (new `freq=` tests; the batch==freq equivalence to the old every=/count=)

**Interfaces:**
- Produces: `resample(values, index=None, *, freq, agg="last", origin=0, label="left", fill="skip")`. `_resample_freq_to_engine(freq, index) -> (mode, width)` where `mode` is `"count"` or `"span"` and `width` is the integer bucket size the engine consumes.

- [ ] **Step 1: Write the failing equivalence tests**

```python
def test_freq_no_index_equals_count():
    import numpy as np
    from screamer.streams import resample
    v = np.array([1.0, 2, 3, 4, 5, 6])
    old = resample(v, count=2, agg="mean")          # current API
    new = resample(v, freq=2, agg="mean")           # freq, no index -> count
    np.testing.assert_array_equal(np.asarray(new.values), np.asarray(old.values))
    np.testing.assert_array_equal(np.asarray(new.index), np.asarray(old.index))


def test_freq_integer_index_equals_every():
    import numpy as np
    from screamer.streams import resample
    v = np.array([1.0, 2, 3, 4, 5]); k = np.array([0, 1, 2, 10, 11])
    old = resample(v, k, every=10, agg="sum")       # current API (index-span)
    new = resample(v, k, freq=10, agg="sum")        # freq, integer index -> span
    np.testing.assert_array_equal(np.asarray(new.values), np.asarray(old.values))
    np.testing.assert_array_equal(np.asarray(new.index), np.asarray(old.index))


def test_freq_timedelta_on_integer_index_raises():
    import numpy as np, pytest
    from screamer.streams import resample
    v = np.array([1.0, 2, 3]); k = np.array([0, 1, 2])
    with pytest.raises((TypeError, ValueError)):
        resample(v, k, freq="1min", agg="mean")     # offset freq needs datetime64 index
```

- [ ] **Step 2: Run to confirm failure** (`freq` is not yet a parameter).

- [ ] **Step 3: Add `_resample_freq_to_engine`**

```python
def _resample_freq_to_engine(freq, index):
    """Translate the contextual freq into (mode, width) for the engine.

    - index is None            -> freq must be a positive int; mode "count".
    - index is integer dtype   -> freq must be a positive int; mode "span".
    - index is datetime64      -> freq is an offset string or timedelta; convert to
      integer index units; mode "span".
    Raises a clear error on a nonsensical (index dtype, freq type) pair.
    """
    # ... implementation: dtype inspection + offset/timedelta -> int units ...
```

(Full implementation to be written during execution; it reuses the existing datetime64 unit handling if present, else adds it. The datetime64 offset parsing is the one genuinely new piece; keep it minimal - support the offsets the notebooks actually use.)

- [ ] **Step 4: Re-signature `resample`** to `freq=`, routing through `_resample_freq_to_engine` to the existing engine call. Internally it still calls the same node with the computed count/span; only the public argument changes.

- [ ] **Step 5-6:** Run the equivalence tests (pass), then the resample suite.

- [ ] **Step 7: Commit.**

---

### Task 2: migrate all `every=` / `count=` call sites (source + tests)

**Files:** `screamer/streams.py` (internal `_resample_dict`/`_resample_ohlcv`/`_resample_via_cpp` still speak count/span internally - keep those internal names, only the PUBLIC `resample` takes `freq=`); every `tests/test_*` that calls `resample(..., every=)` / `count=`.

- [ ] Grep `every=` / `count=` in `screamer/` and `tests/`; for each PUBLIC `resample(...)` call, replace with `freq=` (no index -> was count; with index -> was every). Internal helper signatures may keep `every`/`count` OR be renamed to `mode,width` - decide during execution for minimal churn. Do NOT change the multi-column node's internal mode enum.
- [ ] The dict-form `resample(t, agg={...}, every=/count=)` public calls migrate to `freq=` per D2(a).
- [ ] Run the full suite after each file batch; keep it green.
- [ ] Commit in reviewable batches.

---

### Task 3: docs, notebooks, migration table, help.json

**Files:** `docs/functions_streams/resample.md`, `docs/functions_streams/Stream.md`, the 4 notebooks using `every=`/`count=`, `docs/multistream.md` (migration table), regenerate `screamer/data/help.json` + `docs/function_index.txt`.

- [ ] Update the `resample` doc page to document `freq=` with the index-type table; remove `every=`/`count=`.
- [ ] Migrate the 4 notebooks to `freq=`; re-execute them clean (nbmake).
- [ ] Add `every=`/`count=` -> `freq=` rows to the migration table.
- [ ] Regenerate help.json + function_index.txt; confirm no `every=`/`count=` remain in generated artifacts or the resample signature.
- [ ] Run `tests/test_doc_coverage.py`, `tests/test_build_help_registry.py`, and the notebook (nbmake) checks.
- [ ] Commit.

---

## Self-review notes

- **Equivalence is the oracle:** each `freq=` call must produce byte-identical values+index to the `every=`/`count=` call it replaces (Task 1 asserts this directly). The engine is unchanged, so this must hold.
- **The only genuinely new code** is datetime64 `freq` (offset str / timedelta) -> integer index units. Keep it to the offsets the notebooks use; document the supported set.
- **Deferred to Stage 4b:** `(index, NaN)` heartbeats and retiring `advance()` / `dag.live()`.
- **Do not touch** the C++ resample node, the multi-column node's mode enum, or any operator from Stages 2/3/3b.
