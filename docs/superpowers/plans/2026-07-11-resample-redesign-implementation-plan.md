# resample redesign implementation plan (contextual `freq`, `agg`-as-functor, composition)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Implement the approved design `docs/superpowers/specs/2026-07-11-resample-redesign-freq-agg-functor-design.md`: collapse `every=`/`count=` into one contextual `freq=`, make `agg` any functor with short string synonyms, delete the `agg={dict}`/`Input`-ports form, and move multi-column / multi-input bars to composition (`combine_latest` + arithmetic).

**Architecture:** The C++ engine is unchanged (it already buckets by count or index-span and already applies an owned reducer per bar with reset - the `agg=<functor>` path is verified working today). All new logic is a thin Python layer: a contextual `freq` -> `(mode, width)` translator, a small `agg` string -> functor map, one new `count` reducer, and deletion of the dict path. Hard cutover (pre-1.0), no deprecation shims.

**Tech Stack:** Python 3 / numpy (no new deps; offsets via `np.timedelta64`); the existing pybind11 engine; pytest; Sphinx/myst-nb notebooks.

## Global Constraints

- **Equivalence is the oracle:** every `freq=` call must produce byte-identical values and index to the `every=`/`count=` call it replaces (engine unchanged); every `agg="name"` must equal `agg=<mapped functor>`. `batch == lazy` (Stage 2/3) is preserved.
- **Causality unchanged;** the C++ core stays pure integer-index space; all `freq` interpretation is Python-layer.
- No version-file edits. No em-dashes and no ` -- ` (double-hyphen) in code, comments, docstrings, or notebooks.
- `origin=`, `label=`, `fill=` carry over unchanged.
- Contextual `freq` (from the spec):

  | index | `freq` | mode |
  |---|---|---|
  | none | positive `int` | count (every N events) |
  | integer | positive `int` | span (N index units) |
  | `datetime64` | offset str (`"1min"`) or `timedelta`/`np.timedelta64` | span (converted to int64 units) |

- Baseline suite: 3959 passed, 2 skipped, 0 failed (the 2 old `TestBOP` failures are already fixed). Zero new failures.

## Decisions to confirm before/at execution (recommended defaults in brackets)

- **D-a. `count` reducer.** Add a small `ExpandingCount` C++ functor (counts non-NaN inputs since reset) so `agg="count"` is a real functor like the rest. [Recommended over a resample-layer special-case, to keep `agg` uniformly a functor.]
- **D-b. datetime64 offset scope.** Support `timedelta` / `np.timedelta64` directly, plus a small string parser for the common calendar-free offsets (`"s"`, `"min"`/`"T"`, `"h"`, `"D"`, with an integer multiplier, e.g. `"5min"`). No pandas dependency; month/business offsets are out of scope for v1 (raise a clear error). [Keeps the core dependency-free.]
- **D-c. finance aliases. RESOLVED (user): include them.** Ship the general names (`max`, `min`, `first`, `last`) AND the finance synonyms `high`->max, `low`->min, `open`->first, `close`->last, plus the `ohlc` combo. The `resample` docs must explicitly mention that these string synonyms exist (and that any functor is also accepted).

---

### Task 1: `freq=` translation layer and re-signature (no-index count + integer-index span)

**Files:**
- Modify: `screamer/streams.py` (add `_resample_freq_to_engine`; re-signature `resample`; replace `_resample_validate`)
- Test: `tests/test_streams_resample.py`

**Interfaces:**
- Produces: `resample(values, index=None, *, freq, agg="last", origin=0, label="left", fill="skip")`; `_resample_freq_to_engine(freq, index) -> (mode, width)` where `mode` in `{"count", "span"}` and `width` is the positive int the engine consumes. (datetime64 handled in Task 2.)

- [ ] **Step 1: Write the failing equivalence tests**

```python
def test_freq_no_index_equals_count():
    import numpy as np
    from screamer.streams import resample
    v = np.array([1.0, 2, 3, 4, 5, 6])
    old = resample(v, count=2, agg="mean")
    new = resample(v, freq=2, agg="mean")
    np.testing.assert_array_equal(np.asarray(new.values), np.asarray(old.values))
    np.testing.assert_array_equal(np.asarray(new.index), np.asarray(old.index))


def test_freq_integer_index_equals_every():
    import numpy as np
    from screamer.streams import resample
    v = np.array([1.0, 2, 3, 4, 5]); k = np.array([0, 1, 2, 10, 11])
    old = resample(v, k, every=10, agg="sum")
    new = resample(v, k, freq=10, agg="sum")
    np.testing.assert_array_equal(np.asarray(new.values), np.asarray(old.values))
    np.testing.assert_array_equal(np.asarray(new.index), np.asarray(old.index))


def test_freq_rejects_nonpositive_and_missing():
    import numpy as np, pytest
    from screamer.streams import resample
    with pytest.raises((TypeError, ValueError)):
        resample(np.array([1.0, 2, 3]))                 # freq is required
    with pytest.raises(ValueError):
        resample(np.array([1.0, 2, 3]), freq=0)         # must be positive
```

- [ ] **Step 2: Run to confirm failure** (`freq` not yet a parameter; `resample` still requires `every`/`count`).

- [ ] **Step 3: Add `_resample_freq_to_engine`** (integer / no-index only for now; datetime64 branch raises a clear "handled in Task 2" placeholder that Task 2 fills):

```python
def _resample_freq_to_engine(freq, index):
    """Translate the contextual freq into (mode, width) for the engine.

    index is None       -> width = int(freq); mode "count".
    index integer dtype -> width = int(freq); mode "span".
    index datetime64    -> offset/timedelta -> int64 units (Task 2).
    Raises on a non-positive width or a nonsensical (index dtype, freq type) pair.
    """
    if index is not None and np.asarray(index).dtype.kind == "M":   # datetime64
        return _resample_datetime_freq(freq, index)                 # Task 2
    width = int(freq)
    if width <= 0:
        raise ValueError("resample: freq must be a positive integer")
    return ("count" if index is None else "span"), width
```

- [ ] **Step 4: Re-signature `resample`** to `def resample(values, index=None, *, freq, agg="last", origin=0, label="left", fill="skip")`. Internally compute `(mode, width) = _resample_freq_to_engine(freq, index)` and route to the SAME engine call the current `every`(span)/`count` paths use (map `mode=="span"` to the existing `every=width` engine argument, `mode=="count"` to `count=width`). Do not change the engine or the numeric path - only the public argument and the internal translation.

- [ ] **Step 5: Replace `_resample_validate`** so it validates `freq`/`agg`/`label`/`fill` (drop the every/count XOR check; keep the positivity guard inside `_resample_freq_to_engine`).

- [ ] **Step 6-7:** Run the new tests (pass), then the whole `tests/test_streams_resample.py` (the existing every=/count= tests still exist here and are migrated in Task 4; for now keep them running against a temporary internal shim OR migrate the few in this file - choose minimal churn and note it).

- [ ] **Step 8: Commit.**

---

### Task 2: datetime64 `freq` (offset string / timedelta -> int64 units)

**Files:**
- Modify: `screamer/streams.py` (`_resample_datetime_freq`, and index int64-view handling)
- Test: `tests/test_streams_resample.py`

**Interfaces:** Consumes `_resample_freq_to_engine` (Task 1). Produces `_resample_datetime_freq(freq, index) -> ("span", width_int64)` and coerces a `datetime64` index to its int64 view for the engine.

- [ ] **Step 1: Failing tests** (minute bars over a datetime64 index; timedelta and string offsets agree; bad pairs raise):

```python
def test_freq_datetime_offset_and_timedelta_agree():
    import numpy as np
    from screamer.streams import resample
    t = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:30",
                  "2020-01-01T00:01:10"], dtype="datetime64[s]")
    v = np.array([1.0, 2.0, 3.0])
    a = resample(v, t, freq="1min", agg="sum")
    b = resample(v, t, freq=np.timedelta64(60, "s"), agg="sum")
    np.testing.assert_array_equal(np.asarray(a.values), np.asarray(b.values))


def test_freq_timedelta_on_integer_index_raises():
    import numpy as np, pytest
    from screamer.streams import resample
    with pytest.raises((TypeError, ValueError)):
        resample(np.array([1.0, 2, 3]), np.array([0, 1, 2]), freq="1min", agg="sum")
```

- [ ] **Step 2: Run to confirm failure.**

- [ ] **Step 3: Implement `_resample_datetime_freq`** using numpy only (no pandas): accept `timedelta`, `np.timedelta64`, or a small offset-string grammar `"<int?><unit>"` with `unit in {s, min|T, h, D}`; convert to an `int64` count of the index's own resolution units; view the datetime64 index as int64. Raise a clear error for an int/float `freq` on a datetime64 index and for unsupported offsets (month/business). Keep the supported set to the calendar-free offsets the notebooks use (D-b).

- [ ] **Step 4-6:** Tests pass; the string offset and the `timedelta` produce identical bars; integer/float `freq` on a datetime64 index raises; run the resample suite.

- [ ] **Step 7: Commit.**

---

### Task 3: `agg` string synonyms + `ExpandingCount`

**Files:**
- Create (D-a): a minimal `ExpandingCount` functor (C++ + binding) OR, if the reviewer prefers, a resample-layer count; recommended is the functor for uniformity. If C++: `include/screamer/.../expanding_count.h` + binding; run `make install-dev`.
- Modify: `screamer/streams.py` (`_AGG_SYNONYMS` map; resolve a string `agg` to a functor)
- Test: `tests/test_streams_resample.py`

**Interfaces:** Consumes the working `agg=<functor>` path. Produces `_AGG_SYNONYMS = {"sum": ExpandingSum, "mean": ExpandingMean, "max": ExpandingMax, "high": ExpandingMax, "min": ExpandingMin, "low": ExpandingMin, "first": First, "open": First, "last": Last, "close": Last, "std": ExpandingStd, "var": ExpandingVar, "count": ExpandingCount}` (finance synonyms high/low/open/close included per D-c) and `agg="ohlc"` multi-output (kept from the current ohlc path).

- [ ] **Step 1: Failing tests** (each synonym equals its functor; count works; unknown string raises; functor still accepted):

```python
def test_agg_string_equals_functor():
    import numpy as np
    from screamer.streams import resample
    from screamer import ExpandingMean, ExpandingStd
    v = np.array([1.0, 2, 3, 4, 5, 6]); k = np.arange(6)
    for name, func in [("mean", ExpandingMean()), ("std", ExpandingStd())]:
        s = resample(v, k, freq=3, agg=name)
        f = resample(v, k, freq=3, agg=func)
        np.testing.assert_allclose(np.asarray(s.values), np.asarray(f.values), equal_nan=True)


def test_agg_count():
    import numpy as np
    from screamer.streams import resample
    v = np.array([1.0, np.nan, 3, 4, 5, 6]); k = np.arange(6)
    out = resample(v, k, freq=3, agg="count")
    np.testing.assert_array_equal(np.asarray(out.values), [2, 3])   # NaN excluded


def test_agg_unknown_string_raises():
    import numpy as np, pytest
    from screamer.streams import resample
    with pytest.raises(ValueError):
        resample(np.array([1.0, 2, 3]), freq=3, agg="nonsense")
```

- [ ] **Step 2: Run to confirm failure.**

- [ ] **Step 3: Add `ExpandingCount`** (D-a) if going the functor route: a stateful reducer that increments a counter per non-NaN input and outputs the running count; `reset()` zeroes it. Build and `make install-dev`.

- [ ] **Step 4: Add `_AGG_SYNONYMS`** and resolve a string `agg` to `synonyms[agg]()` before the functor path; keep `agg="ohlc"`/`"ohlcv"` on their existing multi-output path; a string not in the map raises a clear error listing the valid names.

- [ ] **Step 5-6:** Tests pass; run the resample suite and the full suite (a C++ change means a rebuild).

- [ ] **Step 7: Commit.**

---

### Task 4: delete the `agg={dict}` / `Input`-ports form; move multi-column to composition

**Files:**
- Modify: `screamer/streams.py` (delete the dict branch in `resample`, `_split_reducer_expr`, `_resample_dict`; keep or internalize `multi_resample` per its other uses; the `is_node` dict branch that built `multi_resample` is removed)
- Test: `tests/test_multi_resample_dict.py` (rewrite to the composition form), `tests/test_streams_resample.py`

**Interfaces:** After this, `resample` accepts only `agg=<functor|string>`; a `dict` `agg` raises a clear error pointing to the `combine_latest` composition.

- [ ] **Step 1: Write the composition oracle tests** (OHLCV via `combine_latest`, VWAP via composition, each vs a hand-computed expected):

```python
def test_ohlcv_via_composition():
    import numpy as np
    from screamer.streams import resample, combine_latest
    price = np.array([10., 11, 9, 12, 8, 13]); vol = np.array([1., 2, 1, 3, 1, 2])
    k = np.arange(6)
    o = resample(price, k, freq=3, agg="first"); h = resample(price, k, freq=3, agg="max")
    l = resample(price, k, freq=3, agg="min");   c = resample(price, k, freq=3, agg="last")
    v = resample(vol,   k, freq=3, agg="sum")
    rows, idx = combine_latest(o, h, l, c, v)
    # bar 0 = [10,11,9,10,4], bar 1 = [12,13,8,13,6]
    np.testing.assert_array_equal(np.asarray(rows)[0], [10, 11, 9, 10, 4])
    np.testing.assert_array_equal(np.asarray(rows)[1], [12, 13, 8, 13, 6])


def test_vwap_via_composition():
    import numpy as np
    from screamer.streams import resample
    price = np.array([10., 20, 30, 40]); vol = np.array([1., 3, 1, 1]); k = np.arange(4)
    num = resample(price * vol, k, freq=2, agg="sum")
    den = resample(vol,         k, freq=2, agg="sum")
    vwap = np.asarray(num.values) / np.asarray(den.values)
    np.testing.assert_allclose(vwap, [(10 + 60) / 4, (30 + 40) / 2])


def test_agg_dict_raises_with_migration_hint():
    import numpy as np, pytest
    from screamer.streams import resample
    from screamer import First
    with pytest.raises((ValueError, TypeError)):
        resample(np.array([1.0, 2, 3]), freq=3, agg={"open": First()})
```

- [ ] **Step 2: Run to confirm the composition tests pass on the current engine** (they use only single-column `resample` + `combine_latest`, which already work) and the dict-raises test fails (dict still accepted today).

- [ ] **Step 3: Delete the dict path** - the `isinstance(agg, dict)` branches in `resample` (both the `is_node` graph branch that called `multi_resample` and the eager `_resample_dict` branch), plus `_split_reducer_expr` and `_resample_dict`. A `dict` `agg` now raises a clear error: "resample no longer accepts agg={...}; build columns with combine_latest of per-stat resamples (see docs)". Leave the low-level `multi_resample` node only if another caller uses it (grep); otherwise remove it too.

- [ ] **Step 4: Rewrite `tests/test_multi_resample_dict.py`** to the composition form (OHLCV via `combine_latest`, signed-flow via upstream `PosPart`/`NegPart` + `sum`), preserving the numeric assertions. Do not weaken them.

- [ ] **Step 5-6:** Run the affected files, then the full suite.

- [ ] **Step 7: Commit.**

---

### Task 5: migrate call sites, docs, notebooks, help.json

**Files:** every `resample(..., every=|count=)` and `agg={...}` public call site in `screamer/` and `tests/`; `docs/functions_streams/resample.md`; the 4 resample notebooks; `docs/multistream.md` (migration table); regenerate `screamer/data/help.json` + `docs/function_index.txt`.

- [ ] **Step 1: Grep and migrate** `every=`/`count=` -> `freq=` (no index -> was count; with index -> was every) and any `agg={...}` -> composition, across `screamer/` and the ~14 test files. Run the suite after each batch; keep it green.
- [ ] **Step 2: Update `docs/functions_streams/resample.md`** to document `freq=` (the index-type table), `agg=` (the string-synonym table (general max/min/first/last AND finance high/low/open/close, plus a sentence stating synonyms exist) + "any functor" + the value-at-bar-end / single-input caveats), and the composition recipe for multi-column / VWAP. Remove `every=`/`count=`/`agg={dict}`. (The deeper docs+notebook rewrite and the "Dag as a reusable user-defined function" framing are the separate deferred docs task - this step is the minimal cutover so the page is not wrong.)
- [ ] **Step 3: Migrate the 4 resample notebooks** to `freq=`/`agg=` and the composition OHLCV; re-execute clean (nbmake).
- [ ] **Step 4: Add `every=`/`count=`/`agg={dict}` -> new-form rows to the `docs/multistream.md` migration table.**
- [ ] **Step 5: Regenerate** `help.json` + `function_index.txt`; confirm no `every=`/`count=` remain in the resample signature or generated artifacts.
- [ ] **Step 6:** Run `tests/test_doc_coverage.py`, `tests/test_build_help_registry.py`, the notebook (nbmake) checks, and the full suite.
- [ ] **Step 7: Commit.**

---

## Self-review notes

- **Engine unchanged:** Tasks 1-2 only translate arguments; the numeric path and `batch == lazy` are untouched. Task 3 adds one reducer + a string map. Task 4 deletes; Task 5 migrates.
- **Equivalence first:** every `freq=` and `agg="name"` is asserted equal to the `every=`/`count=` / `agg=<functor>` it replaces.
- **New code is small and dependency-free:** `_resample_freq_to_engine`, `_resample_datetime_freq` (numpy `timedelta64`), `_AGG_SYNONYMS`, one `ExpandingCount`.
- **Deferred (not this plan):** `(index, NaN)` heartbeats + retiring `advance()`/`dag.live()`; the deep docs/notebook rewrite + "Dag as a reusable user-defined function" framing (tracked in memory `project_docs_resample_dag_rewrite`); `median`/`quantile` buffering aggs; month/business datetime offsets.
- **Open decisions surfaced up front** (D-a count reducer, D-b datetime offset scope, D-c finance aliases) - confirm before Task 3 / Task 2.
