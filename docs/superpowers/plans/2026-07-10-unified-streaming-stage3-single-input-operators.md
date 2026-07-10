# Unified Streaming Stage 3: single-input stream operators as lazy callables

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make each single-input stream operator (`resample`, `dropna`, `filter`, `select`) ONE callable that dispatches on input type per Rule A - concrete data runs eager (container-preserving), a lazy iterator returns a lazy iterator - and retire the parallel `resample_iter` / `dropna_iter` / `filter_iter` / `select_iter` functions and the Python `_ResampleAccum` windowing class.

**Architecture:** `resample`'s lazy path routes through the Stage-2 lazy `Dag` (`_LazyDag`), driving the same C++ resample node as batch, so no Python windowing math survives (this is what retires `_ResampleAccum`). `dropna` / `filter` / `select` are element-wise (their lazy inputs can carry multi-column rows, which `_LazyDag`'s scalar-only `push_event` cannot represent), so their existing streaming generators are preserved verbatim - renamed to private `_*_lazy` helpers and reached through the operator's own dispatch. The public API becomes uniform (one callable, dispatch on input type) even though the lazy implementation differs by operator kind.

**Tech Stack:** Python 3 / numpy; pybind11 C++ engine (unchanged - no C++ edits in this stage); pytest.

## Global Constraints

- **Causality is a hard rule:** all operators causal, no lookahead; **batch and streaming MUST give identical results.** The batch call is the oracle: `op(array)` values equal `list(op(generator_of_(value,index)))` values, bit-for-bit.
- **Efficiency is the product:** treat dead allocations, redundant scans, and O(n) where O(1) is trivial as real defects.
- Keep the Python wrapper minimal; this stage adds no C++ and reuses the existing C++ resample node via the Stage-2 `_LazyDag`.
- **Rule A dispatch (container preservation):** a *lazy iterator* is an object with `__next__` that is NOT a `list`, `tuple`, `numpy.ndarray`, or `Stream`. Only such an input takes the lazy path; lists/tuples/arrays/Streams stay eager.
- No edits to version files. No em-dashes and no ` -- ` (double-hyphen) in prose, comments, or docstrings.
- Do NOT touch `merge` / `combine_latest` / `merge_iter` / `combine_latest_iter` (multi-input; deferred to Stage 3b). Do NOT touch `split` (it has no `_iter` twin). Do NOT change `resample`'s `every=` / `count=` signature (the `freq=` re-signature is Stage 4).
- The 2 pre-existing `tests/test_oscillators_hlc.py::TestBOP` failures are unrelated to this work; they predate the branch and must NOT be counted as introduced failures.

---

### Task 1: `resample` lazy dispatch via `_LazyDag`; retire `resample_iter` + `_ResampleAccum`

**Files:**
- Modify: `screamer/streams.py` (add `_is_lazy_stream` helper; rename `_resample_eager_via_cpp` to `_resample_via_cpp` taking a feed; add lazy branch to `resample`; delete `resample_iter` and `_ResampleAccum`; drop `resample_iter` from `__all__`)
- Test: `tests/test_streams_resample.py` (update the 2 existing `resample_iter` oracle tests; add new lazy tests)

**Interfaces:**
- Consumes (from Stage 2, unchanged): `Dag(inputs, outputs)` in `screamer/dag.py`; `dag(feed)` returns a batch `(values, index)` tuple when `feed` is a concrete `(vals, idx)` pair and a lazy iterator (yielding `(value, index)`) when `feed` is a generator of `(value, index)` events.
- Produces (used by Task 2): `_is_lazy_stream(x) -> bool` module-level helper in `screamer/streams.py`.

- [ ] **Step 1: Write the failing oracle test**

Add to `tests/test_streams_resample.py`:

```python
def test_resample_lazy_equals_batch_every():
    import numpy as np
    from screamer.streams import resample
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    idx = np.array([0, 1, 2, 10, 11, 20, 21])
    batch = resample(vals, idx, every=10, agg="mean")   # a Stream, unpackable
    bv, bk = batch.values, batch.index
    gen = ((float(v), int(k)) for v, k in zip(vals, idx))
    out = resample(gen, every=10, agg="mean")
    assert hasattr(out, "__next__") and not isinstance(out, tuple)
    rows = list(out)                                     # [(bar_value, bar_label), ...]
    np.testing.assert_allclose([r[0] for r in rows], np.asarray(bv), equal_nan=True)
    np.testing.assert_array_equal([r[1] for r in rows], np.asarray(bk))
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/test_streams_resample.py::test_resample_lazy_equals_batch_every -q`
Expected: FAIL - `resample(gen, ...)` currently feeds a generator into `Stream(values, index)` / `np.asarray`, producing a 0-d object array or a numeric error, not a lazy iterator.

- [ ] **Step 3: Add the `_is_lazy_stream` helper**

In `screamer/streams.py`, immediately after the `Stream` class definition (near the top, after line ~120), add:

```python
def _is_lazy_stream(x):
    """Rule A: a lazy stream is an iterator (has ``__next__``) that is NOT a
    concrete container (list, tuple, ndarray, or Stream). Generators and
    ``iter(...)`` qualify; concrete data does not."""
    return hasattr(x, "__next__") and not isinstance(x, (list, tuple, np.ndarray, Stream))
```

- [ ] **Step 4: Refactor `_resample_eager_via_cpp` into a feed-taking `_resample_via_cpp`**

Replace `_resample_eager_via_cpp` (currently `screamer/streams.py:674-692`) with:

```python
def _resample_via_cpp(feed, *, every, count, agg, origin, label, fill="skip"):
    """Run resample on the C++ engine via a one-node Dag, for batch OR lazy input.

    Builds the minimal ``Input -> Resample`` graph the Node regime already uses,
    then defers to ``dag(feed)``. Rule A on the Dag decides the mode: a concrete
    ``(vals, idx)`` pair runs batch and returns ``(out_values, out_index)``; a lazy
    iterator of ``(value, index)`` events returns a lazy iterator of
    ``(bar_value, bar_label)``. No Python windowing - all bucketing and NaN-ignore
    accumulation happens in the C++ core.
    """
    from .dag import Input, Dag
    src = Input("x")
    node = resample(src, every=every, count=count, agg=agg,
                    origin=origin, label=label, fill=fill)
    dag = Dag([src], [node])
    return dag(feed)
```

Then update the eager caller (currently `screamer/streams.py:968`) from:

```python
    out_v, out_idx = _resample_eager_via_cpp(
        vals, idx, every=every, count=count, agg=agg, origin=origin, label=label,
        fill=fill)
```

to:

```python
    out_v, out_idx = _resample_via_cpp(
        (vals, idx), every=every, count=count, agg=agg, origin=origin, label=label,
        fill=fill)
```

- [ ] **Step 5: Add the lazy branch to `resample`**

In `resample` (`screamer/streams.py:828`), insert the lazy branch AFTER the `if is_node(values):` block (ends at line ~933) and BEFORE `stream = values if isinstance(values, Stream) else Stream(values, index)` (line ~934):

```python
    if _is_lazy_stream(values):
        # Rule A: a lazy iterator of (value, index) events -> a lazy iterator of
        # (bar_value, bar_label). Drive the same C++ resample node as batch through
        # the Stage-2 lazy Dag; no Python windowing (this retires _ResampleAccum).
        if isinstance(agg, dict) or agg in ("ohlcv", "ohlcv2"):
            raise ValueError(
                "resample(<iterator>) supports string and functor scalar aggs "
                "only; dict and ohlcv/ohlcv2 aggs are eager-only. Materialize the "
                "stream to an array for those, or build the columns inside a Dag.")
        return _resample_via_cpp(values, every=every, count=count, agg=agg,
                                 origin=origin, label=label, fill=fill)
```

- [ ] **Step 6: Run the new test to confirm it passes**

Run: `python -m pytest tests/test_streams_resample.py::test_resample_lazy_equals_batch_every -q`
Expected: PASS

- [ ] **Step 7: Delete `resample_iter` and `_ResampleAccum`; drop from `__all__`**

- Delete the `resample_iter` function (`screamer/streams.py:1038-1093`).
- Delete the `_ResampleAccum` class (`screamer/streams.py:615-672`; delete through the blank line before the next def). Confirm no other reference remains: `grep -n "_ResampleAccum\|resample_iter" screamer/streams.py` returns nothing.
- Remove `"resample_iter",` from `__all__` (line ~23 in `screamer/streams.py`).

- [ ] **Step 8: Add the count-mode, NaN, functor-agg, dict-rejection, and laziness tests**

Replace the two existing oracle tests `test_resample_iter_matches_batch` (`tests/test_streams_resample.py:110`) and `test_resample_by_count_iter_and_nan` (line 120) - which call the now-deleted `resample_iter` - with versions that call `resample(gen, ...)`:

```python
def test_resample_lazy_equals_batch_count_and_nan():
    import numpy as np
    from screamer.streams import resample
    vals = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
    idx = np.array([0, 1, 2, 3, 4, 5])
    batch = resample(vals, idx, count=2, agg="mean")
    bv, bk = batch.values, batch.index
    gen = ((float(v), int(k)) for v, k in zip(vals, idx))
    rows = list(resample(gen, count=2, agg="mean"))
    np.testing.assert_allclose([r[0] for r in rows], np.asarray(bv), equal_nan=True)
    np.testing.assert_array_equal([r[1] for r in rows], np.asarray(bk))


def test_resample_lazy_functor_agg_equals_batch():
    import numpy as np
    from screamer import ExpandingSum
    from screamer.streams import resample
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    idx = np.array([0, 1, 2, 3, 4])
    batch = resample(vals, idx, count=2, agg=ExpandingSum())
    bv, bk = batch.values, batch.index
    gen = ((float(v), int(k)) for v, k in zip(vals, idx))
    rows = list(resample(gen, count=2, agg=ExpandingSum()))
    np.testing.assert_allclose([r[0] for r in rows], np.asarray(bv), equal_nan=True)
    np.testing.assert_array_equal([r[1] for r in rows], np.asarray(bk))


def test_resample_lazy_rejects_multicolumn_aggs():
    import numpy as np, pytest
    from screamer.streams import resample
    gen = ((float(v), int(v)) for v in range(4))
    with pytest.raises(ValueError):
        list(resample(gen, count=2, agg="ohlcv"))


def test_resample_lazy_is_lazy():
    """The lazy path must pull events on demand, not eagerly on construction."""
    from screamer.streams import resample
    pulled = []

    def spy():
        for i, v in enumerate([1.0, 2.0, 3.0, 4.0]):
            pulled.append(v)
            yield (v, i)

    it = resample(spy(), count=2, agg="last")
    assert pulled == []            # nothing consumed before first next()
    first = next(it)
    assert pulled == [1.0, 2.0]    # exactly the first bucket's events consumed
    assert first == (2.0, 0)       # count=2, last, label="left" -> value 2.0 at index 0
```

If either deleted test name was referenced elsewhere, `grep -rn "test_resample_iter_matches_batch\|test_resample_by_count_iter_and_nan" tests/` and remove stale references.

- [ ] **Step 9: Run the resample suite**

Run: `python -m pytest tests/test_streams_resample.py -q`
Expected: PASS (all resample tests green).

- [ ] **Step 10: Run the full suite**

Run: `python -m pytest -q`
Expected: `3930 passed` minus any net test-count change from the 2 replaced tests, plus the 4 new tests; 2 pre-existing `TestBOP` failures; 2 skipped. Zero NEW failures.

- [ ] **Step 11: Commit**

```bash
git add screamer/streams.py tests/test_streams_resample.py
git commit -m "feat(streams): resample dispatches on input type; lazy path via C++ Dag

resample(<iterator>) now returns a lazy iterator driven by the same C++
resample node as batch (via the Stage-2 lazy Dag), retiring resample_iter and
the Python _ResampleAccum windowing class. String and functor scalar aggs are
supported lazily; dict/ohlcv aggs stay eager-only.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `dropna` / `filter` / `select` lazy dispatch; retire their `*_iter`

**Files:**
- Modify: `screamer/streams.py` (rename `dropna_iter`/`filter_iter`/`select_iter` to private `_dropna_lazy`/`_filter_lazy`/`_select_lazy`; add a lazy branch to each public operator; drop the three `*_iter` names from `__all__`)
- Test: `tests/test_streams_shape.py`, `tests/test_streams_select.py` (update `*_iter` call sites to the operator; add oracle + laziness tests)

**Interfaces:**
- Consumes: `_is_lazy_stream(x)` from Task 1.
- Produces: `dropna(events)`, `filter(events, predicate)`, `select(events, columns)` return lazy iterators when `events` is a lazy iterator of `(value, index)`.

- [ ] **Step 1: Write failing oracle tests**

Add to `tests/test_streams_shape.py`:

```python
def test_dropna_lazy_equals_batch():
    import numpy as np
    from screamer.streams import dropna
    vals = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    idx = np.array([10, 11, 12, 13, 14])
    bv, bk = dropna(vals, idx)                          # raw -> (values, index)
    gen = ((float(v), int(k)) for v, k in zip(vals, idx))
    out = dropna(gen)
    assert hasattr(out, "__next__") and not isinstance(out, tuple)
    rows = list(out)
    np.testing.assert_allclose([r[0] for r in rows], np.asarray(bv))
    np.testing.assert_array_equal([r[1] for r in rows], np.asarray(bk))


def test_filter_lazy_equals_batch():
    import numpy as np
    from screamer.streams import filter as sfilter
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    idx = np.array([0, 1, 2, 3, 4])
    bv, bk = sfilter(vals, lambda v: v > 2, index=idx)
    gen = ((float(v), int(k)) for v, k in zip(vals, idx))
    rows = list(sfilter(gen, lambda v: v > 2))
    np.testing.assert_allclose([r[0] for r in rows], np.asarray(bv))
    np.testing.assert_array_equal([r[1] for r in rows], np.asarray(bk))
```

Add to `tests/test_streams_select.py`:

```python
def test_select_lazy_equals_batch():
    import numpy as np
    from screamer.streams import select
    vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    idx = np.array([0, 1, 2])
    bv, bk = select(vals, 1, index=idx)                 # scalar column -> 1-D
    gen = ((row.tolist(), int(k)) for row, k in zip(vals, idx))
    rows = list(select(gen, 1))
    np.testing.assert_allclose([r[0] for r in rows], np.asarray(bv))
    np.testing.assert_array_equal([r[1] for r in rows], np.asarray(bk))
```

- [ ] **Step 2: Run to confirm they fail**

Run: `python -m pytest tests/test_streams_shape.py::test_dropna_lazy_equals_batch tests/test_streams_shape.py::test_filter_lazy_equals_batch tests/test_streams_select.py::test_select_lazy_equals_batch -q`
Expected: FAIL (generators currently hit `np.asarray` in the eager path).

- [ ] **Step 3: Rename the three `*_iter` generators to private helpers**

In `screamer/streams.py`, rename (body UNCHANGED, only the name and the `_iter` -> `_lazy` suffix):
- `def dropna_iter(events, how="any"):` (line 479) -> `def _dropna_lazy(events, how="any"):`
- `def filter_iter(events, predicate):` (line 495) -> `def _filter_lazy(events, predicate):`
- `def select_iter(events, columns):` (line 588) -> `def _select_lazy(events, columns):`

Update each error message inside them that names the old function (e.g. `select_iter: column ...` -> `select: column ...`) so raised messages match the public entry point.

- [ ] **Step 4: Add the lazy branch to each public operator**

In `dropna` (`screamer/streams.py:429`), after the `how` validation and the `if is_node(values):` block (line 443), before `regime = ...` (line 444), add:

```python
    if _is_lazy_stream(values):
        return _dropna_lazy(values, how)
```

In `filter` (`screamer/streams.py:456`), after the `if is_node(values):` block (line 469), before `regime = ...` (line 470), add:

```python
    if _is_lazy_stream(values):
        return _filter_lazy(values, predicate)
```

In `select` (`screamer/streams.py:554`), after the `if is_node(values):` block (line 566), before `regime = ...` (line 567), add:

```python
    if _is_lazy_stream(values):
        return _select_lazy(values, columns)
```

- [ ] **Step 5: Drop the three `*_iter` names from `__all__`**

In `screamer/streams.py`, remove `"dropna_iter",`, `"filter_iter",`, and `"select_iter",` from `__all__` (lines ~19-21). Keep the base names `"dropna"`, `"filter"`, `"select"`.

- [ ] **Step 6: Update existing `*_iter` call sites in tests**

`grep -rn "dropna_iter\|filter_iter\|select_iter" tests/` and update each. The direct-value tests in `tests/test_streams_shape.py` (`test_dropna_iter_matches_batch` line 134, `test_filter_iter_streaming` line 156, `test_filter_iter_positional` line 162) and `tests/test_streams_select.py` (`test_select_iter_matches_batch` line 111) should call the operator on a generator instead of the `*_iter` function. Example for `test_select_iter_matches_batch`: replace `select_iter(events, cols)` with `select(events, cols)` where `events` is a generator; keep the same expected values (do NOT weaken assertions).

- [ ] **Step 7: Add laziness tests**

Add to `tests/test_streams_shape.py`:

```python
def test_dropna_lazy_is_lazy():
    from screamer.streams import dropna
    pulled = []

    def spy():
        for i, v in enumerate([1.0, 2.0, 3.0]):
            pulled.append(v)
            yield (v, i)

    it = dropna(spy())
    assert pulled == []
    first = next(it)
    assert pulled == [1.0]
    assert first == (1.0, 0)
```

- [ ] **Step 8: Run the affected suites**

Run: `python -m pytest tests/test_streams_shape.py tests/test_streams_select.py -q`
Expected: PASS.

- [ ] **Step 9: Run the full suite**

Run: `python -m pytest -q`
Expected: Zero new failures; 2 pre-existing `TestBOP`; 2 skipped.

- [ ] **Step 10: Commit**

```bash
git add screamer/streams.py tests/test_streams_shape.py tests/test_streams_select.py
git commit -m "feat(streams): dropna/filter/select dispatch on input type

Each operator now returns a lazy iterator when fed a lazy iterator of
(value, index) events; the streaming generators are preserved as private
_*_lazy helpers reached through the operator's dispatch. Retires dropna_iter,
filter_iter, and select_iter from the public API.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: exports, devtools, docs, docstrings, migration table

**Files:**
- Modify: `screamer/__init__.py` (drop the four `*_iter` re-exports)
- Modify: `devtools/build_function_index.py` (drop `*_iter` from `STREAM_NAMES`)
- Modify: doc pages and docstrings under `docs/` that reference the retired names; add a migration note
- Test: `tests/test_streams_public_api.py` (and any doc-coverage test) - update expectations

**Interfaces:**
- Consumes: the public API after Tasks 1-2 (no `resample_iter`/`dropna_iter`/`filter_iter`/`select_iter`/`_ResampleAccum`).

- [ ] **Step 1: Update `screamer/__init__.py`**

Remove `resample_iter`, `dropna_iter`, `filter_iter`, and `select_iter` from the streams import (line ~11) and from the flat `__all__` (line ~19). Leave `merge_iter` and `combine_latest_iter` in place (Stage 3b owns those). Verify import: `python -c "import screamer; assert not hasattr(screamer, 'resample_iter')"`.

- [ ] **Step 2: Update `devtools/build_function_index.py`**

`grep -n "resample_iter\|dropna_iter\|filter_iter\|select_iter" devtools/build_function_index.py` and remove those four names from the `STREAM_NAMES` list (keep `merge_iter`/`combine_latest_iter`).

- [ ] **Step 3: Update the public-API test**

Run `python -m pytest tests/test_streams_public_api.py -q` and update any assertion that enumerates the four retired names (it should assert they are ABSENT now, or simply not list them). Do not remove coverage of the surviving public names.

- [ ] **Step 4: Update docs and docstrings**

`grep -rln "resample_iter\|dropna_iter\|filter_iter\|select_iter\|_ResampleAccum" docs/ screamer/` (exclude `docs/_build/`, which is generated). For each source doc page or docstring, replace the retired-function usage with the operator-on-an-iterator form. Add a short migration table to the DAG/streaming doc page (e.g. `docs/dag.md` or the streams reference page):

```
| Old (removed)                     | New                              |
|-----------------------------------|----------------------------------|
| resample_iter(events, every=W)    | resample(events, every=W)        |
| dropna_iter(events)               | dropna(events)                   |
| filter_iter(events, pred)         | filter(events, pred)             |
| select_iter(events, cols)         | select(events, cols)             |
```

Note in prose: feed a lazy iterator of `(value, index)` events to get a lazy iterator back; feed arrays/Streams to get the eager result (Rule A). No em-dashes / no ` -- `.

- [ ] **Step 5: Rebuild docs if the project builds them in CI**

If a doc build or doc-coverage test exists (`grep -rn "test_doc_coverage" tests/`), run it: `python -m pytest tests/test_doc_coverage.py -q` (or the project's `make docs`), and fix any dangling reference it flags.

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest -q`
Expected: Zero new failures; 2 pre-existing `TestBOP`; 2 skipped.

- [ ] **Step 7: Commit**

```bash
git add screamer/__init__.py devtools/build_function_index.py docs tests/test_streams_public_api.py
git commit -m "docs(streams): retire *_iter from public API, exports, and docs

Drop resample_iter/dropna_iter/filter_iter/select_iter from __init__ exports,
the function index, and the docs; add an old->new migration table. The single
callable dispatches on input type (Rule A).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-review notes (for the implementer and reviewer)

- **Oracle everywhere:** every operator's lazy path is asserted equal to its batch result on values AND index, not row counts. Do not weaken any existing assertion when migrating a `*_iter` call site.
- **Laziness:** at least `resample` and `dropna` have an explicit pull-count test proving the driver consumes on demand, not eagerly on construction.
- **Rule A negative paths:** `resample(iter)` rejects dict/ohlcv aggs; concrete inputs (arrays/lists/Streams) still take the eager path unchanged.
- **No behavior change to eager paths:** `_resample_via_cpp((vals, idx), ...)` must return the identical `(out_values, out_index)` the old `_resample_eager_via_cpp` did; the only change is that it now also accepts a lazy feed.
- **Scope discipline:** `merge`, `combine_latest`, `merge_iter`, `combine_latest_iter`, and `split` are untouched. `resample`'s `every=`/`count=` signature is unchanged.
