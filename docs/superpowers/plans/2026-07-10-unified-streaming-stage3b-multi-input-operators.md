# Unified Streaming Stage 3b: multi-input stream operators as lazy callables

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the two multi-input stream operators `combine_latest` and `merge` dispatch on input type per Rule A - fed concrete data they run eager (unchanged), fed lazy iterators they return a lazy iterator - and retire the parallel `combine_latest_iter` and `merge_iter` functions. A lazy source with no index gets a per-source arrival counter as its index (so positional works lazily, symmetric with eager positional arrays).

**Architecture:** A shared classifier decides whether the N lazy sources are all positional (bare scalar items -> per-source counter) or all indexed ((value, index) items); a mix raises, matching the eager operators. `combine_latest` then splits by kind: positional is strict lockstep (a Python `zip` with an equal-length check - pure tupling, no numeric logic), and indexed is as-of alignment driven by the existing C++ `combine_latest` node through the Stage-2 lazy `Dag` (`_LazyDag`). `merge` is a small k-way merge by index (the counter when positional), pure input routing with no numeric logic. Each lazy sub-path mirrors its eager counterpart exactly, so `batch == lazy` holds by construction.

**Tech Stack:** Python 3 / numpy; the existing pybind11 C++ engine (unchanged - no C++ edits); pytest.

## Global Constraints

- **Causality is a hard rule:** causal, no lookahead; **batch and streaming MUST give identical results.** The batch call is the oracle: `op(arrays)` equals `list(op(generators))` in values, index, and (for merge) source, event for event.
- **Rule A dispatch:** a *lazy iterator* is an object with `__next__` that is NOT a `list`, `tuple`, `numpy.ndarray`, or `Stream`. Only such inputs take the lazy path. Reuse the existing module-level `_is_lazy_stream(x)` helper in `screamer/streams.py`; do NOT redefine it.
- **No-index means the arrival counter:** a lazy source that yields bare scalars is positional; internally each such source is numbered `0, 1, 2, ...` (per source) to order/align it. Positional OUTPUT keeps index `None` (matching eager positional), so the counter never leaks into results.
- **Uniform per call:** all sources positional, or all indexed. A mix raises `ValueError` (combine_latest and merge already require this eagerly). A mix of lazy and concrete inputs raises `TypeError`.
- **Positional `combine_latest` requires equal length** (aligned clocks), exactly as the eager operator does; unequal-length positional sources raise `ValueError`. (Decision (a): mirror eager; do NOT relax eager to as-of.)
- Efficiency is the product: no dead allocations or redundant scans on the hot path.
- No version-file edits. No em-dashes and no ` -- ` (double-hyphen) in code, comments, or docstrings.
- Scope: do NOT touch `resample`/`dropna`/`filter`/`select` (Stage 3, done) or their lazy paths, `split`, or any C++ code. The single-input operators already dispatch; this stage only adds `combine_latest` and `merge`.
- The 2 pre-existing `tests/test_oscillators_hlc.py::TestBOP` failures are unrelated and must NOT be counted as introduced.

---

### Task 1: shared source classifier + `combine_latest` lazy dispatch; retire `combine_latest_iter`

**Files:**
- Modify: `screamer/streams.py` (add `_classify_lazy_sources`; add `_combine_latest_zip_lazy` and `_combine_latest_asof_lazy`; add a lazy branch to `combine_latest`; delete `combine_latest_iter`; drop it from `__all__`)
- Modify: `screamer/__init__.py` (drop `combine_latest_iter` from import + `__all__`)
- Test: `tests/test_streams_combine_latest.py`, `tests/test_streams_identity.py`

**Interfaces:**
- Consumes: `_is_lazy_stream(x)` (exists); the Stage-2 lazy `Dag` (`dag(*generators)` returns a `_LazyDag` yielding `(row_tuple, index)` for a single multi-column output).
- Produces (used by Task 2): `_classify_lazy_sources(values, who) -> (positional: bool, sources: list_of_iterators)` and the module-level sentinel `_EMPTY`.

- [ ] **Step 1: Write the failing oracle tests**

Add to `tests/test_streams_combine_latest.py`:

```python
def test_combine_latest_lazy_indexed_equals_batch():
    import numpy as np
    from screamer.streams import combine_latest
    av, ak = np.array([10.0, 20.0, 30.0]), np.array([1, 2, 4])
    bv, bk = np.array([1.0, 2.0, 3.0]),   np.array([1, 3, 4])
    brows, bidx = combine_latest(av, bv, index=[ak, bk])       # batch oracle
    ga = ((float(v), int(k)) for v, k in zip(av, ak))
    gb = ((float(v), int(k)) for v, k in zip(bv, bk))
    out = combine_latest(ga, gb)
    assert hasattr(out, "__next__") and not isinstance(out, tuple)
    rows = list(out)                                            # [((a,b), index), ...]
    np.testing.assert_allclose([list(r[0]) for r in rows], np.asarray(brows))
    np.testing.assert_array_equal([r[1] for r in rows], np.asarray(bidx))


def test_combine_latest_lazy_positional_equals_batch():
    import numpy as np
    from screamer.streams import combine_latest
    a = [10.0, 20.0, 30.0]
    b = [1.0, 2.0, 3.0]
    brows, bidx = combine_latest(np.array(a), np.array(b))      # aligned clocks, index None
    out = list(combine_latest((x for x in a), (x for x in b)))  # bare-value sources
    np.testing.assert_allclose([list(r[0]) for r in out], np.asarray(brows))
    assert all(r[1] is None for r in out)                       # positional -> None index
    assert bidx is None


def test_combine_latest_lazy_positional_unequal_raises():
    import pytest
    from screamer.streams import combine_latest
    out = combine_latest((x for x in [1.0, 2.0, 3.0]), (x for x in [10.0, 20.0]))
    with pytest.raises(ValueError):
        list(out)                                              # error surfaces at exhaustion


def test_combine_latest_lazy_mixed_sources_raise():
    import numpy as np, pytest
    from screamer.streams import combine_latest
    # mix positional (bare) and indexed ((v,k)) lazy sources -> ValueError
    with pytest.raises(ValueError):
        list(combine_latest((x for x in [1.0, 2.0]),
                            ((float(v), int(k)) for v, k in [(1.0, 0), (2.0, 1)])))
    # mix lazy and concrete -> TypeError
    with pytest.raises(TypeError):
        combine_latest((x for x in [1.0, 2.0]), np.array([1.0, 2.0]))
```

- [ ] **Step 2: Run to confirm they fail**

Run: `python -m pytest tests/test_streams_combine_latest.py -k lazy -q`
Expected: FAIL (lazy generators currently fall into the eager path and error in `_to_streams`/`np.asarray`).

- [ ] **Step 3: Add the sentinel and the classifier**

In `screamer/streams.py`, near `_is_lazy_stream` (after it), add:

```python
_EMPTY = object()   # sentinel: a lazy source produced no first item


def _classify_lazy_sources(values, who):
    """Classify N lazy sources as positional or indexed and return
    (positional, sources).

    Peeks each source's first item: a bare scalar item means positional (its
    index is a per-source arrival counter), a ``(value, index)`` 2-tuple means
    indexed. Every source must be uniformly positional or indexed; a mix raises
    ValueError (matching the eager operators). ``sources`` are the original
    iterators re-chained with their peeked head, so no event is lost.
    """
    import itertools
    iters = [iter(v) for v in values]
    heads, sources = [], []
    for it in iters:
        try:
            head = next(it)
        except StopIteration:
            head = _EMPTY
        heads.append(head)
        sources.append(it if head is _EMPTY else itertools.chain([head], it))
    kinds = {isinstance(h, tuple) and len(h) == 2 for h in heads if h is not _EMPTY}
    if len(kinds) > 1:
        raise ValueError(
            f"{who}: cannot mix positional (bare-value) and indexed "
            "((value, index)) lazy sources; give every source an index, or none")
    indexed = kinds.pop() if kinds else False   # all-empty -> positional (no rows)
    return (not indexed), sources
```

- [ ] **Step 4: Add the two lazy combine_latest generators**

In `screamer/streams.py`, add (just before `combine_latest`):

```python
def _combine_latest_zip_lazy(sources):
    """Positional (aligned-clock) combine_latest over lazy bare-value sources.

    Strict lockstep: every source must have equal length, matching the eager
    positional contract. Yields ``(row_tuple, None)``. Raises ValueError at the
    first length mismatch (lengths are unknowable until the streams run out).
    """
    import itertools
    for tup in itertools.zip_longest(*sources, fillvalue=_EMPTY):
        if any(x is _EMPTY for x in tup):
            raise ValueError(
                "combine_latest: positional (no-index) lazy sources must have "
                "equal length (aligned clocks); source lengths differ")
        yield tuple(float(x) for x in tup), None


def _combine_latest_asof_lazy(sources, emit):
    """Indexed combine_latest over lazy (value, index) sources: as-of alignment
    driven by the C++ combine_latest node through the Stage-2 lazy Dag. Yields
    ``(row_tuple, index)`` - byte-identical to the batch combine_latest."""
    from .dag import Input, Dag
    ins = [Input(f"_cl{i}") for i in range(len(sources))]
    dag = Dag(inputs=ins, outputs=[combine_latest(*ins, emit=emit)])
    yield from dag(*sources)
```

- [ ] **Step 5: Add the lazy branch to `combine_latest`**

In `combine_latest` (`screamer/streams.py:329`), insert AFTER the `if any(is_node(v) ...)` graph block (ends ~line 351) and BEFORE `regime = _regime(values)` (line 352):

```python
    if values and all(_is_lazy_stream(v) for v in values):
        if func is not None:
            raise ValueError(
                "combine_latest(func=...) is not supported for lazy iterator "
                "inputs; apply the function to the aligned output instead")
        positional, sources = _classify_lazy_sources(values, "combine_latest")
        if positional:
            return _combine_latest_zip_lazy(sources)
        return _combine_latest_asof_lazy(sources, emit)
    if any(_is_lazy_stream(v) for v in values):
        raise TypeError(
            "combine_latest: cannot mix lazy iterator and concrete inputs; pass "
            "all generators or all arrays/Streams")
```

- [ ] **Step 6: Run the new tests**

Run: `python -m pytest tests/test_streams_combine_latest.py -k lazy -q`
Expected: PASS.

- [ ] **Step 7: Delete `combine_latest_iter`; drop from exports**

- Delete `combine_latest_iter` (`screamer/streams.py:364-386`). Confirm `grep -n "combine_latest_iter" screamer/streams.py` returns nothing outside a migration comment.
- Remove `"combine_latest_iter",` from `__all__` in `screamer/streams.py`.
- In `screamer/__init__.py`, remove `combine_latest_iter` from the `from .streams import (...)` line and from `__all__`.

- [ ] **Step 8: Migrate the existing `combine_latest_iter` identity test**

In `tests/test_streams_identity.py`, `test_combine_latest_batch_equals_stream` (line 63) calls `combine_latest_iter`. Rewrite it to build `(value, index)` generators from the same arrays and call `combine_latest(*gens)`, asserting the lazy rows and index equal the batch `combine_latest(*arrays, index=...)` result. Do NOT weaken the assertions; keep the same parametrization over `n_series`/`dtype`/`emit` where the indexed path applies (positional-equal-length can stay a separate case).

- [ ] **Step 9: Add a laziness test**

Add to `tests/test_streams_combine_latest.py`:

```python
def test_combine_latest_lazy_is_lazy():
    """Construction pulls nothing; the first next() pulls one head per source."""
    from screamer.streams import combine_latest
    pulled = {"a": [], "b": []}

    def spy(name, items):
        for i, v in enumerate(items):
            pulled[name].append(v)
            yield (float(v), i)

    it = combine_latest(spy("a", [1.0, 2.0]), spy("b", [10.0, 20.0]))
    assert pulled == {"a": [], "b": []}      # nothing before first next()
    next(it)
    assert pulled["a"] == [1.0] and pulled["b"] == [10.0]   # one head per source
```

- [ ] **Step 10: Run the affected suites and the full suite**

Run: `python -m pytest tests/test_streams_combine_latest.py tests/test_streams_identity.py -q` then `python -m pytest -q`
Expected: affected files PASS; full suite has zero new failures (2 pre-existing `TestBOP`, 2 skipped).

- [ ] **Step 11: Commit**

```bash
git add screamer/streams.py screamer/__init__.py tests/test_streams_combine_latest.py tests/test_streams_identity.py
git commit -m "feat(streams): combine_latest dispatches on input type; lazy path

Lazy inputs return a lazy iterator: positional (bare-value) sources align in
strict lockstep (equal length required, matching eager); indexed ((value,index))
sources align as-of via the C++ combine_latest node through the Stage-2 lazy Dag.
A no-index source is numbered by a per-source arrival counter. Retires
combine_latest_iter.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `merge` lazy dispatch; retire `merge_iter`

**Files:**
- Modify: `screamer/streams.py` (add `_merge_lazy`; add a lazy branch to `merge`; delete `merge_iter`; drop it from `__all__`)
- Modify: `screamer/__init__.py` (drop `merge_iter`)
- Test: `tests/test_streams_merge.py`, `tests/test_streams_identity.py`

**Interfaces:**
- Consumes: `_classify_lazy_sources` and `_EMPTY` (Task 1); `_is_lazy_stream` (exists).

- [ ] **Step 1: Write the failing oracle tests**

Add to `tests/test_streams_merge.py`:

```python
def test_merge_lazy_indexed_equals_batch():
    import numpy as np
    from screamer.streams import merge
    av, ak = np.array([1.0, 2.0, 3.0]), np.array([0, 2, 4])
    bv, bk = np.array([10.0, 20.0]),     np.array([1, 3])
    mvals, msrc, midx = merge(av, bv, index=[ak, bk])          # batch oracle
    ga = ((float(v), int(k)) for v, k in zip(av, ak))
    gb = ((float(v), int(k)) for v, k in zip(bv, bk))
    out = merge(ga, gb)
    assert hasattr(out, "__next__") and not isinstance(out, tuple)
    rows = list(out)                                           # [(value, index, source), ...]
    np.testing.assert_allclose([r[0] for r in rows], np.asarray(mvals))
    np.testing.assert_array_equal([r[1] for r in rows], np.asarray(midx))
    np.testing.assert_array_equal([r[2] for r in rows], np.asarray(msrc))


def test_merge_lazy_positional_equals_batch_unequal_lengths():
    import numpy as np
    from screamer.streams import merge
    a = [1.0, 2.0, 3.0]
    b = [10.0, 20.0]
    mvals, msrc, midx = merge(np.array(a), np.array(b))        # positional, index None
    rows = list(merge((x for x in a), (x for x in b)))         # bare-value sources
    np.testing.assert_allclose([r[0] for r in rows], np.asarray(mvals))
    assert all(r[1] is None for r in rows)                     # positional -> None index
    np.testing.assert_array_equal([r[2] for r in rows], np.asarray(msrc))
    assert midx is None
```

- [ ] **Step 2: Run to confirm they fail**

Run: `python -m pytest tests/test_streams_merge.py -k lazy -q`
Expected: FAIL.

- [ ] **Step 3: Add `_merge_lazy`**

In `screamer/streams.py`, add (just before `merge`):

```python
def _merge_lazy(sources, positional):
    """K-way merge of lazy sources by index, ties broken by source order.

    Positional sources are ordered by a per-source arrival counter (row number)
    and yield index None; indexed sources ((value, index)) order by their index
    and yield it. Yields ``(value, index_or_None, source)`` - byte-identical to
    the eager merge, event for event.
    """
    iters = [iter(s) for s in sources]
    counters = [0] * len(iters)
    heads = [None] * len(iters)         # (order_key, value, out_index) or None

    def pull(i):
        try:
            item = next(iters[i])
        except StopIteration:
            return None
        if positional:
            key = counters[i]
            counters[i] += 1
            return (key, float(item), None)
        value, index = item
        return (int(index), float(value), int(index))

    for i in range(len(iters)):
        heads[i] = pull(i)
    while True:
        best = -1
        for i, h in enumerate(heads):           # smallest key wins; ties -> lower i
            if h is not None and (best < 0 or h[0] < heads[best][0]):
                best = i
        if best < 0:
            return
        _, value, out_index = heads[best]
        yield value, out_index, best
        heads[best] = pull(best)
```

- [ ] **Step 4: Add the lazy branch to `merge`**

In `merge` (`screamer/streams.py:286`), insert AFTER the `if any(is_node(v) ...)` raise block (ends ~line 301) and BEFORE `kind, idx_list, ... = _merge_to_indexed(...)` (line 302):

```python
    if values and all(_is_lazy_stream(v) for v in values):
        positional, sources = _classify_lazy_sources(values, "merge")
        return _merge_lazy(sources, positional)
    if any(_is_lazy_stream(v) for v in values):
        raise TypeError(
            "merge: cannot mix lazy iterator and concrete inputs; pass all "
            "generators or all arrays/Streams")
```

- [ ] **Step 5: Run the new tests**

Run: `python -m pytest tests/test_streams_merge.py -k lazy -q`
Expected: PASS.

- [ ] **Step 6: Delete `merge_iter`; drop from exports**

- Delete `merge_iter` (`screamer/streams.py:308-326`). Confirm `grep -n "merge_iter" screamer/streams.py` returns nothing outside a migration comment.
- Remove `"merge_iter",` from `__all__` in `screamer/streams.py`.
- In `screamer/__init__.py`, remove `merge_iter` from the import and `__all__`.

- [ ] **Step 7: Migrate the existing `merge_iter` tests**

`grep -rn "merge_iter" tests/` and update each. `tests/test_streams_identity.py` has `test_merge_batch_equals_stream_indexed` (line 30) and `test_merge_batch_equals_stream_positional` (line 47); `tests/test_streams_merge.py` has `test_merge_iter_indexed_matches_batch` (line 102). Rewrite them to build generators (indexed: `(value, index)`; positional: bare values) and call `merge(*gens)`, asserting values, index, and source equal the batch `merge` oracle. Do NOT weaken assertions or drop parametrization.

- [ ] **Step 8: Add a laziness test**

Add to `tests/test_streams_merge.py`:

```python
def test_merge_lazy_is_lazy():
    from screamer.streams import merge
    pulled = {"a": [], "b": []}

    def spy(name, items):
        for i, v in enumerate(items):
            pulled[name].append(v)
            yield (float(v), i)

    it = merge(spy("a", [1.0, 2.0]), spy("b", [10.0, 20.0]))
    assert pulled == {"a": [], "b": []}
    next(it)
    assert pulled["a"] == [1.0] and pulled["b"] == [10.0]   # one head per source
```

- [ ] **Step 9: Run affected + full suite**

Run: `python -m pytest tests/test_streams_merge.py tests/test_streams_identity.py -q` then `python -m pytest -q`
Expected: zero new failures.

- [ ] **Step 10: Commit**

```bash
git add screamer/streams.py screamer/__init__.py tests/test_streams_merge.py tests/test_streams_identity.py
git commit -m "feat(streams): merge dispatches on input type; lazy k-way merge

Lazy inputs return a lazy iterator of (value, index, source): a k-way merge by
index (a per-source arrival counter when positional, yielding index None), ties
broken by source order - identical to the eager merge. Retires merge_iter.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: exports, devtools, docs, migration table

**Files:**
- Modify: `devtools/build_function_index.py` (drop `merge_iter`, `combine_latest_iter` from `STREAM_NAMES`)
- Modify: `docs/functions_streams/merge.md`, `docs/functions_streams/combine_latest.md` (remove `covers:` + `.. autofunction::` for the retired names; add a Rule A sentence)
- Modify: `docs/multistream.md` (update prose; extend the migration table)
- Regenerate: `screamer/data/help.json`, `docs/function_index.txt`
- Test: `tests/test_streams_public_api.py` and the doc-coverage tests

**Interfaces:**
- Consumes: the public API after Tasks 1-2 (no `merge_iter`/`combine_latest_iter`).

- [ ] **Step 1: Verify exports already clean**

`python -c "import screamer; assert not any(hasattr(screamer, n) for n in ['merge_iter','combine_latest_iter'])"` (Tasks 1-2 removed them from `__init__.py`). If it fails, fix `screamer/__init__.py`.

- [ ] **Step 2: `devtools/build_function_index.py`**

Remove `"merge_iter"` and `"combine_latest_iter"` from `STREAM_NAMES`. After this, ALL `*_iter` names are gone from that list.

- [ ] **Step 3: Doc pages**

In `docs/functions_streams/merge.md` and `docs/functions_streams/combine_latest.md`: delete the retired name from the `covers:` frontmatter (remove the now-empty `covers:` key if it becomes empty) and delete the `.. autofunction:: screamer.streams.<name>_iter` directive. Add one sentence: feeding lazy iterators returns a lazy iterator (a no-index source is numbered by a per-source arrival counter); feeding arrays/Streams returns the eager result (Rule A).

- [ ] **Step 4: `docs/multistream.md`**

Update any prose that presents `merge_iter`/`combine_latest_iter` as separate functions to the unified model. Extend the migration table with:

```
| combine_latest_iter(a, b) | combine_latest(a, b) |
| merge_iter(a, b)          | merge(a, b)          |
```

- [ ] **Step 5: Regenerate derived artifacts**

Run from the repo root with the repo importable:
- `PYTHONPATH=$(pwd) python devtools/build_help_registry.py` (writes `screamer/data/help.json`)
- `PYTHONPATH=$(pwd) python devtools/build_function_index.py` (writes `docs/function_index.txt`)

`build_help_registry.py` does `getattr(screamer, name)` on each page's `covers:` list, so it crashes if a page still names a retired function - that means Step 3 is incomplete. After regenerating, confirm `grep -c "merge_iter\|combine_latest_iter" screamer/data/help.json docs/function_index.txt` returns 0 for both (retired names may remain ONLY in the `docs/multistream.md` migration table).

- [ ] **Step 6: Update the public-API test**

Run `python -m pytest tests/test_streams_public_api.py -q`; update the `PUBLIC` set / assertions so the retired names are absent (do not drop coverage of surviving names).

- [ ] **Step 7: Doc + full suite**

Run: `python -m pytest tests/test_doc_coverage.py tests/test_build_help_registry.py -q` then `python -m pytest -q`
Expected: doc/help tests PASS; full suite zero new failures.

- [ ] **Step 8: Commit**

```bash
git add devtools/build_function_index.py docs screamer/data/help.json tests/test_streams_public_api.py
git commit -m "docs(streams): retire merge_iter/combine_latest_iter from API and docs

Drop the last two *_iter names from the function index, doc pages, and
help.json; extend the old->new migration table. All six stream operators now
dispatch on input type (Rule A); no *_iter functions remain.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-review notes (for implementer and reviewer)

- **Oracle everywhere:** every lazy test asserts values, index, and (merge) source against the batch result, event for event - never counts.
- **Positional index is None:** positional lazy output yields index `None` (not the internal counter), so it is byte-identical to eager positional (which returns index `None`). The counter is an internal ordering device only.
- **Equal-length combine_latest:** positional lazy `combine_latest` raises `ValueError` on unequal lengths (surfacing at stream exhaustion), matching the eager equal-length contract. This is Decision (a): eager is NOT relaxed to as-of.
- **Uniformity:** all-positional or all-indexed per call; a mix raises `ValueError`. Lazy mixed with concrete raises `TypeError`.
- **Laziness:** construction pulls nothing; the first `next()` pulls exactly one head per source (tested for both operators).
- **Scope discipline:** `resample`/`dropna`/`filter`/`select`/`split` and all C++ untouched. After this stage, no `*_iter` stream function remains anywhere in the public API.
