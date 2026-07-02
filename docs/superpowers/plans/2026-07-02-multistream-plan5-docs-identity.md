# Multi-stream Plan 5 — docs + identity matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the multi-stream foundation (layer 1): close the deferred hardening/coverage minors, add a systematic batch-vs-stream cross-mode identity test matrix, and write the user-facing `docs/multistream.md` documenting the principles, with cross-links from the existing policy pages.

**Architecture:** Three focused tasks — (1) small robustness + test-gap fixes carried from Plan 4 reviews; (2) a parametrized identity matrix proving batch and streaming forms agree across key types, series counts, and firing modes; (3) the documentation page plus cross-links. No new runtime features.

**Tech Stack:** Python, numpy, pytest, Markdown.

## Global Constraints

- **Cross-mode identity** is the property under test: batch `merge`/`combine_latest`/`dropna` and their streaming forms must produce byte-identical events (`np.testing.assert_array_equal`, NaN==NaN equal).
- **No silent data loss**: `split` with an explicit `n` too small must raise rather than drop events.
- Documentation must state the four load-bearing principles verbatim in intent: the order key + its two tiers; compute-is-shape-preserving vs combinators-change-cardinality; alignment-is-a-separate-layer; causality + cross-mode identity (no `bfill`/lookahead).
- No changes to runtime behavior of existing public functions except the `split` guard (a new error on misuse) and docstring clarifications.
- Never hand-edit `screamer/__init__.py` or version files. Docs build is not required to pass here (Markdown only), but keep `docs/` conventions.
- Tests: `poetry run pytest tests/test_streams_identity.py tests/test_streams_shape.py -v`.

---

## File Structure

- `screamer/streams.py` (modify) — `split` n-guard; docstring notes on `dropna` float64 normalization and `combine_latest`/`pace` metric-key expectations.
- `tests/test_streams_shape.py` (modify) — add `filter_iter`, 2-D `dropna_iter`, 2-D `filter`, and `split` under-`n` guard tests.
- `tests/test_streams_identity.py` (create) — parametrized batch-vs-stream identity matrix.
- `docs/multistream.md` (create) — the principles page.
- `docs/polymorphic_api.md`, `docs/nan_policy.md` (modify) — cross-links.

---

### Task 1: hardening + test-gap fills

**Files:**
- Modify: `screamer/streams.py`
- Test: `tests/test_streams_shape.py`

**Interfaces:**
- Changes: `split(keys, values, sources, n=None)` raises `ValueError` when an explicit `n` is smaller than `max(sources)+1` (would silently drop events). `dropna` docstring notes it normalizes surviving values to float64.

- [ ] **Step 1: Write the failing/added tests**

Append to `tests/test_streams_shape.py`:

```python
import pytest


def test_split_rejects_too_small_n():
    keys = np.array([1, 2, 3], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0])
    src = np.array([0, 1, 2], dtype=np.uint32)     # needs n >= 3
    with pytest.raises(ValueError):
        streams.split(keys, vals, src, n=2)         # would drop source 2 silently


def test_filter_iter_streaming():
    events = [(1, -1.0), (2, 2.0), (3, -3.0), (4, 4.0)]
    got = list(streams.filter_iter(events, lambda v: v > 0))
    assert got == [(2, 2.0), (4, 4.0)]


def test_dropna_iter_2d_rows():
    events = [(1, (1.0, 10.0)), (2, (np.nan, 20.0)), (3, (3.0, 30.0))]
    got_any = list(streams.dropna_iter(events, how="any"))
    assert [k for k, _ in got_any] == [1, 3]
    got_all = list(streams.dropna_iter(events, how="all"))
    assert [k for k, _ in got_all] == [1, 2, 3]      # no row is all-NaN


def test_filter_2d_row_predicate():
    keys = np.array([1, 2, 3], dtype=np.int64)
    vals = np.array([[1.0, 1.0], [5.0, 5.0], [2.0, 2.0]])
    gk, gv = streams.filter(keys, vals, lambda row: row.sum() > 5.0)
    np.testing.assert_array_equal(gk, np.array([2], dtype=np.int64))
    np.testing.assert_array_equal(gv, np.array([[5.0, 5.0]]))
```

- [ ] **Step 2: Run to verify the split guard test fails**

Run: `poetry run pytest tests/test_streams_shape.py::test_split_rejects_too_small_n -v`
Expected: FAIL — `split` currently truncates silently (no exception).

- [ ] **Step 3: Add the `split` guard and docstring notes**

In `screamer/streams.py`, in `split`, after computing `n`:

```python
    if n is None:
        n = int(sources.max()) + 1 if sources.size else 0
    elif sources.size and n <= int(sources.max()):
        raise ValueError(
            f"split: n={n} is too small for sources up to {int(sources.max())}; "
            "events would be dropped")
```

Add to `dropna`'s docstring a line: `Surviving values are returned as float64 (values are cast for the NaN test).` Add to `combine_latest`/`pace` docstrings (if not already present) a note that `pace` requires a metric (subtractable) key.

- [ ] **Step 4: Run tests**

Run: `poetry run pytest tests/test_streams_shape.py -v`
Expected: all PASS (existing + 4 new).

- [ ] **Step 5: Commit**

```bash
git add screamer/streams.py tests/test_streams_shape.py
git commit -m "harden(streams): split n-guard; filter_iter/2-D coverage; docstrings"
```

---

### Task 2: cross-mode identity matrix

**Files:**
- Test: `tests/test_streams_identity.py` (create)

**Interfaces:**
- Consumes: `streams.merge`/`merge_iter`, `combine_latest`/`combine_latest_iter`, `pace`, `dropna`/`dropna_iter`.
- Produces: a parametrized test module asserting batch == streaming across key dtypes (`int64`, `float64`), series counts (2, 3, 5), and firing modes (`when_all`, `on_any`).

- [ ] **Step 1: Write the matrix**

Create `tests/test_streams_identity.py`:

```python
import asyncio

import numpy as np
import pytest

from screamer import streams


def _make_series(n_series, size, dtype, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_series):
        if dtype == np.int64:
            k = np.sort(rng.integers(0, size * 4, size=size)).astype(np.int64)
        else:
            k = np.sort(rng.uniform(0, size * 4.0, size=size)).astype(np.float64)
        v = rng.standard_normal(size)
        out.append((k, v))
    return out


CONFIGS = [
    (n_series, dtype)
    for n_series in (2, 3, 5)
    for dtype in (np.int64, np.float64)
]


@pytest.mark.parametrize("n_series,dtype", CONFIGS)
def test_merge_batch_equals_stream(n_series, dtype):
    series = _make_series(n_series, 80, dtype, seed=100 + n_series)
    bk, bv, bs = streams.merge(*series)
    events = list(streams.merge_iter(*series))
    np.testing.assert_array_equal([e[0] for e in events], bk)
    np.testing.assert_array_equal([e[1] for e in events], bv)
    np.testing.assert_array_equal(np.array([e[2] for e in events], dtype=np.uint32), bs)


@pytest.mark.parametrize("n_series,dtype", CONFIGS)
@pytest.mark.parametrize("emit", ["when_all", "on_any"])
def test_combine_latest_batch_equals_stream(n_series, dtype, emit):
    series = _make_series(n_series, 80, dtype, seed=200 + n_series)
    bk, ba = streams.combine_latest(*series, emit=emit)
    events = list(streams.combine_latest_iter(*series, emit=emit))
    got_k = np.array([e[0] for e in events], dtype=bk.dtype)
    got_a = np.array([list(e[1]) for e in events], dtype=np.float64).reshape(len(events), n_series)
    np.testing.assert_array_equal(got_k, bk)
    np.testing.assert_array_equal(got_a, ba)


@pytest.mark.parametrize("n_series,dtype", CONFIGS)
def test_pace_infinite_equals_merge(n_series, dtype):
    series = _make_series(n_series, 60, dtype, seed=300 + n_series)
    bk, bv, bs = streams.merge(*series)

    async def drain():
        out = []
        async for e in streams.pace(*series, speed=float("inf")):
            out.append(e)
        return out

    events = asyncio.run(drain())
    np.testing.assert_array_equal([e[0] for e in events], bk)
    np.testing.assert_array_equal([e[1] for e in events], bv)


def test_dropna_batch_equals_stream_on_combine_output():
    # combine_latest on_any produces NaN warmup rows; dropna must remove the
    # same rows whether applied to the batch array or the event stream.
    series = _make_series(3, 50, np.int64, seed=42)
    bk, ba = streams.combine_latest(*series, emit="on_any")
    dk, dv = streams.dropna(bk, ba, how="any")

    events = list(streams.combine_latest_iter(*series, emit="on_any"))
    kept = list(streams.dropna_iter(events, how="any"))
    np.testing.assert_array_equal(np.array([k for k, _ in kept], dtype=bk.dtype), dk)
    stream_vals = np.array([list(v) for _, v in kept], dtype=np.float64).reshape(len(kept), 3)
    np.testing.assert_array_equal(stream_vals, dv)
```

- [ ] **Step 2: Run the matrix**

Run: `poetry run pytest tests/test_streams_identity.py -v`
Expected: all parametrized cases PASS. If a case fails, it is a real batch/stream divergence — STOP and report rather than weakening the assertion.

- [ ] **Step 3: Commit**

```bash
git add tests/test_streams_identity.py
git commit -m "test(streams): batch==stream identity matrix across dtypes/series/emit"
```

---

### Task 3: `docs/multistream.md` + cross-links

**Files:**
- Create: `docs/multistream.md`
- Modify: `docs/polymorphic_api.md`, `docs/nan_policy.md`

**Interfaces:**
- Produces: the user-facing principles page and two cross-links. No code.

- [ ] **Step 1: Create the documentation page**

Create `docs/multistream.md` with this content:

````markdown
# Streams, keys, and alignment

screamer's single-series operators (`RollingMean`, `RollingCorr`, …) assume
lockstep alignment: row `i` of one input pairs with row `i` of another. Real
multi-stream data breaks that assumption — feeds tick at different rates, arrive
out of step, and drop samples. The `screamer.streams` module adds a small,
composable layer for combining, splitting, filtering, and replaying streams that
do **not** tick together, while keeping every existing operator unchanged.

The whole design rests on four principles.

## 1. Every stream has an order key

A stream is a sequence of `(key, value)` events. The **key** is whatever you
already use to order data — a `datetime64` timestamp, an `int64` tick count, a
`float64` second, or, when you supply none, the **row number**. screamer only
ever *orders* and *compares* keys; it never interprets them.

Two capability tiers:

- **Comparable key** (can be ordered) — enough for `merge`, `combine_latest`,
  and backtest replay. Row numbers and any numeric key qualify.
- **Metric key** (differences are meaningful) — additionally enables wall-clock
  replay (`pace`), because a sleep duration only exists when key deltas convert
  to time.

Keys are numeric (`int64` or `double`), chosen from your array's dtype;
`datetime64` is carried losslessly as its underlying `int64`. The lockstep
behavior of the core operators is exactly the degenerate "no key → row number"
case, so nothing you already rely on changes.

## 2. Compute preserves shape; combinators change cardinality

| Layer | Cardinality | Examples |
|---|---|---|
| **Compute functors** | preserved (output length == input length) | `RollingMean`, `RollingCorr`, `FillNa`, `Ffill` |
| **Combinators** | may change it | `merge`, `combine_latest`, `dropna`, `filter`, `split`, `pace` |

Compute functors handle `NaN` internally via their `nan_policy` (see
[NaN policy](nan_policy.md)) and never add or drop rows. Combinators own all
time alignment and stream shaping. `dropna`/`filter`/`split` are the
cardinality-changing tools; `fillna`/`ffill` are shape-preserving and belong to
both worlds.

## 3. Alignment is a separate layer from computation

Time-aware combinators do the key handling and hand *aligned* data to the
unchanged compute functors. The idiom is:

```python
from screamer import combine_latest, RollingCorr

# Two async price feeds, each a (timestamps, prices) pair.
keys, aligned = combine_latest((t_a, p_a), (t_b, p_b))   # as-of latest-value join
corr = RollingCorr(20)(aligned[:, 0], aligned[:, 1])      # functor, untouched
```

`combine_latest` emits an aligned row whenever any input advances, carrying each
input's most recent value (forward-fill). `emit="when_all"` (default) waits
until every input is warm; `emit="on_any"` emits from the first event with
`NaN` for inputs not yet seen. Feed the aligned columns to any existing functor.

Other combinators:

- `merge(*series)` → one key-sorted, source-tagged stream (`keys, values, sources`).
- `split(keys, values, sources)` → the inverse of `merge`.
- `dropna(keys, values, how="any")` / `filter(keys, values, predicate)` → drop events.
- `pace(*series, speed=1.0)` → async replay; `speed=inf` is a max-speed backtest.

Every combinator has a streaming twin (`merge_iter`, `combine_latest_iter`,
`dropna_iter`, `filter_iter`) that yields events one at a time.

## 4. Causal, and identical across modes

- **Causal**: an output at key `t` depends only on events at keys `≤ t`. There
  is no backward-fill and no lookahead operator, ever.
- **Batch == streaming == replay**: the batch form and its streaming twin emit
  byte-identical event sequences; `pace` changes only *when* events are emitted,
  never their values or order. This is what lets you validate a pipeline on
  stored data and run the identical pipeline live. It is enforced by the
  identity matrix in `tests/test_streams_identity.py`.

## See also

- [Polymorphic API](polymorphic_api.md) — the single-series input/output
  contract; lockstep is the row-number-key special case of this page.
- [NaN policy](nan_policy.md) — how compute functors treat `NaN`; `ffill` is the
  same forward-fill carry that `combine_latest` uses.
````

- [ ] **Step 2: Add cross-links**

In `docs/polymorphic_api.md`, near the top (after the intro paragraph), add:

```markdown
> For combining, splitting, filtering, or replaying streams that do **not** tick
> together (different rates, async arrival, missing samples), see
> [Streams, keys, and alignment](multistream.md). The lockstep contract on this
> page is the degenerate "no time key → row number" case of that model.
```

In `docs/nan_policy.md`, near the top, add:

```markdown
> For dropping vs filling `NaN` **across streams** (`dropna`, `fillna`/`ffill`
> in the combinator layer), see [Streams, keys, and alignment](multistream.md).
> `ffill` there is the same forward-fill carry that `combine_latest` uses.
```

- [ ] **Step 3: Verify links and prose**

Run: `ls docs/multistream.md && grep -c "multistream.md" docs/polymorphic_api.md docs/nan_policy.md`
Expected: file exists; each cross-link file reports `1`.

- [ ] **Step 4: Commit**

```bash
git add docs/multistream.md docs/polymorphic_api.md docs/nan_policy.md
git commit -m "docs: add multistream principles page + cross-links"
```

---

## Self-Review

**1. Spec coverage (foundation completion):**
- Deferred minors closed: `split` n-guard (Task 1), `filter_iter`/2-D `dropna_iter`/2-D `filter` coverage (Task 1), `dropna` float64-normalization docstring (Task 1). ✓
- Cross-mode identity matrix across dtypes/series/emit, plus `pace`-inf==merge and dropna batch==stream (Task 2). ✓
- `docs/multistream.md` covering all four principles (order key + tiers, cardinality split, alignment layer, causality + identity) (Task 3). ✓
- Cross-links from `polymorphic_api.md` and `nan_policy.md` (Task 3). ✓
- This completes foundation layer 1. The DAG (layer 3) remains a separate future spec, explicitly out of scope here.

**2. Placeholder scan:** none — every step has concrete code, prose, or an exact command.

**3. Type consistency:** test helpers (`_make_series`, `CONFIGS`), the public function names, and the `split` guard signature match their definitions in earlier plans. The docs reference only shipped names (`merge`, `combine_latest`, `dropna`, `filter`, `split`, `pace`, and the `_iter` twins).

---

## Foundation complete

With Plan 5 merged, multi-stream foundation (layer 1) is done: the order-key model, the pull-source/push-graph C++ substrate, `merge`/`pace`, `combine_latest`, `filter`/`dropna`/`split`, the public API, the cross-mode identity guarantee, and the documentation. The computational DAG (layer 3) — a graph of these nodes materialized and executed as an all-C++ structure — is the next major effort and gets its own spec, built directly on this push-graph substrate.
