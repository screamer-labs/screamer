# Stream Shaping in the DAG — Phase B (`resample`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A causal windowed `resample` (downsampling) stream op — eager (+ `_iter`) and as a C++ DAG push-node — with by-key-interval and by-count modes and reducers `first/last/min/max/sum/count/mean/ohlc`, plus the engine end-of-input flush so the trailing partial bucket emits and batch == streaming stays byte-identical.

**Architecture:** New stateful `dag::ResampleNode` (single-pass O(1) accumulator, NaN-ignore); `NodeKind::Resample` + a `ResampleParams` sub-struct; `CompiledGraph::flush()` exposed and called by `Dag.stream`; `CombineLatestNode` ports fixed to forward `flush`. Eager `resample` mirrors the engine exactly (integer key-space) and is the identity oracle.

**Tech Stack:** C++17, pybind11, CMake (auto-globs `include/screamer/dag/*.h`, `bindings/*.cpp`), Python 3.11, pytest, numpy.

## Global Constraints

- **Causality:** no lookahead. Batch and streaming MUST be byte-identical (identity harness). The trailing partial bucket emits only on end-of-input flush; both batch (`run_batch` already flushes) and streaming (`Dag.stream` will call flush) flush a finite feed → identical.
- **No Python in the C++ engine:** reducers are C++-only (an `agg` enum), never a callback.
- **NaN policy "ignore":** NaN values are skipped in the accumulator (not stored, not counted); a bucket with events but no finite values emits NaN (or 0 for sum/count). Use `screamer::isnan2(double)` (from `float_info.h`), not `std::isnan`.
- **Integer key-space:** the engine is `int64`-keyed; `width`/`origin`/`count` are `int64`; bucketing uses true floor-division. Eager `resample` casts keys to `int64` so it matches the engine exactly.
- **Width-1 input:** reducers consume width-1 frames. `agg="ohlc"` emits width-4; all others width-1. A width>1 input frame is an error.
- **Sparse output:** only buckets that received ≥1 event emit; empty interior buckets are skipped.
- **Flush idempotency:** `ResampleNode::flush` emits-then-clears, so a second flush is a no-op (flush can arrive multiple times through combine_latest fan-in).
- **Build:** after any C++ change run `make install-dev` (NOT just `make build`). Then `poetry run pytest`.
- Do NOT edit version files, `screamer/__init__.py`, or run `make patch/minor/major`.

---

### Task 1: Eager `resample` + `resample_iter` (the oracle)

**Files:**
- Modify: `screamer/streams.py` (add `resample`, `resample_iter`, helpers)
- Test: `tests/test_streams_resample.py` (create)

**Interfaces:**
- Produces: `resample(keys, values, *, width=None, count=None, agg="last", origin=0, label="left")` → `(out_keys_int64, out_values)`; `out_values` is 1-D except `agg="ohlc"` → `(M,4)`. `resample_iter(events, *, width=..., count=..., agg=..., origin=..., label=...)` generator yielding `(label_key, value)` where value is a float (or 4-list for ohlc). Exactly one of `width`/`count`. These mirror the C++ engine exactly and are the identity oracle.

- [ ] **Step 1: Write the failing test**

Create `tests/test_streams_resample.py`:

```python
import math
import numpy as np
import pytest

from screamer.streams import resample, resample_iter


def test_resample_by_key_last_left_label():
    # width 10, keys 0..25 -> buckets [0,10),[10,20),[20,30)
    keys = np.array([0, 3, 10, 12, 20], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    k, v = resample(keys, vals, width=10, agg="last")
    np.testing.assert_array_equal(k, [0, 10, 20])
    np.testing.assert_array_equal(v, [2.0, 4.0, 5.0])   # last in each bucket; [20,30) trailing


def test_resample_by_key_right_label():
    keys = np.array([0, 3, 10, 12], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    k, v = resample(keys, vals, width=10, agg="sum", label="right")
    np.testing.assert_array_equal(k, [10, 20])          # right = bucket end
    np.testing.assert_array_equal(v, [3.0, 7.0])


def test_resample_by_key_origin():
    keys = np.array([2, 7, 12], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0])
    # origin=2, width=5 -> buckets [2,7),[7,12),[12,17)
    k, v = resample(keys, vals, width=5, origin=2, agg="first")
    np.testing.assert_array_equal(k, [2, 7, 12])
    np.testing.assert_array_equal(v, [1.0, 2.0, 3.0])


def test_resample_aggregations():
    keys = np.array([0, 1, 2, 3], dtype=np.int64)
    vals = np.array([4.0, 2.0, 8.0, 6.0])
    # single bucket width 10
    assert resample(keys, vals, width=10, agg="min")[1].tolist() == [2.0]
    assert resample(keys, vals, width=10, agg="max")[1].tolist() == [8.0]
    assert resample(keys, vals, width=10, agg="sum")[1].tolist() == [20.0]
    assert resample(keys, vals, width=10, agg="count")[1].tolist() == [4.0]
    assert resample(keys, vals, width=10, agg="mean")[1].tolist() == [5.0]
    assert resample(keys, vals, width=10, agg="first")[1].tolist() == [4.0]


def test_resample_ohlc_width4():
    keys = np.array([0, 1, 2, 3], dtype=np.int64)
    vals = np.array([4.0, 2.0, 8.0, 6.0])
    k, v = resample(keys, vals, width=10, agg="ohlc")
    assert v.shape == (1, 4)
    np.testing.assert_array_equal(v[0], [4.0, 8.0, 2.0, 6.0])   # open,high,low,close


def test_resample_nan_ignore():
    keys = np.array([0, 1, 2], dtype=np.int64)
    vals = np.array([np.nan, 4.0, np.nan])
    # bucket has events but one finite value
    assert resample(keys, vals, width=10, agg="mean")[1].tolist() == [4.0]
    assert resample(keys, vals, width=10, agg="count")[1].tolist() == [1.0]


def test_resample_all_nan_bucket_emits_nan():
    keys = np.array([0, 1], dtype=np.int64)
    vals = np.array([np.nan, np.nan])
    k, v = resample(keys, vals, width=10, agg="mean")
    assert len(k) == 1 and math.isnan(v[0])
    # sum of all-nan bucket is 0.0, count is 0
    assert resample(keys, vals, width=10, agg="sum")[1].tolist() == [0.0]
    assert resample(keys, vals, width=10, agg="count")[1].tolist() == [0.0]


def test_resample_sparse_skips_empty_buckets():
    keys = np.array([0, 100], dtype=np.int64)   # width 10: buckets 0 and 10, nothing between
    vals = np.array([1.0, 2.0])
    k, v = resample(keys, vals, width=10, agg="last")
    np.testing.assert_array_equal(k, [0, 100])   # only the two non-empty buckets


def test_resample_by_count():
    keys = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    k, v = resample(keys, vals, count=2, agg="sum")
    # buckets [1,2],[3,4],[5] (trailing partial); left label = first key of bucket
    np.testing.assert_array_equal(k, [10, 30, 50])
    np.testing.assert_array_equal(v, [3.0, 7.0, 5.0])


def test_resample_by_count_right_label():
    keys = np.array([10, 20, 30, 40], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    k, v = resample(keys, vals, count=2, agg="last", label="right")
    np.testing.assert_array_equal(k, [20, 40])   # right = last key of bucket


def test_resample_iter_matches_batch():
    keys = np.array([0, 3, 10, 12, 20], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bk, bv = resample(keys, vals, width=10, agg="mean")
    events = list(zip(keys.tolist(), vals.tolist()))
    got = list(resample_iter(events, width=10, agg="mean"))
    assert [k for k, _ in got] == bk.tolist()
    assert [v for _, v in got] == bv.tolist()


def test_resample_validation_errors():
    keys = np.array([0, 1], dtype=np.int64); vals = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="exactly one"):
        resample(keys, vals)                      # neither width nor count
    with pytest.raises(ValueError, match="exactly one"):
        resample(keys, vals, width=10, count=2)   # both
    with pytest.raises(ValueError, match="agg"):
        resample(keys, vals, width=10, agg="nope")
    with pytest.raises(ValueError, match="label"):
        resample(keys, vals, width=10, label="middle")
    with pytest.raises(ValueError, match="1-D"):
        resample(keys, np.zeros((2, 2)), width=10)   # wide input
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_streams_resample.py -q`
Expected: FAIL with `ImportError: cannot import name 'resample'`.

- [ ] **Step 3: Implement `resample` + `resample_iter`**

In `screamer/streams.py`, add (after `select_iter`). Note `is_node`/`make_combinator_node` are already imported at the top of the module.

```python
_RESAMPLE_AGGS = ("first", "last", "min", "max", "sum", "count", "mean", "ohlc")


class _ResampleAccum:
    """Single-pass O(1) NaN-ignore accumulator. Mirrors the C++ ResampleAccum."""
    __slots__ = ("count", "s", "mn", "mx", "first", "last", "has")

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.s = 0.0
        self.mn = 0.0
        self.mx = 0.0
        self.first = 0.0
        self.last = 0.0
        self.has = False

    def add(self, v):
        self.has = True
        if math.isnan(v):
            return
        if self.count == 0:
            self.mn = self.mx = self.first = self.last = v
        else:
            if v < self.mn:
                self.mn = v
            if v > self.mx:
                self.mx = v
            self.last = v
        self.s += v
        self.count += 1

    def emit(self, agg):
        nan = float("nan")
        c = self.count
        if agg == "first":
            return self.first if c else nan
        if agg == "last":
            return self.last if c else nan
        if agg == "min":
            return self.mn if c else nan
        if agg == "max":
            return self.mx if c else nan
        if agg == "sum":
            return self.s
        if agg == "count":
            return float(c)
        if agg == "mean":
            return self.s / c if c else nan
        # ohlc
        return [self.first if c else nan, self.mx if c else nan,
                self.mn if c else nan, self.last if c else nan]


def _resample_validate(width, count, agg, label):
    if (width is None) == (count is None):
        raise ValueError("resample: pass exactly one of width= or count=")
    if agg not in _RESAMPLE_AGGS:
        raise ValueError(f"resample: agg must be one of {_RESAMPLE_AGGS}")
    if label not in ("left", "right"):
        raise ValueError('resample: label must be "left" or "right"')


def resample(keys, values=None, *, width=None, count=None, agg="last",
             origin=0, label="left"):
    """Causal windowed downsample of a width-1 (key, value) stream.

    Exactly one of `width` (fixed key-interval; buckets [origin+n*width,
    origin+(n+1)*width)) or `count` (fixed event-count). agg is one of
    first/last/min/max/sum/count/mean/ohlc (ohlc -> width-4). label "left"
    stamps the bucket start (by-key) or first key (by-count); "right" stamps the
    bucket end / last key. NaN values are ignored. Only non-empty buckets emit;
    the trailing partial bucket is emitted at end of input. Integer key-space.

    Graph form: resample(stream, ...) where stream is a Node.
    """
    _resample_validate(width, count, agg, label)
    if is_node(keys):
        return make_combinator_node(resample, (keys,), {
            "width": width, "count": count, "agg": agg,
            "origin": origin, "label": label})
    if values is None:
        raise ValueError("resample: values is required (eager form is "
                         "resample(keys, values, ...))")
    keys = np.asarray(keys)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("resample: expects a 1-D value stream (width-1)")

    out_keys, out_vals = [], []
    acc = _ResampleAccum()

    def flush_bucket(label_key):
        out_keys.append(label_key)
        out_vals.append(acc.emit(agg))

    if width is not None:
        w = int(width)
        o = int(origin)
        started = False
        bucket = 0
        cur_label = 0
        for k, v in zip(keys.tolist(), values.tolist()):
            k = int(k)
            nb = (k - o) // w          # Python // is floor division
            if not started:
                started = True
                bucket = nb
                acc.reset()
                cur_label = (o + nb * w) if label == "left" else (o + (nb + 1) * w)
            elif nb != bucket:
                if acc.has:
                    flush_bucket(cur_label)
                bucket = nb
                acc.reset()
                cur_label = (o + nb * w) if label == "left" else (o + (nb + 1) * w)
            acc.add(v)
        if acc.has:
            flush_bucket(cur_label)
    else:
        n = int(count)
        cib = 0
        first_key = 0
        last_key = 0
        for k, v in zip(keys.tolist(), values.tolist()):
            k = int(k)
            if cib == 0:
                first_key = k
            last_key = k
            acc.add(v)
            cib += 1
            if cib == n:
                flush_bucket(first_key if label == "left" else last_key)
                acc.reset()
                cib = 0
        if cib > 0:
            flush_bucket(first_key if label == "left" else last_key)

    ok = np.array(out_keys, dtype=np.int64)
    if agg == "ohlc":
        ov = (np.array(out_vals, dtype=np.float64).reshape(-1, 4)
              if out_vals else np.empty((0, 4), dtype=np.float64))
    else:
        ov = np.array(out_vals, dtype=np.float64)
    return ok, ov


def resample_iter(events, *, width=None, count=None, agg="last",
                  origin=0, label="left"):
    """Streaming resample over (key, value) tuples. Yields (label_key, value)."""
    _resample_validate(width, count, agg, label)
    acc = _ResampleAccum()
    if width is not None:
        w = int(width)
        o = int(origin)
        started = False
        bucket = 0
        cur_label = 0
        for k, v in events:
            k = int(k)
            nb = (k - o) // w
            if not started:
                started = True
                bucket = nb
                acc.reset()
                cur_label = (o + nb * w) if label == "left" else (o + (nb + 1) * w)
            elif nb != bucket:
                if acc.has:
                    yield cur_label, acc.emit(agg)
                bucket = nb
                acc.reset()
                cur_label = (o + nb * w) if label == "left" else (o + (nb + 1) * w)
            acc.add(float(v))
        if acc.has:
            yield cur_label, acc.emit(agg)
    else:
        n = int(count)
        cib = 0
        first_key = 0
        last_key = 0
        for k, v in events:
            k = int(k)
            if cib == 0:
                first_key = k
            last_key = k
            acc.add(float(v))
            cib += 1
            if cib == n:
                yield (first_key if label == "left" else last_key), acc.emit(agg)
                acc.reset()
                cib = 0
        if cib > 0:
            yield (first_key if label == "left" else last_key), acc.emit(agg)
```

Ensure `import math` is present at the top of `screamer/streams.py` (add it if missing).

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_streams_resample.py -q`
Expected: PASS (12 passed).

- [ ] **Step 5: Commit**

```bash
git add screamer/streams.py tests/test_streams_resample.py
git commit -m "feat(streams): add eager resample + resample_iter (causal downsampling)"
```

---

### Task 2: Engine flush + `ResampleNode` (by-key mode)

**Files:**
- Create: `include/screamer/dag/resample_params.h`
- Create: `include/screamer/dag/resample_node.h`
- Modify: `include/screamer/dag/combine_latest_node.h` (Port forwards `flush`)
- Modify: `include/screamer/dag/graph.h` (`NodeKind::Resample`, `NodeSpec.resample`, `add_resample`)
- Modify: `include/screamer/dag/compiled_graph.h` (include, `flush()`, node-width, wiring + reset list)
- Modify: `bindings/bindings_dag.cpp` (`add_resample`, `_CompiledGraph.flush`)
- Modify: `screamer/dag.py` (dispatch `"resample"`, `stream()` calls flush)
- Modify: `screamer/streams.py` (already dispatches via Task 1's Node branch — verify)
- Test: `tests/test_dag_resample.py` (create)

**Interfaces:**
- Consumes: `dag::Frame`/`Sink`, `screamer::isnan2`, the node-plumbing pattern from Phase A.
- Produces:
  - `dag::ResampleParams{mode, agg, label, width, origin, count}` + enums `ResampleMode{ByKey,ByCount}`, `ResampleAgg{First,Last,Min,Max,Sum,Count,Mean,Ohlc}`, `ResampleLabel{Left,Right}`; `resample_width(agg)`.
  - `dag::ResampleNode<Key>(ResampleParams, Sink<Key>&)` — stateful, `push`/`flush`/`reset`.
  - `GraphBuilder::add_resample(inputs, ResampleParams)`.
  - `CompiledGraph::flush()`; `_CompiledGraph.flush()`; `_GraphBuilder.add_resample(inputs, mode, agg, label, width, origin, count)`.
  - `Dag.stream` calls `self._cg.flush()` before `drain()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dag_resample.py`:

```python
import numpy as np
import pytest

from screamer import Input, Dag
from screamer.streams import resample, combine_latest


AGGS = ["first", "last", "min", "max", "sum", "count", "mean"]


def _stream_1d(dag, *feeds):
    bk, bv = dag(*feeds)
    sk, sv = dag.stream(*feeds)
    return (bk, bv), (sk, sv)


@pytest.mark.parametrize("agg", AGGS)
def test_resample_by_key_batch_stream_oracle(agg):
    keys = np.array([0, 3, 10, 12, 20, 25], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[resample(x, width=10, agg=agg)])
    (bk, bv), (sk, sv) = _stream_1d(dag, (keys, vals))
    ek, ev = resample(keys, vals, width=10, agg=agg)   # eager oracle
    np.testing.assert_array_equal(bk, ek)
    np.testing.assert_array_equal(bv.reshape(-1), ev)
    np.testing.assert_array_equal(sk, ek)
    np.testing.assert_array_equal(sv.reshape(-1), ev)


def test_resample_by_key_right_label_and_origin():
    keys = np.array([2, 7, 12, 19], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[resample(x, width=5, origin=2, agg="sum", label="right")])
    (bk, bv), (sk, sv) = _stream_1d(dag, (keys, vals))
    ek, ev = resample(keys, vals, width=5, origin=2, agg="sum", label="right")
    np.testing.assert_array_equal(bk, ek); np.testing.assert_array_equal(bv.reshape(-1), ev)
    np.testing.assert_array_equal(sk, ek); np.testing.assert_array_equal(sv.reshape(-1), ev)


def test_resample_ohlc_width4_in_graph():
    keys = np.array([0, 1, 2, 10, 11], dtype=np.int64)
    vals = np.array([4.0, 2.0, 8.0, 6.0, 7.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[resample(x, width=10, agg="ohlc")])
    bk, bv = dag((keys, vals))
    sk, sv = dag.stream((keys, vals))
    ek, ev = resample(keys, vals, width=10, agg="ohlc")
    assert bv.shape[1] == 4
    np.testing.assert_array_equal(bv, ev)
    np.testing.assert_array_equal(sv, ev)
    np.testing.assert_array_equal(bk, ek)


def test_resample_trailing_bucket_flush_batch_equals_stream():
    # the last bucket [20,30) is partial (only key 25); it must emit in BOTH modes
    keys = np.array([0, 10, 25], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[resample(x, width=10, agg="last")])
    bk, bv = dag((keys, vals))
    sk, sv = dag.stream((keys, vals))
    np.testing.assert_array_equal(bk, [0, 10, 20])
    np.testing.assert_array_equal(bv.reshape(-1), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(sk, bk)
    np.testing.assert_array_equal(sv, bv)


def test_resample_flush_through_combine_latest():
    # resample downstream of a functor fed by combine_latest: flush must reach
    # the resample THROUGH the combine_latest port (the Port::flush fix).
    from screamer import Sub
    ak = np.array([0, 10, 25], dtype=np.int64); av = np.array([5.0, 6.0, 7.0])
    bk = np.array([0, 10, 25], dtype=np.int64); bv = np.array([1.0, 1.0, 1.0])
    a, b = Input("a"), Input("b")
    dag = Dag(inputs=[a, b], outputs=[resample(Sub()(combine_latest(a, b)), width=10, agg="last")])
    rbk, rbv = dag((ak, av), (bk, bv))
    rsk, rsv = dag.stream((ak, av), (bk, bv))
    # batch == stream is the key guarantee (trailing bucket present in both)
    np.testing.assert_array_equal(rbk, rsk)
    np.testing.assert_array_equal(rbv, rsv)
    # the [20,30) bucket (key 25) must be present -> 3 buckets
    assert len(rbk) == 3


def test_resample_feeds_functor():
    from screamer import RollingMean
    keys = np.array([0, 1, 10, 11, 20], dtype=np.int64)
    vals = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[RollingMean(2)(resample(x, width=10, agg="mean"))])
    bk, bv = dag((keys, vals))
    sk, sv = dag.stream((keys, vals))
    np.testing.assert_array_equal(bk, sk)
    np.testing.assert_array_equal(bv, sv)


def test_existing_stream_dag_unaffected_by_flush():
    # a non-resample streaming dag must be byte-identical to before (flush no-op)
    from screamer import RollingMean
    keys = np.array([1, 2, 3, 4], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[RollingMean(2)(x)])
    bk, bv = dag((keys, vals))
    sk, sv = dag.stream((keys, vals))
    np.testing.assert_array_equal(bk, sk)
    np.testing.assert_array_equal(bv, sv)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_dag_resample.py -q`
Expected: FAIL — `resample(x, ...)` on a Node builds a combinator node the `Dag` does not yet accept.

- [ ] **Step 3a: Add `resample_params.h`**

Create `include/screamer/dag/resample_params.h`:

```cpp
#ifndef SCREAMER_DAG_RESAMPLE_PARAMS_H
#define SCREAMER_DAG_RESAMPLE_PARAMS_H

#include <cstddef>
#include <cstdint>

namespace screamer { namespace dag {

enum class ResampleMode  { ByKey, ByCount };
enum class ResampleAgg   { First, Last, Min, Max, Sum, Count, Mean, Ohlc };
enum class ResampleLabel { Left, Right };

struct ResampleParams {
    ResampleMode  mode  = ResampleMode::ByKey;
    ResampleAgg   agg   = ResampleAgg::Last;
    ResampleLabel label = ResampleLabel::Left;
    std::int64_t  width  = 1;   // ByKey
    std::int64_t  origin = 0;   // ByKey
    std::int64_t  count  = 1;   // ByCount
};

inline std::size_t resample_width(ResampleAgg a) {
    return a == ResampleAgg::Ohlc ? 4u : 1u;
}

}} // namespace screamer::dag
#endif
```

- [ ] **Step 3b: Add `resample_node.h`**

Create `include/screamer/dag/resample_node.h`:

```cpp
#ifndef SCREAMER_DAG_RESAMPLE_NODE_H
#define SCREAMER_DAG_RESAMPLE_NODE_H

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include "screamer/common/float_info.h"
#include "screamer/dag/frame.h"
#include "screamer/dag/resample_params.h"

namespace screamer { namespace dag {

// Single-pass O(1) NaN-ignore accumulator. add() folds one value; emit() writes
// the reducer result. `has` marks that any event (NaN or not) fell in the bucket.
struct ResampleAccum {
    std::int64_t count = 0;
    double sum = 0.0, mn = 0.0, mx = 0.0, first = 0.0, last = 0.0;
    bool has = false;

    void reset() { count = 0; sum = 0.0; mn = mx = first = last = 0.0; has = false; }

    void add(double v) {
        has = true;
        if (screamer::isnan2(v)) return;      // ignore policy
        if (count == 0) { mn = mx = first = last = v; }
        else {
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            last = v;
        }
        sum += v;
        ++count;
    }

    void emit(ResampleAgg agg, double* out) const {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        switch (agg) {
        case ResampleAgg::First: out[0] = count ? first : nan; break;
        case ResampleAgg::Last:  out[0] = count ? last  : nan; break;
        case ResampleAgg::Min:   out[0] = count ? mn    : nan; break;
        case ResampleAgg::Max:   out[0] = count ? mx    : nan; break;
        case ResampleAgg::Sum:   out[0] = sum; break;
        case ResampleAgg::Count: out[0] = static_cast<double>(count); break;
        case ResampleAgg::Mean:  out[0] = count ? sum / static_cast<double>(count) : nan; break;
        case ResampleAgg::Ohlc:
            out[0] = count ? first : nan;
            out[1] = count ? mx    : nan;
            out[2] = count ? mn    : nan;
            out[3] = count ? last  : nan;
            break;
        }
    }
};

// Stateful windowing push-node. Buckets a width-1 stream by key-interval or event
// count, reduces each bucket with a fixed C++ reducer, and emits the bucket on the
// causal boundary (a key crossing the bucket end / the Nth event). flush() emits
// the trailing partial bucket and is idempotent (emit-then-clear).
template <class Key>
class ResampleNode : public Sink<Key> {
public:
    ResampleNode(ResampleParams p, Sink<Key>& downstream)
        : p_(p), downstream_(downstream), out_(resample_width(p.agg)) { reset(); }

    void push(const Frame<Key>& f) override {
        if (f.width != 1)
            throw std::runtime_error("dag::ResampleNode: expects a width-1 input stream");
        if (p_.mode == ResampleMode::ByKey) push_by_key(f.key, f.values[0]);
        else                                push_by_count(f.key, f.values[0]);
    }

    void flush() override {
        if (p_.mode == ResampleMode::ByKey) {
            if (acc_.has) emit(cur_label_);
        } else {
            if (count_in_bucket_ > 0) emit(p_.label == ResampleLabel::Left ? first_key_ : last_key_);
        }
        // idempotent: clear so a repeat flush emits nothing
        started_ = false;
        count_in_bucket_ = 0;
        acc_.reset();
        downstream_.flush();
    }

    void reset() {
        acc_.reset();
        started_ = false;
        bucket_ = 0;
        cur_label_ = Key{};
        count_in_bucket_ = 0;
        first_key_ = last_key_ = Key{};
    }

private:
    void push_by_key(Key k, double v) {
        std::int64_t nb = floordiv(static_cast<std::int64_t>(k) - p_.origin, p_.width);
        if (!started_) {
            started_ = true; bucket_ = nb; acc_.reset(); set_key_label(nb);
        } else if (nb != bucket_) {
            if (acc_.has) emit(cur_label_);
            bucket_ = nb; acc_.reset(); set_key_label(nb);
        }
        acc_.add(v);
    }

    void set_key_label(std::int64_t nb) {
        std::int64_t start = p_.origin + nb * p_.width;
        cur_label_ = static_cast<Key>(p_.label == ResampleLabel::Left ? start : start + p_.width);
    }

    void push_by_count(Key k, double v) {
        if (count_in_bucket_ == 0) first_key_ = k;
        last_key_ = k;
        acc_.add(v);
        ++count_in_bucket_;
        if (count_in_bucket_ == p_.count) {
            emit(p_.label == ResampleLabel::Left ? first_key_ : last_key_);
            acc_.reset();
            count_in_bucket_ = 0;
        }
    }

    void emit(Key label) {
        acc_.emit(p_.agg, out_.data());
        downstream_.push(Frame<Key>{label, out_.data(), out_.size()});
    }

    static std::int64_t floordiv(std::int64_t a, std::int64_t b) {
        std::int64_t q = a / b, r = a % b;
        if (r != 0 && ((r < 0) != (b < 0))) --q;
        return q;
    }

    ResampleParams p_;
    Sink<Key>& downstream_;
    std::vector<double> out_;
    ResampleAccum acc_;
    bool started_ = false;
    std::int64_t bucket_ = 0;
    Key cur_label_{};
    std::int64_t count_in_bucket_ = 0;
    Key first_key_{}, last_key_{};
};

}} // namespace screamer::dag
#endif
```

- [ ] **Step 3c: Make `CombineLatestNode` ports forward `flush`**

In `include/screamer/dag/combine_latest_node.h`, add a private forwarder and override `Port::flush`. Add after `on_port`:
```cpp
    void flush_downstream() { downstream_.flush(); }
```
And in `struct Port` (after its `push`):
```cpp
        void flush() override { node.flush_downstream(); }
```
(With N ports each wired to a different input, `downstream_.flush()` may be called up to N times; downstream stateful nodes flush idempotently, so this is safe.)

- [ ] **Step 3d: Extend the graph spec + builder**

In `include/screamer/dag/graph.h`:

Add the include at the top:
```cpp
#include "screamer/dag/resample_params.h"
```

Add `Resample` to the enum:
```cpp
enum class NodeKind { Input, Functor, CombineLatest, DropNa, Select, Resample };
```

Add a `resample` field to `NodeSpec` (after `columns`):
```cpp
    std::vector<std::size_t> columns;     // Select only
    ResampleParams resample;              // Resample only
    std::vector<std::size_t> inputs;      // producer node ids
```

Add the builder method (after `add_select`):
```cpp
std::size_t add_resample(std::vector<std::size_t> inputs, ResampleParams rp) {
    NodeSpec ns{NodeKind::Resample, nullptr, true, false, {}, rp, std::move(inputs)};
    spec_.nodes.push_back(std::move(ns));
    return spec_.nodes.size() - 1;
}
```
NOTE: inserting the `resample` field shifts aggregate-initializer positions again. Update EVERY existing `add_*` `NodeSpec{...}` initializer to include a default `ResampleParams{}` in the `resample` position (between `columns` and `inputs`). The current 6-field form `{kind, op, when_all, how_all, columns, inputs}` becomes 7-field `{kind, op, when_all, how_all, columns, resample, inputs}`:
```cpp
spec_.nodes.push_back(NodeSpec{NodeKind::Input, nullptr, true, false, {}, {}, {}});
spec_.nodes.push_back(NodeSpec{NodeKind::Functor, op, true, false, {}, {}, std::move(inputs)});
spec_.nodes.push_back(NodeSpec{NodeKind::CombineLatest, nullptr, when_all, false, {}, {}, std::move(inputs)});
spec_.nodes.push_back(NodeSpec{NodeKind::DropNa, nullptr, true, how_all, {}, {}, std::move(inputs)});
// add_select builds its NodeSpec via a named local; give it the resample field too:
NodeSpec ns{NodeKind::Select, nullptr, true, false, std::move(columns), {}, std::move(inputs)};
```
Read the CURRENT `graph.h` before editing — match the exact field order `{kind, op, when_all, how_all, columns, resample, inputs}` in every initializer.

- [ ] **Step 3e: Wire `Resample` + `flush()` in the compiler**

In `include/screamer/dag/compiled_graph.h`:

Add the include:
```cpp
#include "screamer/dag/resample_node.h"
```

Add the `Resample` case to the `node_width` pass (width depends on agg):
```cpp
    case NodeKind::Resample:      node_width[id] = resample_width(nd.resample.agg); break;
```

Add a reset list member (near `reset_combines_`):
```cpp
    std::vector<ResampleNode<std::int64_t>*> reset_resamples_;
```

Add the `Resample` wiring case (single-input, STATEFUL → registered for reset):
```cpp
case NodeKind::Resample: {
    auto rn = std::make_shared<ResampleNode<std::int64_t>>(ns.resample, *downstream);
    reset_resamples_.push_back(rn.get());
    node_input_sink[id] = [ptr = rn.get()](std::size_t) -> Sink<std::int64_t>* {
        return ptr;
    };
    owned_.push_back(rn);
    break;
}
```

Reset resample nodes in `reset()` (alongside `reset_combines_`):
```cpp
    for (auto* r : reset_resamples_) r->reset();
```

Add a public `flush()` method (after `push_event`), mirroring the end-of-batch flush that `run_batch` already performs:
```cpp
// Emits every open trailing bucket (end-of-input signal). Existing nodes emit
// nothing on flush; resample nodes emit their partial bucket. Idempotent.
void flush() {
    for (auto* s : input_sinks_) if (s) s->flush();
}
```

- [ ] **Step 3f: Bind `add_resample` + `_CompiledGraph.flush`**

In `bindings/bindings_dag.cpp`:

In `struct PyCompiledGraph`, add:
```cpp
void flush() { cg->flush(); }
```
And in its `py::class_` (after `push_event`):
```cpp
        .def("flush", &PyCompiledGraph::flush)
```

In `struct PyGraphBuilder`, add (after `add_select`) — take plain ints across the boundary and assemble `ResampleParams`:
```cpp
std::size_t add_resample(std::vector<std::size_t> inputs, int mode, int agg,
                         int label, std::int64_t width, std::int64_t origin,
                         std::int64_t count) {
    dag::ResampleParams rp;
    rp.mode   = static_cast<dag::ResampleMode>(mode);    // 0=ByKey, 1=ByCount
    rp.agg    = static_cast<dag::ResampleAgg>(agg);      // 0..7 First..Ohlc
    rp.label  = static_cast<dag::ResampleLabel>(label);  // 0=Left, 1=Right
    rp.width  = width;
    rp.origin = origin;
    rp.count  = count;
    return builder.add_resample(std::move(inputs), rp);
}
```
And in the `_GraphBuilder` `py::class_` (after `add_select`):
```cpp
.def("add_resample", [](PyGraphBuilder& b, std::vector<std::size_t> inputs,
                        int mode, int agg, int label,
                        std::int64_t width, std::int64_t origin, std::int64_t count) {
    return b.add_resample(std::move(inputs), mode, agg, label, width, origin, count);
}, py::arg("inputs"), py::arg("mode"), py::arg("agg"), py::arg("label"),
   py::arg("width"), py::arg("origin"), py::arg("count"))
```
Ensure `#include "screamer/dag/resample_params.h"` is available (compiled_graph.h/graph.h already pull it transitively, but add it explicitly to `bindings_dag.cpp` if `dag::ResampleParams` is unresolved).

- [ ] **Step 3g: Dispatch `resample` in Python + flush in `stream()`**

In `screamer/dag.py`, add the enum maps near the top of the module (module scope):
```python
_RESAMPLE_AGG_CODE = {"first": 0, "last": 1, "min": 2, "max": 3,
                      "sum": 4, "count": 5, "mean": 6, "ohlc": 7}
```
Add the `resample` branch to `build()`'s combinator dispatch (after `select`):
```python
                elif name == "resample":
                    mode = 1 if kwargs.get("count") is not None else 0   # 0=ByKey,1=ByCount
                    agg = _RESAMPLE_AGG_CODE[kwargs.get("agg", "last")]
                    label = 1 if kwargs.get("label", "left") == "right" else 0
                    width = int(kwargs["width"]) if kwargs.get("width") is not None else 1
                    origin = int(kwargs.get("origin", 0))
                    count = int(kwargs["count"]) if kwargs.get("count") is not None else 1
                    nid = gb.add_resample(inp, mode, agg, label, width, origin, count)
```
In `Dag.stream`, add the flush call after the push loop, before `drain()`:
```python
        for k, v, s in zip(mk, mv, ms):
            self._cg.push_event(int(s), int(k), float(v))
        self._cg.flush()          # end-of-input: emit trailing resample buckets
        results = self._cg.drain()
```

- [ ] **Step 4: Build and run the tests**

```bash
make install-dev
poetry run pytest tests/test_dag_resample.py tests/test_streams_resample.py -q
```
Expected: PASS (by-count-only tests are added in Task 3; all by-key + flush tests here pass).
If a build error mentions `NodeSpec` aggregate-initializer arity, an `add_*` initializer is missing the `resample` field — fix per Step 3d.

- [ ] **Step 5: Regression + commit**

```bash
poetry run pytest tests/test_dag_identity.py tests/test_dag_dropna.py tests/test_dag_select.py -q
git add include/screamer/dag/resample_params.h include/screamer/dag/resample_node.h include/screamer/dag/combine_latest_node.h include/screamer/dag/graph.h include/screamer/dag/compiled_graph.h bindings/bindings_dag.cpp screamer/dag.py tests/test_dag_resample.py
git commit -m "feat(dag): resample by-key push-node + engine end-of-input flush"
```
Expected: identity + Phase A regressions green (flush is a no-op for them).

---

### Task 3: `resample` by-count mode in the graph

**Files:**
- Modify: `tests/test_dag_resample.py` (add by-count graph tests)

**Interfaces:**
- Consumes: everything from Task 2 (`ResampleNode` already implements by-count; the dispatch already maps `count=`). Task 3 only proves the by-count path in the graph and pins its identity, since Task 2's node and dispatch already handle both modes.

- [ ] **Step 1: Write the failing/again-passing tests**

Append to `tests/test_dag_resample.py`:

```python
@pytest.mark.parametrize("agg", AGGS)
def test_resample_by_count_batch_stream_oracle(agg):
    keys = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[resample(x, count=2, agg=agg)])
    bk, bv = dag((keys, vals))
    sk, sv = dag.stream((keys, vals))
    ek, ev = resample(keys, vals, count=2, agg=agg)   # eager oracle (incl. trailing partial)
    np.testing.assert_array_equal(bk, ek)
    np.testing.assert_array_equal(bv.reshape(-1), ev)
    np.testing.assert_array_equal(sk, ek)
    np.testing.assert_array_equal(sv.reshape(-1), ev)


def test_resample_by_count_right_label_in_graph():
    keys = np.array([10, 20, 30, 40], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[resample(x, count=2, agg="last", label="right")])
    bk, bv = dag((keys, vals))
    sk, sv = dag.stream((keys, vals))
    ek, ev = resample(keys, vals, count=2, agg="last", label="right")
    np.testing.assert_array_equal(bk, ek); np.testing.assert_array_equal(bv.reshape(-1), ev)
    np.testing.assert_array_equal(sk, ek); np.testing.assert_array_equal(sv.reshape(-1), ev)


def test_resample_by_count_ohlc_in_graph():
    keys = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    vals = np.array([4.0, 2.0, 8.0, 6.0, 7.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[resample(x, count=2, agg="ohlc")])
    bk, bv = dag((keys, vals))
    sk, sv = dag.stream((keys, vals))
    ek, ev = resample(keys, vals, count=2, agg="ohlc")
    assert bv.shape[1] == 4
    np.testing.assert_array_equal(bv, ev)
    np.testing.assert_array_equal(sv, ev)
    np.testing.assert_array_equal(bk, ek)
```

- [ ] **Step 2: Run the tests**

```bash
poetry run pytest tests/test_dag_resample.py -q
```
Expected: PASS. (No source change should be needed — Task 2's `ResampleNode` and dispatch already implement by-count. If any by-count test fails, fix the `ResampleNode::push_by_count` / dispatch in the Task 2 files, then re-run.)

- [ ] **Step 3: Full regression + commit**

```bash
poetry run pytest -q
git add tests/test_dag_resample.py
git commit -m "test(dag): cover resample by-count mode in the graph (batch==stream==oracle)"
```
Expected: full suite green.

---

## Self-Review

**Spec coverage:** eager resample + `_iter` both modes/all aggs (Task 1); by-key graph node + engine flush + combine_latest flush fix (Task 2); by-count graph mode (Task 3). Labels, origin, ohlc width-4, NaN-ignore, sparse, trailing flush, flush-through-combine, flush no-op for existing graphs — all covered by tests.

**Type consistency:** `ResampleParams{mode,agg,label,width,origin,count}`, `ResampleNode<Key>(ResampleParams, Sink&)`, `add_resample(inputs, ResampleParams)` / binding `add_resample(inputs, mode, agg, label, width, origin, count)` with int enum codes (mode 0/1, agg 0..7 First..Ohlc, label 0/1) matching `_RESAMPLE_AGG_CODE` in `dag.py`. `NodeSpec` field order `{kind, op, when_all, how_all, columns, resample, inputs}` (7 fields) is called out with every `add_*` initializer updated in lockstep. `CompiledGraph::flush()` ↔ `_CompiledGraph.flush()` ↔ `Dag.stream`.

**Placeholder scan:** none — every step carries full code.

**Highest-risk items flagged:** (1) `NodeSpec` 7-field initializer arity across all builders; (2) `CombineLatestNode` Port must forward flush (else resample-through-combine loses its trailing bucket) + flush idempotency; (3) `Dag.stream` must call `flush()` before `drain()` so streaming matches batch. All three are explicit steps with code.
