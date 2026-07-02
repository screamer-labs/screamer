# Multi-stream Plan 2 — merge + pace (source/driver layer) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the k-way `MergeSource` (interleave N key-sorted sources into one globally-sorted, source-tagged stream) and the `pace` async driver (wall-clock replay), so stored series can be replayed as a backtest (max speed) or against a real clock — with the batch and streaming event sequences provably identical.

**Architecture:** Builds directly on Plan 1's `Source<Key>`/`Event<Key>`. `MergeSource<Key>` is a `Source` that holds N child sources and a min-heap of their lookahead events, popping in global key order with deterministic tie-break by child index (stored in `Event::source`). The batch driver drains it into a collector; the streaming path exposes a Python-pullable iterator, and `pace` is a thin Python async generator that pulls that iterator and sleeps proportional to key-deltas (injectable clock, so pacing never touches values and is deterministically testable).

**Tech Stack:** C++17, pybind11, numpy, asyncio, pytest.

## Global Constraints

- **Causality**: merge emits an event only after it has been pulled from its child; no lookahead beyond the single next event each child already produced.
- **Cross-mode identity**: the batch merge output and the pull-iterator (streaming) output must be **byte-identical** event sequences — same keys, values, and source tags, in the same order. Guard with a test.
- **Deterministic tie-break**: when multiple children have the same key, emit in ascending child index order. Stable and reproducible.
- **Pacing never changes values or order** — `pace` only controls *when* events are yielded. `speed=inf` (or a no-op clock) must yield the exact backtest sequence.
- **Numeric keys only** (`int64_t`/`double`), chosen at the Python boundary; datetime64→int64 view; floating→double.
- **Value is always `double`**; compute functors never modified; node/source classes stay internal (underscore-prefixed bindings, not exported into `screamer/__init__.py`).
- **Efficiency (product identity):** preallocate merge output to the exact total length (sum of child lengths — known up front); no per-event reallocation; the per-event path stays in C++. Do not allocate buffers that are discarded.
- Build: `make build` **then `make install-dev`** (editable install resolves the binding from `.venv` site-packages; `make build` alone can leave it stale). Test: `poetry run pytest tests/test_streams_merge.py -v`.
- Never hand-edit `screamer/__init__.py` or version files.

---

## File Structure

- `include/screamer/streams/merge_source.h` — `MergeSource<Key>`: k-way heap merge over child `Source<Key>`s.
- `bindings/bindings_streams.cpp` (modify) — `_merge_i64`/`_merge_f64` (batch → arrays); `_MergePuller_i64`/`_f64` (pull iterator); a `return_keys` fast-path fix for `run_chain`.
- `screamer/streams.py` (modify) — `merge(*series)` batch wrapper; `replay(*series)` / `pace(...)` async generator with injectable clock.
- `tests/test_streams_merge.py` — merge ordering/tie-break/tags, cross-mode identity, pace timing.

---

### Task 1: `MergeSource` + batch merge binding

**Files:**
- Create: `include/screamer/streams/merge_source.h`
- Modify: `bindings/bindings_streams.cpp`
- Modify: `screamer/streams.py`
- Test: `tests/test_streams_merge.py`

**Interfaces:**
- Consumes: `Event<Key>`, `Source<Key>` (Plan 1, `event.h`); `VectorSource<Key>` (`vector_source.h`).
- Produces:
  - C++: `MergeSource<Key>(std::vector<Source<Key>*> children)`; `next()` returns events in global key order, `Event::source` = child index, ties by ascending child index.
  - Binding: `_merge_i64(list[np.ndarray[int64]] keys, list[np.ndarray[float64]] values) -> (np.ndarray[int64], np.ndarray[float64], np.ndarray[uint32])` (keys, values, sources). `_merge_f64` identical with float64 keys.
  - Python: `screamer.streams.merge(*series)` where each series is `(keys, values)`; returns `(keys, values, sources)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_streams_merge.py`:

```python
import numpy as np
from screamer import streams


def _reference_merge(series):
    # Stable global sort by key, ties broken by source index (the series order).
    keys = np.concatenate([np.asarray(k) for k, _ in series])
    vals = np.concatenate([np.asarray(v) for _, v in series])
    src = np.concatenate([np.full(len(k), i, dtype=np.uint32)
                          for i, (k, _) in enumerate(series)])
    order = np.argsort(keys, kind="stable")   # stable => source-order tie-break
    return keys[order], vals[order], src[order]


def test_merge_two_int_sorted_series():
    a_k = np.array([1, 3, 5, 7], dtype=np.int64)
    a_v = np.array([10.0, 30.0, 50.0, 70.0])
    b_k = np.array([2, 4, 6], dtype=np.int64)
    b_v = np.array([20.0, 40.0, 60.0])
    got_k, got_v, got_s = streams.merge((a_k, a_v), (b_k, b_v))
    exp_k, exp_v, exp_s = _reference_merge([(a_k, a_v), (b_k, b_v)])
    np.testing.assert_array_equal(got_k, exp_k)
    np.testing.assert_array_equal(got_v, exp_v)
    np.testing.assert_array_equal(got_s, exp_s)


def test_merge_ties_break_by_source_order():
    a_k = np.array([1, 2, 2], dtype=np.int64)
    a_v = np.array([1.0, 2.0, 2.5])
    b_k = np.array([2, 3], dtype=np.int64)
    b_v = np.array([20.0, 30.0])
    got_k, got_v, got_s = streams.merge((a_k, a_v), (b_k, b_v))
    # at key==2: source 0's events (2.0, 2.5) come before source 1's (20.0)
    np.testing.assert_array_equal(got_k, np.array([1, 2, 2, 2, 3], dtype=np.int64))
    np.testing.assert_array_equal(got_v, np.array([1.0, 2.0, 2.5, 20.0, 30.0]))
    np.testing.assert_array_equal(got_s, np.array([0, 0, 0, 1, 1], dtype=np.uint32))


def test_merge_float_keys():
    a_k = np.array([0.5, 2.5], dtype=np.float64)
    a_v = np.array([5.0, 25.0])
    b_k = np.array([1.5], dtype=np.float64)
    b_v = np.array([15.0])
    got_k, got_v, got_s = streams.merge((a_k, a_v), (b_k, b_v))
    exp_k, exp_v, exp_s = _reference_merge([(a_k, a_v), (b_k, b_v)])
    np.testing.assert_array_equal(got_k, exp_k)
    np.testing.assert_array_equal(got_v, exp_v)
    np.testing.assert_array_equal(got_s, exp_s)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_streams_merge.py -v`
Expected: FAIL — `AttributeError: module 'screamer.streams' has no attribute 'merge'`.

- [ ] **Step 3: Create `MergeSource`**

`include/screamer/streams/merge_source.h`:

```cpp
#ifndef SCREAMER_STREAMS_MERGE_SOURCE_H
#define SCREAMER_STREAMS_MERGE_SOURCE_H

#include <cstdint>
#include <queue>
#include <vector>
#include "screamer/streams/event.h"

namespace screamer { namespace streams {

// K-way merge of N individually key-sorted child sources into one globally
// key-sorted stream. Event::source is set to the child index. Ties (equal
// keys across children) break by ascending child index, deterministically.
template <class Key>
class MergeSource : public Source<Key> {
public:
    explicit MergeSource(std::vector<Source<Key>*> children)
        : children_(std::move(children)) {
        // Prime the heap with the first event from each child.
        for (std::uint32_t i = 0; i < children_.size(); ++i) {
            if (auto e = children_[i]->next()) {
                heap_.push(Node{e->key, i, e->value});
            }
        }
    }

    std::optional<Event<Key>> next() override {
        if (heap_.empty()) return std::nullopt;
        Node top = heap_.top();
        heap_.pop();
        // Pull the child's next event to keep the heap primed.
        if (auto e = children_[top.source]->next()) {
            heap_.push(Node{e->key, top.source, e->value});
        }
        return Event<Key>{top.key, top.value, top.source};
    }

private:
    struct Node {
        Key key;
        std::uint32_t source;
        double value;
    };
    // Min-heap on (key, source): smaller key first; equal keys -> smaller
    // source index first (deterministic tie-break by child order).
    struct Greater {
        bool operator()(const Node& a, const Node& b) const {
            if (a.key != b.key) return a.key > b.key;
            return a.source > b.source;
        }
    };

    std::vector<Source<Key>*> children_;
    std::priority_queue<Node, std::vector<Node>, Greater> heap_;
};

}} // namespace screamer::streams
#endif
```

- [ ] **Step 4: Add the batch merge binding**

In `bindings/bindings_streams.cpp`, add the include and a `merge_batch<Key>` helper, then register both instantiations. Add near the other includes:

```cpp
#include "screamer/streams/merge_source.h"
```

Add the helper (uses exact total length — sum of child sizes — so the output is allocated once, no reallocation):

```cpp
template <class Key>
static py::tuple merge_batch(py::list key_arrays, py::list value_arrays) {
    std::size_t n_children = key_arrays.size();
    if (value_arrays.size() != n_children) {
        throw std::runtime_error("merge: keys/values list length mismatch");
    }

    // Materialize child VectorSources and total length.
    std::vector<py::array_t<Key>> keys;
    std::vector<py::array_t<double>> vals;
    keys.reserve(n_children);
    vals.reserve(n_children);
    std::vector<std::unique_ptr<VectorSource<Key>>> sources;
    std::vector<Source<Key>*> child_ptrs;
    std::size_t total = 0;

    for (std::size_t i = 0; i < n_children; ++i) {
        keys.push_back(py::cast<py::array_t<Key>>(key_arrays[i]));
        vals.push_back(py::cast<py::array_t<double>>(value_arrays[i]));
        auto kinfo = keys[i].request();
        auto vinfo = vals[i].request();
        if (kinfo.shape[0] != vinfo.shape[0]) {
            throw std::runtime_error("merge: a child's keys/values length differ");
        }
        std::size_t n = static_cast<std::size_t>(vinfo.shape[0]);
        total += n;
        sources.push_back(std::make_unique<VectorSource<Key>>(
            static_cast<const Key*>(kinfo.ptr),
            static_cast<const double*>(vinfo.ptr), n));
        child_ptrs.push_back(sources.back().get());
    }

    py::array_t<Key> out_k(total);
    py::array_t<double> out_v(total);
    py::array_t<std::uint32_t> out_s(total);
    Key* ok = static_cast<Key*>(out_k.request().ptr);
    double* ov = static_cast<double*>(out_v.request().ptr);
    std::uint32_t* os = static_cast<std::uint32_t*>(out_s.request().ptr);

    MergeSource<Key> merge(child_ptrs);
    std::size_t i = 0;
    while (auto e = merge.next()) {
        ok[i] = e->key;
        ov[i] = e->value;
        os[i] = e->source;
        ++i;
    }
    return py::make_tuple(out_k, out_v, out_s);
}
```

In `init_bindings_streams`, register:

```cpp
    m.def("_merge_i64", &merge_batch<std::int64_t>,
          py::arg("key_arrays"), py::arg("value_arrays"));
    m.def("_merge_f64", &merge_batch<double>,
          py::arg("key_arrays"), py::arg("value_arrays"));
```

- [ ] **Step 5: Add the Python `merge` wrapper**

Append to `screamer/streams.py`:

```python
def _key_dtype_kind(keys):
    """Return ('f64' | 'i64', normalized_keys) for a single key array."""
    keys = np.asarray(keys)
    if np.issubdtype(keys.dtype, np.floating):
        return "f64", np.ascontiguousarray(keys, dtype=np.float64)
    if keys.dtype.kind == "M":
        keys = keys.view("int64")
    return "i64", np.ascontiguousarray(keys, dtype=np.int64)


def merge(*series):
    """Merge N (keys, values) series into one key-sorted (keys, values, sources).

    Each series must be individually sorted by key. `sources[i]` is the index
    of the series that emitted event i. Ties break by series order.
    """
    kinds = set()
    norm_keys, norm_vals = [], []
    for keys, values in series:
        kind, k = _key_dtype_kind(keys)
        kinds.add(kind)
        norm_keys.append(k)
        norm_vals.append(np.ascontiguousarray(values, dtype=np.float64))
    if len(kinds) != 1:
        raise TypeError("merge: all series must share one key type (all int/datetime or all float)")
    kind = kinds.pop()
    fn = _b._merge_f64 if kind == "f64" else _b._merge_i64
    return fn(norm_keys, norm_vals)
```

- [ ] **Step 6: Build and run tests**

Run: `make build && make install-dev && poetry run pytest tests/test_streams_merge.py -v`
Expected: all merge tests PASS.

- [ ] **Step 7: Commit**

```bash
git add include/screamer/streams/merge_source.h bindings/bindings_streams.cpp screamer/streams.py tests/test_streams_merge.py
git commit -m "feat(streams): k-way MergeSource + batch merge binding"
```

---

### Task 2: pull iterator + cross-mode identity

**Files:**
- Modify: `bindings/bindings_streams.cpp`
- Modify: `screamer/streams.py`
- Test: `tests/test_streams_merge.py`

**Interfaces:**
- Consumes: `MergeSource<Key>`, `VectorSource<Key>`.
- Produces:
  - Binding: a picklable-free pull object `_MergePuller_i64`/`_MergePuller_f64` constructed from `(list[keys], list[values])`, exposing `next()` that returns a `tuple(key, value, source)` or `None` at exhaustion. It **owns** its child `VectorSource`s and copies of the numpy arrays so they outlive iteration.
  - Python: `streams.merge_iter(*series)` yielding `(key, value, source)` tuples; `streams._merge_events(*series)` returning a list of those tuples (helper for tests).

- [ ] **Step 1: Write the failing test (cross-mode identity)**

Append to `tests/test_streams_merge.py`:

```python
def test_pull_iter_matches_batch_identity():
    rng = np.random.default_rng(7)
    a_k = np.sort(rng.integers(0, 1000, size=200)).astype(np.int64)
    a_v = rng.standard_normal(200)
    b_k = np.sort(rng.integers(0, 1000, size=150)).astype(np.int64)
    b_v = rng.standard_normal(150)

    bk, bv, bs = streams.merge((a_k, a_v), (b_k, b_v))              # batch
    events = list(streams.merge_iter((a_k, a_v), (b_k, b_v)))       # streaming pull

    got_k = np.array([e[0] for e in events], dtype=np.int64)
    got_v = np.array([e[1] for e in events])
    got_s = np.array([e[2] for e in events], dtype=np.uint32)
    np.testing.assert_array_equal(got_k, bk)
    np.testing.assert_array_equal(got_v, bv)
    np.testing.assert_array_equal(got_s, bs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_streams_merge.py::test_pull_iter_matches_batch_identity -v`
Expected: FAIL — `module 'screamer.streams' has no attribute 'merge_iter'`.

- [ ] **Step 3: Add the pull-object binding**

In `bindings/bindings_streams.cpp`, add a class that owns its arrays and sources so lifetime is safe when driven one event at a time from Python:

```cpp
template <class Key>
class MergePuller {
public:
    MergePuller(py::list key_arrays, py::list value_arrays) {
        std::size_t n_children = key_arrays.size();
        if (value_arrays.size() != n_children) {
            throw std::runtime_error("merge: keys/values list length mismatch");
        }
        for (std::size_t i = 0; i < n_children; ++i) {
            // Keep owning copies so the buffers outlive iteration.
            keys_.push_back(py::cast<py::array_t<Key>>(key_arrays[i]));
            vals_.push_back(py::cast<py::array_t<double>>(value_arrays[i]));
        }
        for (std::size_t i = 0; i < n_children; ++i) {
            auto kinfo = keys_[i].request();
            auto vinfo = vals_[i].request();
            if (kinfo.shape[0] != vinfo.shape[0]) {
                throw std::runtime_error("merge: a child's keys/values length differ");
            }
            std::size_t n = static_cast<std::size_t>(vinfo.shape[0]);
            sources_.push_back(std::make_unique<VectorSource<Key>>(
                static_cast<const Key*>(kinfo.ptr),
                static_cast<const double*>(vinfo.ptr), n));
            child_ptrs_.push_back(sources_.back().get());
        }
        merge_ = std::make_unique<MergeSource<Key>>(child_ptrs_);
    }

    py::object next() {
        if (auto e = merge_->next()) {
            return py::make_tuple(e->key, e->value, e->source);
        }
        return py::none();
    }

private:
    std::vector<py::array_t<Key>> keys_;
    std::vector<py::array_t<double>> vals_;
    std::vector<std::unique_ptr<VectorSource<Key>>> sources_;
    std::vector<Source<Key>*> child_ptrs_;
    std::unique_ptr<MergeSource<Key>> merge_;
};
```

Bind both in `init_bindings_streams`:

```cpp
    py::class_<MergePuller<std::int64_t>>(m, "_MergePuller_i64")
        .def(py::init<py::list, py::list>())
        .def("next", &MergePuller<std::int64_t>::next);
    py::class_<MergePuller<double>>(m, "_MergePuller_f64")
        .def(py::init<py::list, py::list>())
        .def("next", &MergePuller<double>::next);
```

- [ ] **Step 4: Add the Python `merge_iter` generator**

Append to `screamer/streams.py`:

```python
def _make_merge_puller(series):
    kinds = set()
    norm_keys, norm_vals = [], []
    for keys, values in series:
        kind, k = _key_dtype_kind(keys)
        kinds.add(kind)
        norm_keys.append(k)
        norm_vals.append(np.ascontiguousarray(values, dtype=np.float64))
    if len(kinds) != 1:
        raise TypeError("merge: all series must share one key type")
    kind = kinds.pop()
    cls = _b._MergePuller_f64 if kind == "f64" else _b._MergePuller_i64
    return cls(norm_keys, norm_vals)


def merge_iter(*series):
    """Yield (key, value, source) events in key order, pulled one at a time."""
    puller = _make_merge_puller(series)
    while True:
        event = puller.next()
        if event is None:
            return
        yield event
```

- [ ] **Step 5: Build and run tests**

Run: `make build && make install-dev && poetry run pytest tests/test_streams_merge.py -v`
Expected: all PASS, including the identity test.

- [ ] **Step 6: Commit**

```bash
git add bindings/bindings_streams.cpp screamer/streams.py tests/test_streams_merge.py
git commit -m "feat(streams): merge pull iterator + batch/stream identity test"
```

---

### Task 3: `pace` — wall-clock replay driver

**Files:**
- Modify: `screamer/streams.py`
- Test: `tests/test_streams_merge.py`

**Interfaces:**
- Consumes: `streams.merge_iter`.
- Produces: `streams.pace(*series, speed=1.0, sleep=None) -> async generator` yielding `(key, value, source)`; sleeps `max(0, key_delta / speed)` between events using `sleep` (default `asyncio.sleep`). `speed=float("inf")` yields with no sleeping (backtest). The injectable `sleep` makes pacing deterministically testable and keeps values/order untouched.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_streams_merge.py`:

```python
import asyncio


def _drain(agen):
    async def run():
        out = []
        async for e in agen:
            out.append(e)
        return out
    return asyncio.run(run())


def test_pace_preserves_order_and_scales_sleeps():
    a_k = np.array([0, 10, 30], dtype=np.int64)     # deltas 10, 20
    a_v = np.array([0.0, 1.0, 3.0])
    b_k = np.array([5, 20], dtype=np.int64)          # interleaves -> 0,5,10,20,30
    b_v = np.array([0.5, 2.0])

    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    events = _drain(streams.pace((a_k, a_v), (b_k, b_v), speed=2.0, sleep=fake_sleep))

    # Order identical to a plain merge
    bk, bv, bs = streams.merge((a_k, a_v), (b_k, b_v))
    np.testing.assert_array_equal(np.array([e[0] for e in events], dtype=np.int64), bk)
    np.testing.assert_array_equal(np.array([e[1] for e in events]), bv)

    # Sleeps == successive key deltas / speed (first event has no preceding sleep).
    # merged keys: 0,5,10,20,30 -> deltas 5,5,10,10 -> /2.0 -> 2.5,2.5,5,5
    assert slept == [2.5, 2.5, 5.0, 5.0]


def test_pace_infinite_speed_no_sleep():
    a_k = np.array([0, 100], dtype=np.int64)
    a_v = np.array([0.0, 1.0])
    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    events = _drain(streams.pace((a_k, a_v), speed=float("inf"), sleep=fake_sleep))
    assert [e[0] for e in events] == [0, 100]
    assert slept == []          # no pacing at infinite speed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_streams_merge.py::test_pace_preserves_order_and_scales_sleeps -v`
Expected: FAIL — `module 'screamer.streams' has no attribute 'pace'`.

- [ ] **Step 3: Implement `pace`**

Append to `screamer/streams.py` (add `import asyncio` at the top of the file with the other imports):

```python
async def pace(*series, speed=1.0, sleep=None):
    """Replay merged series as an async event stream paced by key-deltas.

    Yields (key, value, source) in key order. Between consecutive events it
    awaits `sleep(key_delta / speed)` so wall-clock spacing tracks the key
    spacing. speed=inf disables pacing (backtest at max speed). Pacing never
    changes values or order. `sleep` is injectable for testing; defaults to
    asyncio.sleep. Requires a metric (subtractable) key.
    """
    if sleep is None:
        sleep = asyncio.sleep
    infinite = speed == float("inf")
    prev_key = None
    for key, value, source in merge_iter(*series):
        if not infinite and prev_key is not None:
            delta = key - prev_key
            wait = delta / speed
            if wait > 0:
                await sleep(wait)
        prev_key = key
        yield key, value, source
```

- [ ] **Step 4: Build/run tests** (no C++ change, but run install-dev if import is stale)

Run: `poetry run pytest tests/test_streams_merge.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add screamer/streams.py tests/test_streams_merge.py
git commit -m "feat(streams): pace async replay driver with injectable clock"
```

---

### Task 4: efficiency — drop the dead key buffer in `run_chain`

Addresses the Minor logged across Plan 1 (the `out_k` buffer was always allocated and written even when the caller discards keys). Directly serves the high-speed positioning: no allocation or per-event write that is thrown away.

**Files:**
- Modify: `bindings/bindings_streams.cpp`
- Modify: `screamer/streams.py`
- Test: `tests/test_streams_core.py`

**Interfaces:**
- Consumes: Plan 1 `run_chain<Key>`, `CollectorSink<Key>`, `FunctorNode<Key>`.
- Produces: `run_chain<Key>(fns, keys, values, return_keys)` — when `return_keys` is false it allocates and writes only the value buffer and returns a bare `py::array_t<double>`; when true it returns `(keys, values)`. `_run_chain` passes `return_keys` down.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_streams_core.py`:

```python
def test_run_chain_values_only_still_correct():
    # return_keys=False path must produce identical values to the keyed path.
    x = np.random.default_rng(11).standard_normal(256)
    t = (np.arange(x.size, dtype=np.int64) * 3) + 1
    vals_only = streams._run_chain([RollingMean(4)], x, keys=t)                 # default False
    keys_out, vals_keyed = streams._run_chain([RollingMean(4)], x, keys=t, return_keys=True)
    np.testing.assert_array_equal(vals_only, vals_keyed)
    np.testing.assert_array_equal(keys_out, t)
```

Note: this test passes on the current tuple-returning code too; its purpose is to lock behavior while the C++ is changed to skip the key buffer. Verify it stays green through the refactor.

- [ ] **Step 2: Run the test (baseline green)**

Run: `poetry run pytest tests/test_streams_core.py::test_run_chain_values_only_still_correct -v`
Expected: PASS (behavioral lock before refactor).

- [ ] **Step 3: Add a values-only collector and split the C++ paths**

In `include/screamer/streams/collector_sink.h`, add a sink that writes only values:

```cpp
// Terminal sink that keeps only values (used when the caller discards keys).
template <class Key>
class ValueCollectorSink : public Sink<Key> {
public:
    explicit ValueCollectorSink(double* out_values) : ov_(out_values), n_(0) {}
    void push(const Event<Key>& e) override { ov_[n_] = e.value; ++n_; }
    std::size_t count() const { return n_; }
private:
    double* ov_;
    std::size_t n_;
};
```

In `bindings/bindings_streams.cpp`, change `run_chain<Key>` to take a `bool return_keys` and branch the terminal sink (shared wiring extracted so the functor-chain construction is not duplicated):

```cpp
template <class Key>
static py::object run_chain(std::vector<ScreamerBase*> fns,
                           py::array_t<Key> keys,
                           py::array_t<double> values,
                           bool return_keys) {
    auto vinfo = values.request();
    auto kinfo = keys.request();
    if (kinfo.shape[0] < vinfo.shape[0]) {
        throw std::runtime_error("run_chain: keys array is shorter than values array");
    }
    std::size_t n = static_cast<std::size_t>(vinfo.shape[0]);
    const Key* kptr = static_cast<const Key*>(kinfo.ptr);
    const double* vptr = static_cast<const double*>(vinfo.ptr);

    py::array_t<double> out_v(n);
    double* ov = static_cast<double*>(out_v.request().ptr);

    // Wire the functor chain in front of the chosen terminal sink.
    auto drive = [&](Sink<Key>& terminal) {
        Sink<Key>* downstream = &terminal;
        std::vector<std::unique_ptr<FunctorNode<Key>>> nodes;
        for (auto it = fns.rbegin(); it != fns.rend(); ++it) {
            (*it)->reset();
            nodes.push_back(std::make_unique<FunctorNode<Key>>(**it, *downstream));
            downstream = nodes.back().get();
        }
        VectorSource<Key> src(kptr, vptr, n);
        run_batch<Key>(src, *downstream);
        for (auto* f : fns) f->reset();
    };

    if (return_keys) {
        py::array_t<Key> out_k(n);
        Key* ok = static_cast<Key*>(out_k.request().ptr);
        CollectorSink<Key> collector(ok, ov);
        drive(collector);
        return py::make_tuple(out_k, out_v);
    }
    ValueCollectorSink<Key> collector(ov);
    drive(collector);
    return out_v;
}
```

Update the bindings to pass `return_keys`:

```cpp
    m.def("_run_chain_i64", &run_chain<std::int64_t>,
          py::arg("functors"), py::arg("keys"), py::arg("values"),
          py::arg("return_keys") = false);
    m.def("_run_chain_f64", &run_chain<double>,
          py::arg("functors"), py::arg("keys"), py::arg("values"),
          py::arg("return_keys") = false);
```

- [ ] **Step 4: Thread `return_keys` through the Python hook**

In `screamer/streams.py`, update `_run_chain` so the binding calls pass `return_keys` and unpack accordingly:

```python
def _run_chain(functors, values, keys=None, return_keys=False):
    values = np.ascontiguousarray(values, dtype=np.float64)
    n = values.shape[0]
    functors = list(functors)

    if keys is None:
        keys = np.arange(n, dtype=np.int64)
        kind = "i64"
    else:
        kind, keys = _key_dtype_kind(keys)

    fn = _b._run_chain_f64 if kind == "f64" else _b._run_chain_i64
    result = fn(functors, keys, values, return_keys)
    return result  # tuple (keys, values) if return_keys else values array
```

- [ ] **Step 5: Build and run the full core + merge suites**

Run: `make build && make install-dev && poetry run pytest tests/test_streams_core.py tests/test_streams_merge.py -v`
Expected: all PASS (the values-only path returns the same numbers; keyed path unchanged).

- [ ] **Step 6: Commit**

```bash
git add include/screamer/streams/collector_sink.h bindings/bindings_streams.cpp screamer/streams.py tests/test_streams_core.py
git commit -m "perf(streams): skip key buffer allocation when keys are discarded"
```

---

## Self-Review

**1. Spec coverage (Plan 2 = decomposition item 2 + logged efficiency debt):**
- `merge` k-way heap, source-tagged, deterministic tie-break → Task 1 (`merge_source.h`), tests for order/ties/tags/float keys. ✓
- Batch merge exact-length preallocation (no realloc) → `merge_batch` sizes `total` up front. ✓
- Pull iterator + batch==stream identity → Task 2, `test_pull_iter_matches_batch_identity`. ✓
- `pace` wall-clock replay, `speed=`, backtest at `speed=inf`, values/order untouched, injectable clock → Task 3. ✓
- Metric-key requirement for pacing (subtraction) documented in `pace` docstring. ✓
- Efficiency debt (dead key buffer) → Task 4, values-only sink. ✓
- Deferred (correctly absent): `combine_latest`/`filter`/`split` (Plan 3), docs page (Plan 4).

**2. Placeholder scan:** none — every step has concrete code or an exact command.

**3. Type consistency:** `MergeSource<Key>(vector<Source<Key>*>)`, `Event::source` (uint32), `_merge_i64/_f64 -> (keys,values,sources)`, `_MergePuller_i64/_f64.next()->tuple|None`, `merge`/`merge_iter`/`pace`, and `run_chain(..., return_keys)` are used consistently across tasks. `_key_dtype_kind` is defined in Task 1 and reused by Tasks 2 and 4.

---

## Follow-on

- **Plan 3 — interior combinators:** `CombineLatest<N>` (fused N-input, `emit="when_all"`/`"on_any"`), `Filter`/`Dropna`, `Split`; first public API + `generate_screamer__init__.py` change to export `streams` names.
- **Plan 4 — docs + identity matrix:** `docs/multistream.md`, cross-links, batch==stream==replay test matrix across all combinators.
