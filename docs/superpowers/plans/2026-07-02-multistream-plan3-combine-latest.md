# Multi-stream Plan 3 — combine_latest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `combine_latest` — the as-of latest-value join that aligns N key-sorted streams into one aligned record stream (emit on any input, carry each input's most recent value), realizing the `X(combine_latest(a, b))` idiom.

**Architecture:** `combine_latest` is built directly on Plan 2's `MergeSource`: merge the N sources into one key-ordered tagged stream, then a small `CombineLatest<Key>` operator maintains `latest[N]`/`seen[N]` and, on each tagged event, emits the aligned latest-value vector (respecting the firing rule). Batch returns `(keys, aligned[M, N])`; a pull iterator gives the streaming form; both drive the identical `MergeSource`+`CombineLatest`, so batch and stream are byte-identical. Alignment stays separate from computation: the aligned columns feed existing N-input functors (e.g. `RollingCorr`) unchanged.

**Tech Stack:** C++17, pybind11, numpy, pytest.

## Global Constraints

- **Firing rule:** default `emit="when_all"` (suppress output until every input has produced ≥1 value); opt-in `emit="on_any"` (emit from the first event; not-yet-seen inputs are `NaN`). `latest` initializes to `NaN` so `on_any` warmup is `NaN` exactly.
- **As-of / forward-fill carry:** each emitted row holds every input's most recent value at that key (last-known-value). This is causal — only past/current events, never lookahead.
- **Cross-mode identity:** batch `combine_latest` and streaming `combine_latest_iter` emit byte-identical `(key, aligned-row)` sequences (`np.testing.assert_array_equal`, NaN-aware where needed). Both drive the same `MergeSource`+`CombineLatest`.
- **Alignment ≠ computation:** `combine_latest` only aligns; it does not reduce. Its aligned columns are consumed by existing functors unchanged. An optional Python `func` is a convenience for ad-hoc per-row reduction only.
- **Deterministic tie-break** inherited from `MergeSource`: equal keys resolve by source order; at a shared key the row reflects all updates at that key in source order (the last source's value wins for that step, consistent with the merge order).
- **Numeric keys only** (`int64`/`double`), one key type per call; value always `double`; new bindings underscore-prefixed (internal); compute functors never modified; never hand-edit `screamer/__init__.py` or version files.
- **Efficiency:** accumulate into `std::vector` with `reserve(total)`/`reserve(total*N)` (no reallocation growth), then build the exact-size numpy result once; per-event work stays in C++.
- Build: `make build` **then `make install-dev`**. Test: `poetry run pytest tests/test_streams_combine.py -v`.

---

## File Structure

- `include/screamer/streams/combine_latest.h` — `CombineLatest<Key>` operator (latest/seen state + firing rule).
- `bindings/bindings_streams.cpp` (modify) — `_combine_latest_i64`/`_f64` (batch) and `_CombineLatestPuller_i64`/`_f64` (pull).
- `screamer/streams.py` (modify) — `combine_latest(*series, emit="when_all", func=None)`, `combine_latest_iter(*series, emit="when_all")`.
- `tests/test_streams_combine.py` — firing rules, alignment vs numpy reference, cross-mode identity, the `RollingCorr` idiom.

---

### Task 1: `CombineLatest` operator + batch binding

**Files:**
- Create: `include/screamer/streams/combine_latest.h`
- Modify: `bindings/bindings_streams.cpp`
- Modify: `screamer/streams.py`
- Test: `tests/test_streams_combine.py`

**Interfaces:**
- Consumes: `MergeSource<Key>`, `VectorSource<Key>`, `Event<Key>` (Plan 1/2).
- Produces:
  - C++: `CombineLatest<Key>(size_t n, bool when_all)`; `bool on_event(uint32_t source, double value)` (returns whether to emit); `const std::vector<double>& latest() const`.
  - Binding: `_combine_latest_i64(list[keys], list[values], bool when_all) -> (keys[M], aligned[M,N])`; `_f64` identical with float64 keys.
  - Python: `combine_latest(*series, emit="when_all", func=None)`; returns `(keys, aligned)` where `aligned` is `(M, N)`, or `(keys, reduced)` with `reduced` `(M,)` when `func` is given.

- [ ] **Step 1: Write the failing test**

Create `tests/test_streams_combine.py`:

```python
import numpy as np
from screamer import streams


def _ref_combine_latest(series, when_all):
    """Reference as-of join: merge by (key, source), carry last value per source."""
    events = []
    for i, (k, v) in enumerate(series):
        for kk, vv in zip(np.asarray(k), np.asarray(v)):
            events.append((kk, i, float(vv)))
    events.sort(key=lambda e: (e[0], e[1]))          # stable: key then source
    n = len(series)
    latest = [np.nan] * n
    seen = [False] * n
    out_k, out_rows = [], []
    for kk, src, vv in events:
        latest[src] = vv
        seen[src] = True
        if when_all and not all(seen):
            continue
        out_k.append(kk)
        out_rows.append(list(latest))
    keys = np.array(out_k)
    rows = np.array(out_rows, dtype=np.float64).reshape(len(out_k), n)
    return keys, rows


def test_combine_latest_when_all_default():
    a_k = np.array([1, 3, 5], dtype=np.int64)
    a_v = np.array([10.0, 30.0, 50.0])
    b_k = np.array([2, 4], dtype=np.int64)
    b_v = np.array([20.0, 40.0])
    got_k, got_a = streams.combine_latest((a_k, a_v), (b_k, b_v))   # default when_all
    exp_k, exp_a = _ref_combine_latest([(a_k, a_v), (b_k, b_v)], when_all=True)
    np.testing.assert_array_equal(got_k, exp_k)
    np.testing.assert_array_equal(got_a, exp_a)


def test_combine_latest_on_any_warmup_is_nan():
    a_k = np.array([1, 3, 5], dtype=np.int64)
    a_v = np.array([10.0, 30.0, 50.0])
    b_k = np.array([2, 4], dtype=np.int64)
    b_v = np.array([20.0, 40.0])
    got_k, got_a = streams.combine_latest((a_k, a_v), (b_k, b_v), emit="on_any")
    exp_k, exp_a = _ref_combine_latest([(a_k, a_v), (b_k, b_v)], when_all=False)
    np.testing.assert_array_equal(got_k, exp_k)
    np.testing.assert_array_equal(got_a, exp_a)         # first row has NaN for b


def test_combine_latest_float_keys_three_series():
    rng = np.random.default_rng(3)
    series = []
    for _ in range(3):
        k = np.sort(rng.uniform(0, 100, size=40))
        v = rng.standard_normal(40)
        series.append((k, v))
    got_k, got_a = streams.combine_latest(*series)
    exp_k, exp_a = _ref_combine_latest(series, when_all=True)
    np.testing.assert_array_equal(got_k, exp_k)
    np.testing.assert_array_equal(got_a, exp_a)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_streams_combine.py -v`
Expected: FAIL — `module 'screamer.streams' has no attribute 'combine_latest'`.

- [ ] **Step 3: Create the `CombineLatest` operator**

`include/screamer/streams/combine_latest.h`:

```cpp
#ifndef SCREAMER_STREAMS_COMBINE_LATEST_H
#define SCREAMER_STREAMS_COMBINE_LATEST_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace screamer { namespace streams {

// As-of latest-value join state for N sources fed a tagged event stream (in
// key order). on_event() updates the emitting source's latest value and marks
// it seen; it returns whether an aligned row should be emitted now:
//   when_all: only once every source has produced at least one value
//   on_any:   always (not-yet-seen sources read as NaN)
// latest() is the N-wide aligned row valid immediately after on_event().
class CombineLatest {
public:
    CombineLatest(std::size_t n, bool when_all)
        : latest_(n, std::numeric_limits<double>::quiet_NaN()),
          seen_(n, false), n_(n), seen_count_(0), when_all_(when_all) {}

    bool on_event(std::uint32_t source, double value) {
        if (!seen_[source]) { seen_[source] = true; ++seen_count_; }
        latest_[source] = value;
        if (when_all_ && seen_count_ < n_) return false;
        return true;
    }

    const std::vector<double>& latest() const { return latest_; }

private:
    std::vector<double> latest_;
    std::vector<bool> seen_;
    std::size_t n_;
    std::size_t seen_count_;
    bool when_all_;
};

}} // namespace screamer::streams
#endif
```

- [ ] **Step 4: Add the batch binding**

In `bindings/bindings_streams.cpp`, add includes near the top:

```cpp
#include <cstring>
#include "screamer/streams/combine_latest.h"
```

Add the batch helper (accumulate with reserve, build exact-size result once):

```cpp
template <class Key>
static py::tuple combine_latest_batch(py::list key_arrays,
                                      py::list value_arrays,
                                      bool when_all) {
    std::size_t n = key_arrays.size();
    if (value_arrays.size() != n) {
        throw std::runtime_error("combine_latest: keys/values list length mismatch");
    }
    if (n == 0) {
        throw std::runtime_error("combine_latest: needs at least one series");
    }

    std::vector<py::array_t<Key>> keys;
    std::vector<py::array_t<double>> vals;
    keys.reserve(n);
    vals.reserve(n);
    std::vector<std::unique_ptr<VectorSource<Key>>> sources;
    std::vector<Source<Key>*> child_ptrs;
    sources.reserve(n);
    child_ptrs.reserve(n);
    std::size_t total = 0;

    for (std::size_t i = 0; i < n; ++i) {
        keys.push_back(py::cast<py::array_t<Key>>(key_arrays[i]));
        vals.push_back(py::cast<py::array_t<double>>(value_arrays[i]));
        auto kinfo = keys[i].request();
        auto vinfo = vals[i].request();
        if (kinfo.shape[0] != vinfo.shape[0]) {
            throw std::runtime_error("combine_latest: a child's keys/values length differ");
        }
        std::size_t len = static_cast<std::size_t>(kinfo.shape[0]);
        total += len;
        sources.push_back(std::make_unique<VectorSource<Key>>(
            static_cast<const Key*>(kinfo.ptr),
            static_cast<const double*>(vinfo.ptr), len));
        child_ptrs.push_back(sources.back().get());
    }

    std::vector<Key> out_k;
    std::vector<double> out_v;
    out_k.reserve(total);
    out_v.reserve(total * n);

    CombineLatest cl(n, when_all);
    MergeSource<Key> merge(child_ptrs);
    while (auto e = merge.next()) {
        if (cl.on_event(e->source, e->value)) {
            out_k.push_back(e->key);
            const std::vector<double>& row = cl.latest();
            out_v.insert(out_v.end(), row.begin(), row.end());
        }
    }

    std::size_t m = out_k.size();
    py::array_t<Key> rk(static_cast<py::ssize_t>(m));
    if (m) std::memcpy(rk.request().ptr, out_k.data(), m * sizeof(Key));
    py::array_t<double> rv({static_cast<py::ssize_t>(m), static_cast<py::ssize_t>(n)});
    if (m) std::memcpy(rv.request().ptr, out_v.data(), m * n * sizeof(double));
    return py::make_tuple(rk, rv);
}
```

Register in `init_bindings_streams`:

```cpp
    m.def("_combine_latest_i64", &combine_latest_batch<std::int64_t>,
          py::arg("key_arrays"), py::arg("value_arrays"), py::arg("when_all"));
    m.def("_combine_latest_f64", &combine_latest_batch<double>,
          py::arg("key_arrays"), py::arg("value_arrays"), py::arg("when_all"));
```

- [ ] **Step 5: Add the Python `combine_latest` wrapper**

Append to `screamer/streams.py` (reuses `_normalize_series` from Plan 2):

```python
def combine_latest(*series, emit="when_all", func=None):
    """As-of latest-value join of N (keys, values) series.

    Emits an aligned row whenever any input advances, carrying each input's most
    recent value (forward-fill). Returns (keys, aligned) where aligned is (M, N);
    aligned[:, j] is series j's latest value at each emitted key. emit="when_all"
    (default) suppresses output until every input is warm; emit="on_any" emits
    from the first event with NaN for not-yet-seen inputs. If `func` is given it is
    applied per row (func(*row)) and (keys, reduced) is returned instead.
    """
    if emit not in ("when_all", "on_any"):
        raise ValueError('combine_latest: emit must be "when_all" or "on_any"')
    kind, norm_keys, norm_vals = _normalize_series(series, "combine_latest")
    fn = _b._combine_latest_f64 if kind == "f64" else _b._combine_latest_i64
    keys, aligned = fn(norm_keys, norm_vals, emit == "when_all")
    if func is None:
        return keys, aligned
    reduced = np.array([func(*row) for row in aligned], dtype=np.float64)
    return keys, reduced
```

- [ ] **Step 6: Build and run tests**

Run: `make build && make install-dev && poetry run pytest tests/test_streams_combine.py -v`
Expected: all three tests PASS.

- [ ] **Step 7: Commit**

```bash
git add include/screamer/streams/combine_latest.h bindings/bindings_streams.cpp screamer/streams.py tests/test_streams_combine.py
git commit -m "feat(streams): combine_latest as-of join (batch, aligned output)"
```

---

### Task 2: pull iterator + cross-mode identity

**Files:**
- Modify: `bindings/bindings_streams.cpp`
- Modify: `screamer/streams.py`
- Test: `tests/test_streams_combine.py`

**Interfaces:**
- Consumes: `CombineLatest`, `MergeSource<Key>`, `VectorSource<Key>`.
- Produces:
  - Binding: `_CombineLatestPuller_i64`/`_f64`, constructed from `(list[keys], list[values], bool when_all)`; owns its arrays + sources; `next()` returns `(key, row_tuple)` or `None`, where `row_tuple` is the N latest values.
  - Python: `combine_latest_iter(*series, emit="when_all")` yielding `(key, tuple_of_N_values)`.

- [ ] **Step 1: Write the failing test (cross-mode identity)**

Append to `tests/test_streams_combine.py`:

```python
def test_combine_latest_iter_matches_batch_identity():
    rng = np.random.default_rng(9)
    series = []
    for _ in range(3):
        k = np.sort(rng.integers(0, 500, size=120)).astype(np.int64)
        v = rng.standard_normal(120)
        series.append((k, v))

    bk, ba = streams.combine_latest(*series)                       # batch
    events = list(streams.combine_latest_iter(*series))            # streaming pull

    got_k = np.array([e[0] for e in events], dtype=np.int64)
    got_a = np.array([list(e[1]) for e in events], dtype=np.float64).reshape(len(events), 3)
    np.testing.assert_array_equal(got_k, bk)
    np.testing.assert_array_equal(got_a, ba)


def test_combine_latest_iter_on_any_identity():
    a_k = np.array([1, 4], dtype=np.int64); a_v = np.array([1.0, 4.0])
    b_k = np.array([2, 3], dtype=np.int64); b_v = np.array([2.0, 3.0])
    bk, ba = streams.combine_latest((a_k, a_v), (b_k, b_v), emit="on_any")
    events = list(streams.combine_latest_iter((a_k, a_v), (b_k, b_v), emit="on_any"))
    got_k = np.array([e[0] for e in events], dtype=np.int64)
    got_a = np.array([list(e[1]) for e in events], dtype=np.float64).reshape(len(events), 2)
    np.testing.assert_array_equal(got_k, bk)
    np.testing.assert_array_equal(got_a, ba)      # NaN warmup identical across modes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_streams_combine.py::test_combine_latest_iter_matches_batch_identity -v`
Expected: FAIL — `module 'screamer.streams' has no attribute 'combine_latest_iter'`.

- [ ] **Step 3: Add the pull-object binding**

In `bindings/bindings_streams.cpp`, add (owns arrays + sources like `MergePuller`; declaration order keeps lifetime safe):

```cpp
template <class Key>
class CombineLatestPuller {
public:
    CombineLatestPuller(py::list key_arrays, py::list value_arrays, bool when_all)
        : n_(key_arrays.size()), cl_(key_arrays.size(), when_all) {
        if (value_arrays.size() != n_) {
            throw std::runtime_error("combine_latest: keys/values list length mismatch");
        }
        if (n_ == 0) {
            throw std::runtime_error("combine_latest: needs at least one series");
        }
        for (std::size_t i = 0; i < n_; ++i) {
            keys_.push_back(py::cast<py::array_t<Key>>(key_arrays[i]));
            vals_.push_back(py::cast<py::array_t<double>>(value_arrays[i]));
        }
        std::vector<Source<Key>*> child_ptrs;
        child_ptrs.reserve(n_);
        for (std::size_t i = 0; i < n_; ++i) {
            auto kinfo = keys_[i].request();
            auto vinfo = vals_[i].request();
            if (kinfo.shape[0] != vinfo.shape[0]) {
                throw std::runtime_error("combine_latest: a child's keys/values length differ");
            }
            std::size_t len = static_cast<std::size_t>(kinfo.shape[0]);
            sources_.push_back(std::make_unique<VectorSource<Key>>(
                static_cast<const Key*>(kinfo.ptr),
                static_cast<const double*>(vinfo.ptr), len));
            child_ptrs.push_back(sources_.back().get());
        }
        merge_ = std::make_unique<MergeSource<Key>>(child_ptrs);
    }

    py::object next() {
        while (auto e = merge_->next()) {
            if (cl_.on_event(e->source, e->value)) {
                const std::vector<double>& row = cl_.latest();
                py::tuple t(row.size());
                for (std::size_t j = 0; j < row.size(); ++j) t[j] = row[j];
                return py::make_tuple(e->key, t);
            }
        }
        return py::none();
    }

private:
    std::size_t n_;
    std::vector<py::array_t<Key>> keys_;
    std::vector<py::array_t<double>> vals_;
    std::vector<std::unique_ptr<VectorSource<Key>>> sources_;
    std::unique_ptr<MergeSource<Key>> merge_;
    CombineLatest cl_;
};
```

Bind both in `init_bindings_streams`:

```cpp
    py::class_<CombineLatestPuller<std::int64_t>>(m, "_CombineLatestPuller_i64")
        .def(py::init<py::list, py::list, bool>())
        .def("next", &CombineLatestPuller<std::int64_t>::next);
    py::class_<CombineLatestPuller<double>>(m, "_CombineLatestPuller_f64")
        .def(py::init<py::list, py::list, bool>())
        .def("next", &CombineLatestPuller<double>::next);
```

Note member declaration order: `keys_`, `vals_`, `sources_`, `merge_`, `cl_` — `merge_` (holds `Source*` into `sources_`) is destroyed before `sources_`, which is destroyed before the arrays. Keep this order.

- [ ] **Step 4: Add the Python `combine_latest_iter`**

Append to `screamer/streams.py`:

```python
def combine_latest_iter(*series, emit="when_all"):
    """Yield (key, (v0, v1, ...)) aligned rows one at a time (streaming form)."""
    if emit not in ("when_all", "on_any"):
        raise ValueError('combine_latest: emit must be "when_all" or "on_any"')
    kind, norm_keys, norm_vals = _normalize_series(series, "combine_latest")
    cls = _b._CombineLatestPuller_f64 if kind == "f64" else _b._CombineLatestPuller_i64
    puller = cls(norm_keys, norm_vals, emit == "when_all")
    while True:
        event = puller.next()
        if event is None:
            return
        yield event
```

- [ ] **Step 5: Build and run tests**

Run: `make build && make install-dev && poetry run pytest tests/test_streams_combine.py -v`
Expected: all PASS, including both identity tests.

- [ ] **Step 6: Commit**

```bash
git add bindings/bindings_streams.cpp screamer/streams.py tests/test_streams_combine.py
git commit -m "feat(streams): combine_latest pull iterator + batch/stream identity"
```

---

### Task 3: the `RollingCorr(combine_latest(...))` idiom

Proves the whole point: aligned columns feed an existing unchanged N-input functor, and a `func` reducer works for ad-hoc cases.

**Files:**
- Test: `tests/test_streams_combine.py`

**Interfaces:**
- Consumes: `streams.combine_latest`, and existing functors `RollingCorr`, `RollingSpread` from `screamer`.
- Produces: no new code — end-to-end idiom tests that lock the alignment→computation contract.

- [ ] **Step 1: Write the idiom tests**

Append to `tests/test_streams_combine.py`:

```python
from screamer import RollingCorr


def test_rollingcorr_over_combine_latest():
    # Two async series -> align -> feed existing 2-input functor unchanged.
    rng = np.random.default_rng(21)
    a_k = np.sort(rng.integers(0, 2000, size=300)).astype(np.int64)
    a_v = rng.standard_normal(300)
    b_k = np.sort(rng.integers(0, 2000, size=250)).astype(np.int64)
    b_v = rng.standard_normal(250)

    keys, aligned = streams.combine_latest((a_k, a_v), (b_k, b_v))   # when_all
    assert aligned.shape[1] == 2
    # The idiom: existing functor consumes the aligned columns, untouched.
    corr = RollingCorr(20)(aligned[:, 0], aligned[:, 1])
    # Equivalent to calling the functor on the two aligned columns directly.
    exp = RollingCorr(20)(np.ascontiguousarray(aligned[:, 0]),
                          np.ascontiguousarray(aligned[:, 1]))
    np.testing.assert_array_equal(corr, exp)
    assert corr.shape[0] == keys.shape[0]


def test_combine_latest_func_reducer_spread():
    a_k = np.array([1, 3, 5], dtype=np.int64); a_v = np.array([10.0, 30.0, 50.0])
    b_k = np.array([2, 4], dtype=np.int64);     b_v = np.array([20.0, 40.0])
    keys, spread = streams.combine_latest((a_k, a_v), (b_k, b_v),
                                          func=lambda a, b: a - b)
    _, aligned = streams.combine_latest((a_k, a_v), (b_k, b_v))
    np.testing.assert_array_equal(spread, aligned[:, 0] - aligned[:, 1])
```

- [ ] **Step 2: Run the tests**

Run: `poetry run pytest tests/test_streams_combine.py -v`
Expected: all PASS (no rebuild needed — tests only).

- [ ] **Step 3: Commit**

```bash
git add tests/test_streams_combine.py
git commit -m "test(streams): RollingCorr(combine_latest) idiom + func reducer"
```

---

## Self-Review

**1. Spec coverage (combine_latest portion of the foundation):**
- As-of latest-value join, emit-on-any-event, forward-fill carry → `CombineLatest::on_event` + `latest()`, Task 1. ✓
- Firing rules `when_all` (default) / `on_any` with `NaN` warmup → `when_all` flag + `NaN` init, Tasks 1–2, tested. ✓
- Aligned output, alignment separate from computation → returns `(keys, aligned[M,N])`; idiom test feeds `RollingCorr` unchanged, Task 3. ✓
- Cross-mode identity (batch == stream) → shared `MergeSource`+`CombineLatest`, Task 2 identity tests. ✓
- Optional `func` convenience → Task 1 wrapper + Task 3 test. ✓
- Efficiency (reserve + single exact-size result) → `combine_latest_batch`. ✓
- Deferred (correctly absent): `filter`/`dropna`/`split` and the public-API export (Plan 4); docs page (Plan 5).

**2. Placeholder scan:** none — every step has concrete code or an exact command.

**3. Type consistency:** `CombineLatest(n, when_all)`, `on_event(source,value)->bool`, `latest()`, `_combine_latest_i64/_f64(keys,values,when_all)->(keys,aligned)`, `_CombineLatestPuller_i64/_f64.next()->(key,tuple)|None`, `combine_latest(...emit,func)`, `combine_latest_iter(...emit)` — consistent across tasks. Reuses `_normalize_series` (Plan 2). `emit` validated identically in both wrappers.

**Note (logged debt):** `combine_latest_batch` / `CombineLatestPuller` / `merge_batch` / `MergePuller` share the child-source-construction block. Left inline here to avoid modifying already-merged merge code; consolidate into a `build_vector_sources<Key>` helper in the Plan 4 cleanup.

---

## Follow-on

- **Plan 4 — filter/dropna/split + public API:** cardinality-reducing `filter`/`dropna` (using the `count()` accessors reserved on the sinks), `split`; then export the public `streams` names by teaching `devtools/generate_screamer__init__.py` to include them. Also: consolidate the shared child-source-construction block.
- **Plan 5 — docs + identity matrix:** `docs/multistream.md`, cross-links from `polymorphic_api.md`/`nan_policy.md`, batch==stream==replay matrix across all combinators.
