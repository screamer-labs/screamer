# Shape Contract, Phase 1: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every batch call, iterator of events, and eager stream function in Python and JavaScript read arrays as `(T, *A, F?)`. Multi-input operators then broadcast over assets, a size-1 field axis is consumed, `Resample` and `CombineLatest` take multi-column data, and `reshape`/`layout` let the user name the axes.

**Architecture:** One header, `include/screamer/common/shape.h`, holds every shape rule: layout parsing, the packed-input split, size-1 field consumption, and the asset broadcast as a table of per-column offsets. The Python dispatch (`dispatch.h`, `dispatch.cpp`, `lazy_eval_iterator.h`) and a new WASM binding file (`wasm/shape_bindings.cpp`) both call it. Containers are adapted in thin Python (`screamer/shape.py`, `screamer/streams.py`) and TypeScript (`js/src/runtime.ts`, `js/src/shape.ts`) code. `CombineLatest` (core, eager binding, graph node) learns per-input widths and concatenates fields. `ResampleNode` builds a per-column reducer plan when a single-column aggregation meets a wide frame.

**Tech Stack:** C++17 header-only core, nanobind (Python), Embind/Emscripten (WASM), NumPy, pandas, TypeScript (node:test), pytest.

**Spec:** `docs/superpowers/specs/2026-10-11-shape-contract-design.md`

## Global Constraints

- All operator logic stays in C++; Python and TypeScript only adapt containers and call C++.
- Python and JavaScript behave the same. A feature in one binding but not the other is a tracked gap, listed in the final report.
- Every function is causal, and batch, iterator and live results are identical.
- Do not edit version files (`screamer/__init__.py` `__version__`, `pyproject.toml`, `js/package.json`, `CITATION.cff` or wherever `make patch` writes). There is no release in this plan.
- After any C++ change, run `make install-dev` before running Python tests (a plain `make build` leaves a stale binding).
- JavaScript tests need a fresh WASM build: `cd js && npm run build:wasm && node --import tsx --test test/<file>.test.ts`.
- Docs, docstrings and comments use no em-dashes and follow the project writing style (lead with the point, plain language, say it once).
- Commit on `main` (local merge workflow). Every commit message ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01PZMDN2ijvnSZ7ZnrJUxnd8
  ```
- Treat waste (extra copies, per-event allocation) as a defect: a broadcast input is read through stride tables, never copied K times.

## Decisions this plan makes where the spec is silent

These are recorded here so the reviewer can check them against the spec.

1. **Rule 3 applies only to arrays with at least one asset axis**: a trailing size-1 axis is consumed when `ndim >= 3` (an event when its `ndim >= 2`). `(T, 1)` stays one asset, so every `(T, K)` call keeps its result, as the spec's compatibility section requires.
2. **Per-asset state for iterators** comes from fresh operator copies. Python builds them with a new `_fresh()` method on every wrapped functor, which reconstructs the operator from its captured constructor parameters. JavaScript builds them from a factory that the generated constructors now pass to `wrapOp`. The first event fixes the asset count. The original operator is reset when an asset-shaped stream starts, so all K series start clean.
3. **`CombineLatest` inside a `Pipeline`** learns each input's width from that input's first event. With `emit="when_all"` (the default) this is always exact. With `emit="on_any"`, a row can be emitted before every input has arrived. If an `Input` then turns out to be wider than one column, the node raises a clear error that recommends `when_all`. Phase 2 removes this limit by declaring `Input(layout=...)`.
4. **A layout's result must start with `T`**, since time is always axis 0. With several `A` letters, they are matched in order. Surplus `A` letters on the right are inserted and surplus `A` letters on the left are dropped.
5. **`Resample` with a functor `agg`** keeps taking one column. A multi-column input raises a `ValueError` that says so. The builtin string aggregations get the per-column rule.
6. **JavaScript `combineLatest` is graph-only today.** It gains field concatenation through the shared C++ node. Its `layout` option is about assets, which only phase 2 brings into a `Pipeline`, so it is not added in phase 1. This goes in the final report as a deliberate phase-2 item.
7. **`from_pandas` on a DataFrame with two-level `(asset, field)` columns** now returns a `(T, A, F)` array, as the spec's container table states. Flat columns are unchanged.

## Review Focus

These are the failure modes most likely to hurt a user that no task's main tests target directly. Each one has a pinning test in the task named.

1. **Non-float64 or non-contiguous inputs to a broadcast call** (an `int64` price, a transposed view) must give the same numbers as a contiguous float64 call. Pinned in Task 3 (`test_broadcast_reads_any_dtype_and_strides`).
2. **Empty data**: `T = 0` with a broadcast, or zero assets `(T, 0)`, must return an empty array of the right shape without crashing. Pinned in Task 3 (`test_empty_broadcast_shapes`).
3. **`(T, 1)` input to a one-input operator** must stay `(T, 1)`. Rule 3 must not fire on a 2-D array. Pinned in Task 3 (`test_t_by_1_is_one_asset`).
4. **An iterator whose event shape changes mid-stream** must raise a clear error naming both shapes, never return garbage or crash. Pinned in Task 4 (`test_event_shape_change_raises`) and Task 5 (JS).
5. **`CombineLatest(emit="on_any")` in a `Pipeline` with a wide `Input`** must raise the clear error from decision 3, never misalign columns. Pinned in Task 6 (`test_pipeline_on_any_wide_input_raises_clearly`).

---

## File Structure

| File | Responsibility |
|---|---|
| `include/screamer/common/shape.h` (new) | All shape rules: `plan_layout`, `consume_unit_field`, `input_views`, `broadcast`. |
| `bindings/bindings_core.cpp` | Expose `_layout_plan`, `_broadcast_shape`, `_input_shape` to Python. |
| `include/screamer/common/dispatch.h`, `src/screamer/common/dispatch.cpp` | Use `shape.h` for N-input broadcast, packed N-D split, rule 3. |
| `include/screamer/common/lazy_eval_iterator.h` | Events of any fixed shape, one operator copy per asset. |
| `screamer/_functor_params.py` | `_fresh()` on wrapped functors. |
| `screamer/shape.py` (new) | `reshape` for every Python container; DataFrame axis helpers. |
| `devtools/generate_screamer__init__.py` | Export `screamer.shape.__all__`. |
| `screamer/streams.py` | `from_pandas(layout=)`, `combine_latest(layout=)`, multi-column `resample`. |
| `include/screamer/streams/combine_latest.h` | Per-source widths, concatenated row. |
| `bindings/bindings_streams.cpp` | Eager combine of 2-D value arrays. |
| `include/screamer/dag/combine_latest_node.h`, `include/screamer/dag/compiled_graph.h` | Learn widths, concatenate, clear errors. |
| `include/screamer/dag/resample_params.h`, `include/screamer/dag/resample_node.h` | Per-column plan for wide frames. |
| `wasm/shape_bindings.cpp` (new), `wasm/CMakeLists.txt` | `evalShaped`, `shapedResult`, `layoutPlan` for JS. |
| `js/src/runtime.ts` | N-D/broadcast calls, asset events in iterators, factory-based copies. |
| `devtools/wasm/gen_ts_api.py`, `js/src/generated/ops.ts` | Factories pass a constructor closure. |
| `js/src/shape.ts` (new), `js/src/index.ts` | `reshape` in JS. |
| `screamer/validation.py`, `js/src/validation.ts` | Drop manual broadcasting. |
| `tests/test_shape_contract.py` (new), `devtools/wasm/gen_shape_oracle.py` (new), `wasm/smoke/shape_oracle.json` (generated), `js/test/shape_contract.test.ts` (new) | Contract tests. |
| `docs/polymorphic_api.md`, `docs/multistream.md`, `docs/functions_streams/reshape.md` (new), recipes, `CHANGELOG.md` | Docs. |

---

### Task 1: The shape module

**Files:**
- Create: `include/screamer/common/shape.h`
- Modify: `bindings/bindings_core.cpp` (add three test/adapter bindings inside `init_bindings_core`)
- Test: `tests/test_shape_module.py`

**Interfaces:**
- Produces (C++, namespace `screamer::shape`):
  - `using Shape = std::vector<std::size_t>;`
  - `std::string to_string(const Shape&)` renders `(100, 3)`, or `(100,)` for one axis.
  - `struct LayoutPlan { std::string out; std::vector<int> source; std::vector<std::size_t> dropped; Shape shape; };`
  - `LayoutPlan plan_layout(const std::string& layout, const Shape& in)` throws `std::invalid_argument`.
  - `struct View { Shape shape; std::vector<std::size_t> strides; std::size_t offset = 0; };`
  - `View contiguous_view(const Shape&)`
  - `Shape consume_unit_field(Shape)`
  - `std::vector<View> split_fields(const Shape& packed, std::size_t n)`
  - `std::vector<View> input_views(const std::vector<Shape>& args, std::size_t n_in)`
  - `struct Plan { Shape shape; std::size_t T = 0, columns = 1; std::vector<std::size_t> time_stride; std::vector<std::size_t> offset; };`. Here `offset[i * columns + c]` is the element of input `i` at `t = 0` in column `c`.
  - `Plan broadcast(const std::vector<View>&)` throws `std::invalid_argument`.
- Produces (Python, `screamer.screamer_bindings`):
  - `_layout_plan(layout: str, shape: list[int]) -> (out: str, source: list[int], dropped: list[int], shape: list[int])`
  - `_broadcast_shape(shapes: list[list[int]]) -> list[int]` (plain broadcast, no rule 3)
  - `_input_shape(shapes: list[list[int]], n_in: int) -> list[int]` (an operator call's result shape `(T, *A)`, with the packed split and rule 3)

- [ ] **Step 1: Write the failing tests**

`tests/test_shape_module.py`:

```python
"""The C++ shape module: layout strings and the asset broadcast."""
import pytest

from screamer import screamer_bindings as _b


@pytest.mark.parametrize("layout, shape, out, result", [
    ("TA", [100, 3], "TAF", [100, 3, 1]),
    ("TAF->TA", [100, 3, 1], "TA", [100, 3]),
    ("TF->TAF", [100, 4], "TAF", [100, 1, 4]),
    ("AT->TA", [3, 100], "TA", [100, 3]),
    ("TFA", [100, 4, 3], "TAF", [100, 3, 4]),
    ("T", [100], "TF", [100, 1]),
    ("TAAF", [100, 2, 3, 4], "TAAF", [100, 2, 3, 4]),
    (" T A ", [100, 3], "TAF", [100, 3, 1]),
])
def test_layout_result(layout, shape, out, result):
    got_out, _, _, got_shape = _b._layout_plan(layout, shape)
    assert got_out == out
    assert list(got_shape) == result


def test_layout_source_and_dropped():
    out, source, dropped, shape = _b._layout_plan("TFA->TA", [100, 1, 3])
    assert out == "TA" and list(source) == [0, 2] and list(dropped) == [1]


@pytest.mark.parametrize("layout, shape, message", [
    ("TA", [100, 3, 1], "names 2 axes"),
    ("TX", [100, 3], "unknown axis letter"),
    ("AA", [3, 3], "exactly one T"),
    ("TFF", [100, 2, 2], "at most one F"),
    ("TAF->TA", [100, 3, 4], "only a size-1 axis can be dropped"),
    ("TA->AT", [100, 3], "must start with T"),
    ("TA->TA->TA", [100, 3], "at most one '->'"),
])
def test_layout_errors(layout, shape, message):
    with pytest.raises(ValueError, match=message):
        _b._layout_plan(layout, shape)


def test_layout_error_names_layout_and_shape():
    with pytest.raises(ValueError, match=r"layout 'TAF->TA'.*\(100, 3, 4\)"):
        _b._layout_plan("TAF->TA", [100, 3, 4])


@pytest.mark.parametrize("shapes, result", [
    ([[10], [10, 3]], [10, 3]),
    ([[10, 1], [10, 3]], [10, 3]),
    ([[10, 3], [10, 2, 3]], [10, 2, 3]),
    ([[10, 2, 1], [10, 1, 3]], [10, 2, 3]),
    ([[10], [10]], [10]),
    ([[0, 3], [0]], [0, 3]),
    ([[10, 0], [10]], [10, 0]),
])
def test_broadcast_shape(shapes, result):
    assert list(_b._broadcast_shape(shapes)) == result


def test_broadcast_rejects_mismatched_assets():
    with pytest.raises(ValueError, match=r"\(10, 3\).*\(10, 4\).*do not broadcast"):
        _b._broadcast_shape([[10, 3], [10, 4]])


def test_broadcast_rejects_mismatched_time():
    with pytest.raises(ValueError, match="same length along time"):
        _b._broadcast_shape([[10, 3], [9, 3]])


@pytest.mark.parametrize("shapes, n_in, result", [
    ([[10, 3, 1]], 1, [10, 3]),          # rule 3
    ([[10, 1]], 1, [10, 1]),             # (T, 1) is one asset
    ([[10, 2]], 2, [10]),                # packed (T, N)
    ([[10, 3, 2]], 2, [10, 3]),          # packed (T, K, N)
    ([[10, 3, 1], [10]], 2, [10, 3]),    # rule 3 on a separate argument
])
def test_input_shape(shapes, n_in, result):
    assert list(_b._input_shape(shapes, n_in)) == result


def test_input_shape_rejects_wrong_packed_width():
    with pytest.raises(ValueError, match="expects 2 inputs"):
        _b._input_shape([[10, 3]], 2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_shape_module.py -q`
Expected: FAIL with `AttributeError: module 'screamer.screamer_bindings' has no attribute '_layout_plan'`.

- [ ] **Step 3: Write `include/screamer/common/shape.h`**

```cpp
#ifndef SCREAMER_COMMON_SHAPE_H
#define SCREAMER_COMMON_SHAPE_H

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace screamer { namespace shape {

// The shape contract (docs/polymorphic_api.md): every array is (T, *A, F?).
// T is time (axis 0); A are zero or more asset axes, each asset an independent
// series with its own operator state; F is an optional last axis of fields.
// This header holds the rules that read shapes, once, for the Python and the
// JavaScript bindings.

using Shape = std::vector<std::size_t>;

inline std::string to_string(const Shape& s) {
    std::ostringstream o;
    o << '(';
    for (std::size_t i = 0; i < s.size(); ++i) o << (i ? ", " : "") << s[i];
    if (s.size() == 1) o << ',';
    o << ')';
    return o.str();
}

// ---- layout strings ---------------------------------------------------------
//
// A layout names the axes of an array with T (time), A (asset) and F (field),
// optionally followed by "->" and the wanted result: "TAF->TA" drops a size-1
// field axis, "TF->TAF" inserts a size-1 asset axis, "AT->TA" reorders. Without
// "->" the result is the canonical (T, *A, F), with F inserted when missing.

struct LayoutPlan {
    std::string out;                   // result axis letters, e.g. "TAF"
    std::vector<int> source;           // per result axis: input axis, or -1 (inserted, size 1)
    std::vector<std::size_t> dropped;  // input axes removed; each has size 1
    Shape shape;                       // result shape
};

namespace detail {

inline std::invalid_argument layout_error(const std::string& layout, const std::string& what) {
    return std::invalid_argument("layout '" + layout + "': " + what);
}

inline void check_side(const std::string& side, const std::string& layout, const char* name) {
    std::size_t t = 0, f = 0;
    for (char c : side) {
        if (c == 'T') ++t;
        else if (c == 'F') ++f;
        else if (c != 'A')
            throw layout_error(layout, std::string("unknown axis letter '") + c +
                                       "'; use T (time), A (asset) and F (field)");
    }
    if (t != 1) throw layout_error(layout, std::string("the ") + name + " must name exactly one T axis");
    if (f > 1) throw layout_error(layout, std::string("the ") + name + " can name at most one F axis");
}

}  // namespace detail

inline LayoutPlan plan_layout(const std::string& raw, const Shape& in) {
    std::string layout;
    for (char c : raw) if (c != ' ') layout += c;
    std::string left = layout, right;
    const std::size_t arrow = layout.find("->");
    if (arrow != std::string::npos) {
        left = layout.substr(0, arrow);
        right = layout.substr(arrow + 2);
        if (right.find("->") != std::string::npos)
            throw detail::layout_error(raw, "use at most one '->'");
    }
    detail::check_side(left, raw, "left side");
    if (left.size() != in.size())
        throw detail::layout_error(raw, "names " + std::to_string(left.size()) +
            " axes, but the input has shape " + to_string(in));
    if (arrow == std::string::npos) {
        right = "T";
        for (char c : left) if (c == 'A') right += 'A';
        right += 'F';
    }
    detail::check_side(right, raw, "result");
    if (right[0] != 'T')
        throw detail::layout_error(raw, "the result must start with T, because time is axis 0");

    int t_in = -1, f_in = -1;
    std::vector<int> a_in;
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (left[i] == 'T') t_in = static_cast<int>(i);
        else if (left[i] == 'F') f_in = static_cast<int>(i);
        else a_in.push_back(static_cast<int>(i));
    }
    LayoutPlan p;
    p.out = right;
    std::vector<bool> used(left.size(), false);
    std::size_t a_next = 0;
    for (char c : right) {
        int src;
        if (c == 'T') src = t_in;
        else if (c == 'F') src = f_in;
        else src = a_next < a_in.size() ? a_in[a_next++] : -1;
        p.source.push_back(src);
        if (src >= 0) used[static_cast<std::size_t>(src)] = true;
        p.shape.push_back(src >= 0 ? in[static_cast<std::size_t>(src)] : 1);
    }
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (used[i]) continue;
        if (in[i] != 1)
            throw detail::layout_error(raw, std::string("drops axis ") + left[i] + " (axis " +
                std::to_string(i) + "), which has size " + std::to_string(in[i]) + " in shape " +
                to_string(in) + "; only a size-1 axis can be dropped");
        p.dropped.push_back(i);
    }
    return p;
}

// ---- operator inputs ----------------------------------------------------------

// One input field of shape (T, *A) in a C-contiguous buffer: either a whole
// argument, or one field of a packed (T, *A, N) argument.
struct View {
    Shape shape;                        // (T, *A)
    std::vector<std::size_t> strides;   // element strides, one per axis of shape
    std::size_t offset = 0;             // element offset of the first value
};

inline View contiguous_view(const Shape& shape) {
    View v;
    v.shape = shape;
    v.strides.assign(shape.size(), 1);
    for (std::size_t d = shape.size(); d-- > 1; ) v.strides[d - 1] = v.strides[d] * shape[d];
    return v;
}

// Rule 3: a trailing size-1 field axis is consumed, so (T, K, 1) reads as
// (T, K). Only an array with an asset axis has a field axis to consume:
// (T, 1) stays one asset.
inline Shape consume_unit_field(Shape s) {
    if (s.size() >= 3 && s.back() == 1) s.pop_back();
    return s;
}

// Rule 2: one packed (T, *A, n) array holds the n input fields along its last axis.
inline std::vector<View> split_fields(const Shape& packed, std::size_t n) {
    const View whole = contiguous_view(packed);
    std::vector<View> out(n);
    for (std::size_t j = 0; j < n; ++j) {
        out[j].shape.assign(packed.begin(), packed.end() - 1);
        out[j].strides.assign(whole.strides.begin(), whole.strides.end() - 1);
        out[j].offset = j;
    }
    return out;
}

// The input fields of an n_in-input call, given the argument shapes: one packed
// argument is split along its last axis; separate arguments have a trailing
// size-1 field axis consumed.
inline std::vector<View> input_views(const std::vector<Shape>& args, std::size_t n_in) {
    const std::string n = std::to_string(n_in);
    if (args.size() == 1 && n_in > 1) {
        const Shape& s = args[0];
        if (s.size() < 2 || s.back() != n_in)
            throw std::invalid_argument("This functor expects " + n + " inputs; got one array of shape " +
                to_string(s) + ". Pass a (T, ..., " + n + ") array or " + n + " separate arrays.");
        return split_fields(s, n_in);
    }
    if (args.size() != n_in)
        throw std::invalid_argument("This functor expects " + n + " inputs; got " +
                                    std::to_string(args.size()) + ".");
    std::vector<View> out;
    out.reserve(args.size());
    for (const Shape& s : args) out.push_back(contiguous_view(consume_unit_field(s)));
    return out;
}

// Rule 2: the inputs' asset shapes broadcast with NumPy rules (right-aligned,
// each axis equal or 1) and T must match. The plan lists, per input and per
// result column, where that column starts, so a broadcast input is read in
// place, never copied once per asset.
struct Plan {
    Shape shape;                              // (T, *A): the result without output fields
    std::size_t T = 0, columns = 1;           // columns = product of A
    std::vector<std::size_t> time_stride;     // per input
    std::vector<std::size_t> offset;          // [input * columns + column]
};

inline Plan broadcast(const std::vector<View>& v) {
    if (v.empty()) throw std::invalid_argument("broadcast: no inputs");
    std::string all;
    for (std::size_t i = 0; i < v.size(); ++i) all += (i ? ", " : "") + to_string(v[i].shape);
    Plan p;
    std::size_t na = 0;
    for (const View& x : v) {
        if (x.shape.empty()) throw std::invalid_argument("every input needs a time axis; got shapes " + all);
        if (x.shape[0] != v[0].shape[0])
            throw std::invalid_argument("inputs must have the same length along time (axis 0); got shapes " + all);
        na = std::max(na, x.shape.size() - 1);
    }
    p.T = v[0].shape[0];
    Shape assets(na, 1);
    for (const View& x : v) {
        const std::size_t k = x.shape.size() - 1;
        for (std::size_t j = 0; j < k; ++j) {
            const std::size_t d = x.shape[1 + j];
            std::size_t& a = assets[na - k + j];
            if (d == a || d == 1) continue;
            if (a == 1) { a = d; continue; }
            throw std::invalid_argument("input shapes " + all +
                " do not broadcast: each asset axis must be equal or 1 (time is axis 0)");
        }
    }
    p.shape.push_back(p.T);
    p.shape.insert(p.shape.end(), assets.begin(), assets.end());
    for (std::size_t a : assets) p.columns *= a;
    p.time_stride.resize(v.size());
    p.offset.assign(v.size() * p.columns, 0);
    for (std::size_t i = 0; i < v.size(); ++i) {
        const View& x = v[i];
        const std::size_t k = x.shape.size() - 1;
        p.time_stride[i] = x.strides[0];
        std::vector<std::size_t> astride(na, 0);   // stays 0 on broadcast axes
        for (std::size_t j = 0; j < k; ++j)
            if (x.shape[1 + j] != 1) astride[na - k + j] = x.strides[1 + j];
        std::vector<std::size_t> idx(na, 0);
        for (std::size_t c = 0; c < p.columns; ++c) {
            std::size_t off = x.offset;
            for (std::size_t d = 0; d < na; ++d) off += idx[d] * astride[d];
            p.offset[i * p.columns + c] = off;
            for (std::size_t d = na; d-- > 0; ) {      // odometer, last axis fastest
                if (++idx[d] < assets[d]) break;
                idx[d] = 0;
            }
        }
    }
    return p;
}

}}  // namespace screamer::shape
#endif
```

- [ ] **Step 4: Add the Python bindings**

In `bindings/bindings_core.cpp`, add `#include "screamer/common/shape.h"` and `#include <nanobind/stl/tuple.h>`, then at the end of `init_bindings_core`:

```cpp
    // Shape-contract helpers (include/screamer/common/shape.h), used by
    // screamer.shape.reshape, combine_latest and the tests.
    m.def("_layout_plan", [](const std::string& layout, std::vector<std::size_t> shape) {
        auto p = screamer::shape::plan_layout(layout, shape);
        return nb::make_tuple(p.out, p.source, p.dropped, p.shape);
    }, "layout"_a, "shape"_a);
    m.def("_broadcast_shape", [](const std::vector<std::vector<std::size_t>>& shapes) {
        std::vector<screamer::shape::View> views;
        for (const auto& s : shapes) views.push_back(screamer::shape::contiguous_view(s));
        return screamer::shape::broadcast(views).shape;
    }, "shapes"_a);
    m.def("_input_shape", [](const std::vector<std::vector<std::size_t>>& shapes, std::size_t n_in) {
        return screamer::shape::broadcast(screamer::shape::input_views(shapes, n_in)).shape;
    }, "shapes"_a, "n_in"_a);
```

If `"layout"_a` is not in scope in that file, add `using namespace nb::literals;` next to the existing `namespace nb = nanobind;`. nanobind turns `std::invalid_argument` into `ValueError`.

- [ ] **Step 5: Build and run the tests**

Run: `make install-dev && pytest tests/test_shape_module.py -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add include/screamer/common/shape.h bindings/bindings_core.cpp tests/test_shape_module.py
git commit -m "feat: shape module for layouts and asset broadcasting"
```

---

### Task 2: `reshape` and `from_pandas(layout=)` in Python

**Files:**
- Create: `screamer/shape.py`
- Modify: `devtools/generate_screamer__init__.py` (export `screamer/shape.py`'s `__all__`, the same way `validation_names` is read and emitted)
- Modify: `screamer/streams.py:115-123` (`from_pandas`)
- Modify: `docs/functions_streams/from_pandas.md` (document `layout`)
- Test: `tests/test_reshape.py`

**Interfaces:**
- Consumes: `_b._layout_plan` (Task 1).
- Produces:
  - `screamer.reshape(x, layout)` returns the same container kind.
  - `screamer.shape.frame_axes(obj) -> (values: np.ndarray, labels: list)` gives the values of a Series or DataFrame and one label array per axis. Two-level columns become two axes.
  - `screamer.shape.reshape_array(x: np.ndarray, layout: str) -> np.ndarray`.
  - `from_pandas(obj, layout=None) -> (values, index)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_reshape.py`:

```python
"""reshape names the axes of data with a layout and returns the same container."""
import numpy as np
import pandas as pd
import pytest

import screamer as s


def test_array_cases():
    x = np.arange(12.0).reshape(4, 3)
    assert s.reshape(x, "TA").shape == (4, 3, 1)
    assert s.reshape(x[:, :, None], "TAF->TA").shape == (4, 3)
    assert s.reshape(np.zeros((4, 4)), "TF->TAF").shape == (4, 1, 4)
    np.testing.assert_array_equal(s.reshape(x.T, "AT->TA"), x)
    y = np.arange(24.0).reshape(4, 2, 3)          # (T, F, A)
    np.testing.assert_array_equal(s.reshape(y, "TFA"), y.transpose(0, 2, 1))


def test_array_result_is_a_view_without_reordering():
    x = np.arange(12.0).reshape(4, 3)
    assert np.shares_memory(s.reshape(x, "TA"), x)


def test_values_index_tuple():
    x, idx = np.zeros((4, 3)), np.arange(4)
    v, i = s.reshape((x, idx), "TA")
    assert v.shape == (4, 3, 1) and i is idx


def test_list_and_iterator_of_events():
    rows = [np.arange(3.0) + t for t in range(4)]
    out = s.reshape(rows, "TA")
    assert isinstance(out, list) and out[0].shape == (3, 1)
    gen = s.reshape(iter(rows), "TA")
    assert not isinstance(gen, list) and next(gen).shape == (3, 1)


def test_events_need_time_first():
    with pytest.raises(ValueError, match="must start with T"):
        s.reshape([np.zeros(3)], "AT->TA")


def test_dataframe_flat_to_multiindex():
    df = pd.DataFrame({"BTC": [1.0, 2.0], "ETH": [3.0, 4.0]})
    out = s.reshape(df, "TA")
    assert list(out.columns) == [("BTC", "value"), ("ETH", "value")]
    np.testing.assert_array_equal(out.to_numpy(), df.to_numpy())


def test_dataframe_multiindex_reorder():
    cols = pd.MultiIndex.from_product([["open", "close"], ["BTC", "ETH"]])   # (field, asset)
    df = pd.DataFrame(np.arange(8.0).reshape(2, 4), columns=cols)
    out = s.reshape(df, "TFA")
    assert list(out.columns) == [("BTC", "open"), ("BTC", "close"), ("ETH", "open"), ("ETH", "close")]
    assert out[("ETH", "close")].tolist() == df[("close", "ETH")].tolist()


def test_dataframe_multiindex_must_be_complete():
    cols = pd.MultiIndex.from_tuples([("BTC", "close"), ("ETH", "open")])
    with pytest.raises(ValueError, match="every"):
        s.reshape(pd.DataFrame(np.zeros((2, 2)), columns=cols), "TAF")


def test_series():
    out = s.reshape(pd.Series([1.0, 2.0]), "T")
    assert isinstance(out, pd.DataFrame) and list(out.columns) == ["value"]


def test_from_pandas_layout():
    df = pd.DataFrame({"BTC": [1.0, 2.0], "ETH": [3.0, 4.0]}, index=[10, 20])
    values, index = s.from_pandas(df, layout="TA")
    assert values.shape == (2, 2, 1) and list(index) == [10, 20]
    flat, _ = s.from_pandas(df)
    assert flat.shape == (2, 2)


def test_from_pandas_reads_two_level_columns_as_assets_and_fields():
    cols = pd.MultiIndex.from_product([["BTC", "ETH"], ["open", "close"]])
    values, _ = s.from_pandas(pd.DataFrame(np.arange(8.0).reshape(2, 4), columns=cols))
    assert values.shape == (2, 2, 2)
    assert values[1, 1, 0] == 6.0    # ETH open at t=1


def test_unsupported_container():
    with pytest.raises(TypeError):
        s.reshape("abc", "T")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_reshape.py -q`
Expected: FAIL with `AttributeError: module 'screamer' has no attribute 'reshape'`.

- [ ] **Step 3: Write `screamer/shape.py`**

```python
"""Name the axes of data with a layout string and reshape it to match.

Every array in screamer reads as ``(T, *A, F?)``: time on axis 0, zero or more
asset axes of independent series, and an optional last axis of fields. A layout
names an array's axes with the letters ``T``, ``A`` and ``F``; ``->`` gives the
wanted result. The rules live in C++ (``include/screamer/common/shape.h``);
this module applies them to each container.
"""
import numpy as np

from . import screamer_bindings as _b
from .dag import _is_vi_pair

__all__ = ["reshape"]


def reshape_array(x, layout):
    """Reshape one NumPy array by a layout; a view unless axes are reordered."""
    x = np.asarray(x)
    _, source, dropped, _ = _b._layout_plan(layout, list(x.shape))
    if dropped:
        x = np.squeeze(x, axis=tuple(dropped))
    kept = [a for a in range(x.ndim + len(dropped)) if a not in dropped]
    position = {axis: n for n, axis in enumerate(kept)}
    x = np.transpose(x, [position[a] for a in source if a >= 0])
    for axis, a in enumerate(source):
        if a < 0:
            x = np.expand_dims(x, axis)
    return x


def _event(e, layout):
    if not layout.replace(" ", "").startswith("T"):
        raise ValueError(
            f"layout {layout!r}: for a sequence of events the layout must start with T, "
            "the axis of the sequence")
    return reshape_array(np.asarray(e, dtype=np.float64)[None], layout)[0]


def frame_axes(obj):
    """Values and per-axis labels of a pandas Series or DataFrame.

    A DataFrame with two-level columns reads as three axes, (index, level 0,
    level 1), and must hold every pair of the two levels.
    """
    import pandas as pd
    if isinstance(obj, pd.Series):
        return obj.to_numpy(dtype=np.float64), [obj.index]
    cols = obj.columns
    if isinstance(cols, pd.MultiIndex):
        if cols.nlevels != 2:
            raise ValueError("a DataFrame for screamer has flat or two-level columns")
        levels = [cols.unique(level=0), cols.unique(level=1)]
        full = pd.MultiIndex.from_product(levels)
        if len(cols) != len(full) or set(cols) != set(full):
            raise ValueError(
                "a DataFrame with two-level columns must hold every pair of the two "
                "levels, e.g. every (asset, field)")
        values = obj[full].to_numpy(dtype=np.float64).reshape(
            len(obj), len(levels[0]), len(levels[1]))
        return values, [obj.index, levels[0], levels[1]]
    return obj.to_numpy(dtype=np.float64), [obj.index, cols]


def _frame(obj, layout):
    import pandas as pd
    values, labels = frame_axes(obj)
    out, source, _, _ = _b._layout_plan(layout, list(values.shape))
    arr = reshape_array(values, layout)
    names = [labels[a] if a >= 0 else (["value"] if letter == "F" else [0])
             for letter, a in zip(out, source)]
    if arr.ndim == 1:
        return pd.Series(arr, index=names[0])
    if arr.ndim == 2:
        return pd.DataFrame(arr, index=names[0], columns=names[1])
    if arr.ndim == 3:
        return pd.DataFrame(arr.reshape(len(arr), -1), index=names[0],
                            columns=pd.MultiIndex.from_product([names[1], names[2]]))
    raise ValueError(f"layout {layout!r} gives {arr.ndim} axes; a DataFrame holds at most 3")


def reshape(x, layout):
    """Reshape data by a layout string and return the same kind of container.

    ``layout`` names the axes of ``x`` in order with ``T`` (time), ``A`` (asset)
    and ``F`` (field). Without ``->`` the result is the canonical
    ``(T, *A, F)``; with ``->`` it is the layout on the right. A letter missing
    on the left is inserted with size 1; a letter missing on the right is
    dropped and must have size 1.

    ``x`` may be a NumPy array, a ``(values, index)`` tuple, a pandas Series or
    DataFrame, a list of events, or an iterator of events (``T`` then refers to
    the sequence).

    >>> reshape(np.zeros((100, 3)), "TA").shape
    (100, 3, 1)
    >>> reshape(np.zeros((100, 3, 1)), "TAF->TA").shape
    (100, 3)
    """
    import pandas as pd
    if isinstance(x, np.ndarray):
        return reshape_array(x, layout)
    if _is_vi_pair(x):
        return reshape_array(x[0], layout), x[1]
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return _frame(x, layout)
    if isinstance(x, list):
        return [_event(e, layout) for e in x]
    if hasattr(x, "__next__"):
        return (_event(e, layout) for e in x)
    raise TypeError(
        "reshape takes a NumPy array, a (values, index) tuple, a pandas Series or "
        f"DataFrame, a list of events or an iterator of events; got {type(x).__name__}")
```

- [ ] **Step 4: Export it**

In `devtools/generate_screamer__init__.py`, add `shape_names = read_public_names('screamer/shape.py')` next to `validation_names`, with the same empty-check. Emit `from .shape import (\n    {", ".join(shape_names)}\n)` after the validation import, and add `shape_names` to the `__all__` concatenation. Then run `make regen-init`.

- [ ] **Step 5: `from_pandas(layout=)`**

Replace `from_pandas` in `screamer/streams.py`:

```python
def from_pandas(obj, layout=None):
    """Build a (values, index) tuple from a pandas Series or DataFrame.

    data -> values (numpy array), pandas index -> index array. A DataFrame with
    two-level ``(asset, field)`` columns gives ``(T, A, F)`` values. ``layout``
    names the axes as read (``"TA"`` for one column per asset, for example) and
    reshapes the values like :func:`reshape`. pandas always materializes an
    index (a RangeIndex for a default one), so a positional stream does not
    round-trip to index=None through to_pandas/from_pandas.
    """
    from .shape import frame_axes, reshape_array
    values, _ = frame_axes(obj)
    if layout is not None:
        values = reshape_array(values, layout)
    return values, np.asarray(obj.index)
```

Add a `layout` row to the parameters in `docs/functions_streams/from_pandas.md`, matching how that page lists its other parameters.

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_reshape.py tests/test_streams_polymorphism.py -q`
Expected: all PASS. If an existing test relied on `from_pandas` flattening two-level columns, update it to expect `(T, A, F)` (decision 7).

- [ ] **Step 7: Commit**

```bash
git add screamer/shape.py screamer/streams.py screamer/__init__.py devtools/generate_screamer__init__.py docs/functions_streams/from_pandas.md tests/test_reshape.py
git commit -m "feat: reshape by layout for every container; from_pandas(layout=)"
```

---

### Task 3: Broadcasting, packed N-D inputs, and rule 3 in the Python dispatch

**Files:**
- Modify: `include/screamer/common/dispatch.h` (replace `read_n_arrays` and `maybe_split_TxN`; rewrite `handle_input_Ni_1o_numpy` and `handle_input_Ni_Mo_numpy`; apply rule 3 in `handle_input_1i_1o_numpy` and `handle_input_1i_Mo_numpy`; route lazy iterables in `functor_1i_1o`)
- Modify: `src/screamer/common/dispatch.cpp:80-125` (`process_python_array`: rule 3)
- Test: `tests/test_broadcasting.py`

**Interfaces:**
- Consumes: `shape::input_views`, `shape::broadcast`, `shape::consume_unit_field`, `shape::Plan` (Task 1).
- Produces: `template <size_t N> struct NInputs { std::vector<ContigDouble> buffers; std::array<double*, N> data{}; shape::Plan plan; };`, `template <size_t N> NInputs<N> read_inputs(const std::vector<nb::handle>&)`, and `template <size_t N> std::optional<NInputs<N>> maybe_read_packed(const nb::args&)` (all in `screamer::detail`).

- [ ] **Step 1: Write the failing tests**

`tests/test_broadcasting.py`:

```python
"""Multi-input operators broadcast over assets; a size-1 field axis is consumed."""
import numpy as np
import pytest

import screamer as s

rng = np.random.default_rng(0)
T, K = 200, 3
X = rng.normal(size=(T, K)).cumsum(axis=0) + 100
y = rng.normal(size=T).cumsum() + 100


def test_one_d_broadcasts_against_assets():
    got = s.RollingCorr(20)(X, y)
    want = s.RollingCorr(20)(X, np.repeat(y[:, None], K, axis=1))
    np.testing.assert_array_equal(got, want)
    assert got.shape == (T, K)


def test_column_broadcasts_against_assets():
    np.testing.assert_array_equal(s.Add()(X, y[:, None]), X + y[:, None])


def test_multi_output_engine_with_one_price():
    target = np.sign(rng.normal(size=(T, K)))
    got = s.BacktestPriceTarget()(target, y)
    want = s.BacktestPriceTarget()(target, np.repeat(y[:, None], K, axis=1))
    assert got.shape == (T, K, 4)
    np.testing.assert_array_equal(got, want)


def test_mismatched_assets_raise():
    with pytest.raises(ValueError, match="do not broadcast"):
        s.Add()(np.zeros((T, 3)), np.zeros((T, 4)))


def test_mismatched_time_raises():
    with pytest.raises(ValueError, match="same length along time"):
        s.Add()(np.zeros(T), np.zeros(T - 1))


def test_packed_three_d():
    packed = np.stack([X, X + 1.0], axis=-1)                    # (T, K, 2)
    np.testing.assert_array_equal(s.Sub()(packed), np.full((T, K), -1.0))


def test_unit_field_consumed_one_input():
    np.testing.assert_array_equal(s.RollingMean(5)(X[:, :, None]), s.RollingMean(5)(X))


def test_unit_field_consumed_multi_output():
    assert s.BollingerBands(5)(X[:, :, None]).shape == (T, K, 3)


def test_unit_field_consumed_functorbase_one_input():
    np.testing.assert_array_equal(s.IsNan()(X[:, :, None]), s.IsNan()(X))


def test_unit_field_consumed_multi_input():
    np.testing.assert_array_equal(s.Add()(X[:, :, None], y), s.Add()(X, y))


def test_t_by_1_is_one_asset():
    assert s.RollingMean(5)(X[:, :1]).shape == (T, 1)
    assert s.BollingerBands(5)(X[:, :1]).shape == (T, 1, 3)


def test_broadcast_reads_any_dtype_and_strides():
    Xi = np.round(X).astype(np.int64)
    view = np.asfortranarray(Xi.astype(np.float64))
    want = s.RollingCorr(20)(Xi.astype(np.float64), y)
    np.testing.assert_array_equal(s.RollingCorr(20)(Xi, y), want)
    np.testing.assert_array_equal(s.RollingCorr(20)(view, y), want)
    np.testing.assert_array_equal(s.RollingCorr(20)(Xi.T.copy().T, y), want)


def test_empty_broadcast_shapes():
    assert s.Add()(np.zeros((0, 3)), np.zeros(0)).shape == (0, 3)
    assert s.Add()(np.zeros((T, 0)), np.zeros(T)).shape == (T, 0)
    assert s.BacktestPriceTarget()(np.zeros((0, 2)), np.zeros(0)).shape == (0, 2, 4)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_broadcasting.py -q`
Expected: FAIL (`TypeError: Incompatible input numpy arrays, dimensions mismatch.`, and wrong shapes for the rule-3 cases).

- [ ] **Step 3: Replace the N-input readers in `dispatch.h`**

Add `#include "screamer/common/shape.h"`. Delete `read_n_arrays` and `maybe_split_TxN` and put this in their place, inside `namespace detail`:

```cpp
// The N input fields of a call: one contiguous double buffer per argument (one
// buffer when a packed (T, *A, N) array holds all fields), each field's base
// pointer, and the broadcast plan that says where every column starts.
template <size_t N>
struct NInputs {
    std::vector<ContigDouble> buffers;
    std::array<double*, N> data{};
    shape::Plan plan;
};

// Read the argument arrays (any dtype and strides) and plan their broadcast
// (shape.h rules 2 and 3). Throws TypeError on a non-array argument and
// ValueError on shapes that do not broadcast.
template <size_t N>
inline NInputs<N> read_inputs(const std::vector<nb::handle>& arrays) {
    NInputs<N> r;
    r.buffers.reserve(arrays.size());
    std::vector<shape::Shape> shapes;
    for (nb::handle h : arrays) {
        if (!is_ndarray(h))
            throw nb::type_error("Incompatible input type, a mix of numpy arrays and other.");
        r.buffers.push_back(read_contig_double(nb::cast<nb::ndarray<nb::ro>>(h)));
        if (r.buffers.back().ndim() < 1)
            throw std::runtime_error("Input array must have at least one dimension");
        shapes.push_back(r.buffers.back().shape);
    }
    r.plan = shape::broadcast(shape::input_views(shapes, N));
    for (size_t i = 0; i < N; ++i)
        r.data[i] = r.buffers[arrays.size() == 1 ? 0 : i].data.data();
    return r;
}

// One packed array (rank 2 or more) as the single argument of an N-input call.
template <size_t N>
inline std::optional<NInputs<N>> maybe_read_packed(const nb::args& args) {
    if (args.size() != 1 || !is_ndarray(args[0])) return std::nullopt;
    if (nb::cast<nb::ndarray<nb::ro>>(args[0]).ndim() < 2) return std::nullopt;
    return read_inputs<N>({ args[0] });
}

template <size_t N>
inline NInputs<N> read_tuple_inputs(const nb::tuple& inputs) {
    std::vector<nb::handle> arrays;
    for (nb::handle h : inputs) arrays.push_back(h);
    return read_inputs<N>(arrays);
}
```

In `functor_Ni_1o` and `functor_Ni_Mo`, replace `if (auto cols = detail::maybe_split_TxN<N>(args))` with `if (auto ins = detail::maybe_read_packed<N>(args))` and pass `*ins`. Replace `auto cols = detail::read_n_arrays<N>(inputs);` with `auto ins = detail::read_tuple_inputs<N>(inputs);` and pass `ins`.

- [ ] **Step 4: Rewrite the two N-input batch loops**

```cpp
template <class Op>
nb::object handle_input_Ni_1o_numpy(Op& self, detail::NInputs<Op::kN>& in) {
    constexpr size_t N = Op::kN;
    const shape::Plan& p = in.plan;
    const size_t T = p.T, cols = p.columns;
    double* out = new double[T * cols ? T * cols : 1];
    std::array<int64_t, N> instride{};
    for (size_t i = 0; i < N; ++i) instride[i] = (int64_t) p.time_stride[i];
    for (size_t col = 0; col < cols; ++col) {
        std::array<size_t, N> inoff{};
        for (size_t i = 0; i < N; ++i) inoff[i] = p.offset[i * cols + col];
        self.reset();
        if (!self.process_columns(out, (std::ptrdiff_t) cols, in.data, instride, inoff, col, T)) {
            std::array<size_t, N> idx = inoff;
            size_t idx_out = col;
            typename Op::InputArray call_array;
            for (size_t t = 0; t < T; ++t) {
                for (size_t i = 0; i < N; ++i) { call_array[i] = in.data[i][idx[i]]; idx[i] += p.time_stride[i]; }
                out[idx_out] = self.call(call_array);
                idx_out += cols;
            }
        }
    }
    self.reset();
    return detail::make_owned_array(out, p.shape);
}

template <class Op>
nb::object handle_input_Ni_Mo_numpy(Op& self, detail::NInputs<Op::kN>& in) {
    constexpr size_t N = Op::kN;
    constexpr size_t M = Op::kM;
    const shape::Plan& p = in.plan;
    const size_t T = p.T, cols = p.columns;
    std::vector<size_t> shape = p.shape;
    shape.push_back(M);
    double* out = new double[T * cols * M ? T * cols * M : 1];
    for (size_t col = 0; col < cols; ++col) {
        self.reset();
        std::array<size_t, N> idx{};
        for (size_t i = 0; i < N; ++i) idx[i] = p.offset[i * cols + col];
        typename Op::InputArray call_array;
        for (size_t t = 0; t < T; ++t) {
            for (size_t j = 0; j < N; ++j) { call_array[j] = in.data[j][idx[j]]; idx[j] += p.time_stride[j]; }
            detail::write_tuple_to_memory(&out[(t * cols + col) * M], self.call(call_array));
        }
    }
    self.reset();
    return detail::make_owned_array(out, shape);
}
```

Before relying on `process_columns`, check every override: `grep -rn "process_columns" include/screamer | grep -v "virtual bool"`. Each override must read input `i` at `inputs[i][input_offsets[i] + t * input_strides[i]]` and decline (`return false`) for strides it does not handle. Today the overrides are one-input block extrema (`detail/block_extremum.h`), which this N-input path does not reach. Record the result in the commit message.

- [ ] **Step 5: Rule 3 in the one-input paths, and lazy routing**

In `handle_input_1i_1o_numpy`, change `std::vector<size_t> shape = in.shape;` to `std::vector<size_t> shape = shape::consume_unit_field(in.shape);`. In `handle_input_1i_Mo_numpy`, change it to `std::vector<size_t> shape = shape::consume_unit_field(in.shape);` (followed by the existing `shape.push_back(M);`). The data layout does not change, because only a trailing size-1 axis goes.

In `src/screamer/common/dispatch.cpp` `process_python_array`, change both `return detail::make_owned_array(out, shape);` lines to `return detail::make_owned_array(out, shape::consume_unit_field(shape));`.

In `functor_1i_1o`, insert before `// Case 3: Iterable`:

```cpp
    // Case 2b: a lazy iterator streams through LazyEvalIterator, as for every
    // other operator (lazy in, lazy out).
    if (is_lazy_iterable(input)) {
        std::vector<nb::object> sources{ input };
        return nb::cast(LazyEvalIterator(nb::find(self), std::move(sources)));
    }
```

- [ ] **Step 6: Build and run the tests, then the whole suite**

Run: `make install-dev && pytest tests/test_broadcasting.py -q && pytest -q -x`
Expected: `test_broadcasting.py` all PASS. In the full run, tests that asserted the old messages (`grep -rn "dimensions mismatch\|shape mismatch" tests`) now see a `ValueError` saying "do not broadcast" or "same length along time". Update those assertions to the new type and message. A test that fed a generator to one of the `FunctorBase<_, 1, 1>` operators (`IsNan`, `Not`, `IsFinite`, `TrendMode`, `CyclePhase`, `CycleAmplitude`, `DominantCycle`, `CycleFrequency`, `InstantaneousTrendline`) and expected a list now gets an iterator. Wrap its call in `list(...)`. Then rerun until green.

- [ ] **Step 7: Commit**

```bash
git add include/screamer/common/dispatch.h src/screamer/common/dispatch.cpp tests/
git commit -m "feat: multi-input operators broadcast over assets; size-1 field axis consumed"
```

---

### Task 4: Iterators of events with assets (Python)

**Files:**
- Modify: `screamer/_functor_params.py:143-160` (`_make_wrapper`: add `_fresh`)
- Modify: `include/screamer/common/lazy_eval_iterator.h`
- Test: `tests/test_event_iterators.py`

**Interfaces:**
- Consumes: `shape::input_views`, `shape::broadcast` (Task 1).
- Produces: `op._fresh()` on every wrapped functor, which returns a new instance with the same parameters in its initial state. It also produces `LazyEvalIterator` events of any fixed shape. Scalar events behave exactly as before.

- [ ] **Step 1: Write the failing tests**

`tests/test_event_iterators.py`:

```python
"""An iterator of events gives the same numbers as the batch it stacks into."""
import numpy as np
import pytest

import screamer as s

rng = np.random.default_rng(1)
T, K = 120, 3
X = rng.normal(size=(T, K)).cumsum(axis=0) + 50
y = rng.normal(size=T).cumsum() + 50


def test_fresh_copies_parameters_and_state():
    op = s.RollingMean(7, start_policy="expanding")
    op(np.arange(10.0))
    copy = op._fresh()
    assert type(copy) is type(op) and repr(copy) == repr(op)
    np.testing.assert_array_equal(copy(np.arange(5.0)), s.RollingMean(7, start_policy="expanding")(np.arange(5.0)))


def test_one_input_rows():
    got = np.array(list(s.RollingMean(10)(iter(X))))
    np.testing.assert_array_equal(got, s.RollingMean(10)(X))


def test_one_input_rows_multi_output():
    got = np.array(list(s.BollingerBands(10)(iter(X))))
    np.testing.assert_array_equal(got, s.BollingerBands(10)(X))


def test_unit_field_event_is_consumed():
    got = np.array(list(s.RollingMean(10)(iter(X[:, :, None]))))
    np.testing.assert_array_equal(got, s.RollingMean(10)(X))


def test_functorbase_one_input_rows():
    got = np.array(list(s.IsNan()(iter(X))))
    np.testing.assert_array_equal(got, s.IsNan()(X))


def test_separate_iterators_broadcast():
    got = np.array(list(s.RollingCorr(15)(iter(X), iter(y))))
    np.testing.assert_array_equal(got, s.RollingCorr(15)(X, y))


def test_packed_event_split():
    packed = np.stack([X, X * 2.0], axis=-1)             # events (K, 2)
    got = np.array(list(s.Sub()(iter(packed))))
    np.testing.assert_array_equal(got, -X)


def test_tuple_of_arrays_is_fields():
    got = np.array(list(s.Add()(iter(zip(X, y)))))
    np.testing.assert_array_equal(got, X + y[:, None])


def test_scalar_events_unchanged():
    got = list(s.RollingMean(3)(iter([1.0, 2.0, 3.0])))
    assert got == list(s.RollingMean(3)(np.array([1.0, 2.0, 3.0])))
    assert isinstance(got[0], float)


def test_event_shape_change_raises():
    it = s.RollingMean(3)(iter([np.zeros(3), np.zeros(4)]))
    next(it)
    with pytest.raises(ValueError, match=r"same shape.*\(3,\).*\(4,\)"):
        next(it)


def test_scalar_then_array_raises():
    it = s.RollingMean(3)(iter([1.0, np.zeros(3)]))
    next(it)
    with pytest.raises(ValueError, match="same shape"):
        next(it)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_event_iterators.py -q`
Expected: FAIL (`AttributeError: ... '_fresh'`, `std::bad_cast`).

- [ ] **Step 3: Add `_fresh` in `_make_wrapper`**

After `__repr__` in `screamer/_functor_params.py`:

```python
    def _fresh(self):
        """A new instance with the same parameters, in its initial state.

        Batch calls give each asset its own state by resetting one operator per
        column; an iterator of asset events runs one copy per asset instead.
        """
        params = dict(getattr(self, "_screamer_params", {}))
        args = params.pop("args", [])
        return cls(*args, **params)
```

Add `cls._fresh = _fresh` next to `cls.__repr__ = __repr__`.

- [ ] **Step 4: Rewrite `LazyEvalIterator`**

Replace the class body in `include/screamer/common/lazy_eval_iterator.h` with:

```cpp
#include <string>
#include "screamer/common/shape.h"

// One lazy iterator for any EvalOp. Holds the functor's Python wrapper (keep-alive)
// and the input source(s). Per __next__ it reads one event from each source.
//
// Two input forms, matching the batch dispatch:
//   - `sources` holds n_in() separate iterators;
//   - `sources` holds one iterator whose items hold all n_in() fields: a tuple or
//     list (one value per field) or an array whose last axis has n_in() fields.
//
// Events are scalars (one float out, or a tuple of n_out() floats), or arrays of
// one fixed shape: each holds one value per asset (shape.h rules 2 and 3), the
// first event fixes the shape, and every asset runs its own copy of the op, as
// the batch path gives every column its own state.
class LazyEvalIterator {
public:
    LazyEvalIterator(nb::object op_owner, std::vector<nb::object> iterables)
        : op_owner_(std::move(op_owner)),
          op_(nb::cast<EvalOp&>(op_owner_)),
          n_in_(op_.n_in()), n_out_(op_.n_out()),
          in_(op_.n_in()), out_(op_.n_out()) {
        for (auto& it : iterables) iters_.push_back(it.attr("__iter__")());
        unpack_ = (iters_.size() == 1 && n_in_ > 1);
    }

    LazyEvalIterator& __iter__() { return *this; }

    nb::object __next__() {
        std::vector<nb::object> items;
        for (auto& it : iters_) items.push_back(next_or_stop(it));
        if (mode_ == Mode::Unknown) mode_ = scalar_values(items) ? Mode::Scalar : Mode::Assets;
        if (mode_ == Mode::Scalar) return next_scalar(items);
        return next_assets(items);
    }

private:
    enum class Mode { Unknown, Scalar, Assets };

    static nb::object next_or_stop(nb::object& it) {
        try {
            return it.attr("__next__")();
        } catch (nb::python_error& e) {
            if (e.matches(PyExc_StopIteration)) throw nb::stop_iteration();
            throw;
        }
    }

    // The n_in field values of one event if every one is a plain number.
    bool scalar_values(const std::vector<nb::object>& items) {
        if (unpack_) {
            nb::sequence seq;
            if (!nb::try_cast<nb::sequence>(items[0], seq)) return false;
            if (nb::len(seq) != static_cast<Py_ssize_t>(n_in_)) return false;
            for (std::size_t i = 0; i < n_in_; ++i)
                if (!nb::try_cast<double>(seq[i], in_[i])) return false;
            return true;
        }
        for (std::size_t i = 0; i < n_in_; ++i)
            if (!nb::try_cast<double>(items[i], in_[i])) return false;
        return true;
    }

    nb::object next_scalar(const std::vector<nb::object>& items) {
        if (!scalar_values(items))
            throw nb::value_error("every event must have the same shape; the first events "
                                  "were numbers, this one is not");
        op_.eval(in_.data(), out_.data());
        if (n_out_ == 1) return nb::float_(out_[0]);
        nb::list lst;
        for (std::size_t i = 0; i < n_out_; ++i) lst.append(nb::float_(out_[i]));
        return nb::steal(PySequence_Tuple(lst.ptr()));
    }

    nb::object next_assets(const std::vector<nb::object>& items) {
        nb::module_ np = nb::module_::import_("numpy");
        std::vector<nb::object> parts;   // keep the arrays alive while reading
        if (unpack_ && (nb::isinstance<nb::tuple>(items[0]) || nb::isinstance<nb::list>(items[0]))) {
            for (nb::handle h : items[0]) parts.push_back(nb::borrow<nb::object>(h));   // one per field
        } else {
            parts = items;
        }
        using Arr = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
        std::vector<Arr> arrays;
        std::vector<shape::Shape> shapes;
        for (auto& p : parts) {
            p = np.attr("ascontiguousarray")(p, "dtype"_a = "float64");
            arrays.push_back(nb::cast<Arr>(p));
            shape::Shape s{1};
            for (std::size_t d = 0; d < arrays.back().ndim(); ++d) s.push_back(arrays.back().shape(d));
            shapes.push_back(std::move(s));
        }
        if (copies_.empty()) start_assets(shapes);
        else if (shapes != shapes_)
            throw nb::value_error(("every event must have the same shape; the first was " +
                event_shapes(shapes_) + ", this one is " + event_shapes(shapes)).c_str());

        const std::size_t cols = plan_.columns;
        std::vector<std::size_t> out_shape(plan_.shape.begin() + 1, plan_.shape.end());
        if (n_out_ > 1) out_shape.push_back(n_out_);
        double* out = new double[cols * n_out_ ? cols * n_out_ : 1];
        for (std::size_t c = 0; c < cols; ++c) {
            for (std::size_t i = 0; i < n_in_; ++i)
                in_[i] = arrays[parts.size() == 1 ? 0 : i].data()[plan_.offset[i * cols + c]];
            copies_[c]->eval(in_.data(), out + c * n_out_);
        }
        nb::capsule owner(out, [](void* p) noexcept { delete[] (double*) p; });
        return nb::cast(nb::ndarray<nb::numpy, double>(out, out_shape.size(), out_shape.data(), owner));
    }

    // The first asset event fixes the shape; build one op per asset. Copies come
    // from the wrapper's _fresh(); the original op is reset so every asset
    // starts clean.
    void start_assets(const std::vector<shape::Shape>& shapes) {
        shapes_ = shapes;
        plan_ = shape::broadcast(shape::input_views(shapes, n_in_));
        if (plan_.columns > 1 && !nb::hasattr(op_owner_, "_fresh"))
            throw nb::type_error("events with several assets need an operator built from the "
                                 "screamer namespace, which can copy itself once per asset");
        op_.reset();
        copies_.push_back(&op_);
        for (std::size_t c = 1; c < plan_.columns; ++c) {
            owners_.push_back(op_owner_.attr("_fresh")());
            copies_.push_back(&nb::cast<EvalOp&>(owners_.back()));
        }
    }

    static std::string event_shapes(const std::vector<shape::Shape>& shapes) {
        std::string s;
        for (std::size_t i = 0; i < shapes.size(); ++i) {
            shape::Shape e(shapes[i].begin() + 1, shapes[i].end());
            s += (i ? " and " : "") + shape::to_string(e);
        }
        return s;
    }

    nb::object op_owner_;                 // keeps the functor wrapper alive
    EvalOp& op_;
    std::size_t n_in_, n_out_;
    std::vector<nb::object> iters_;
    std::vector<double> in_, out_;
    bool unpack_ = false;
    Mode mode_ = Mode::Unknown;
    std::vector<shape::Shape> shapes_;    // per argument, with a leading 1 for T
    shape::Plan plan_;
    std::vector<nb::object> owners_;      // the _fresh() copies, kept alive
    std::vector<EvalOp*> copies_;         // one op per asset; copies_[0] is op_
};
```

`shape::to_string` of an empty shape renders `()`. The header needs `using namespace nb::literals;` for `"dtype"_a`. Add it inside `namespace screamer` above the class if the file does not already have it.

- [ ] **Step 5: Build and run**

Run: `make install-dev && pytest tests/test_event_iterators.py tests/test_lazy_streaming.py tests/test_lazy_iterator_lifetime.py tests/test_functor_params.py -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add include/screamer/common/lazy_eval_iterator.h screamer/_functor_params.py tests/test_event_iterators.py
git commit -m "feat: iterators of asset events, one operator copy per asset"
```

---

### Task 5: JavaScript: N-D and broadcast calls, asset events, and `reshape`

**Files:**
- Create: `wasm/shape_bindings.cpp`
- Modify: `wasm/CMakeLists.txt:32-34` (add `"${CMAKE_CURRENT_SOURCE_DIR}/shape_bindings.cpp"` to the sources list)
- Modify: `js/src/runtime.ts` (`wrapOp`, `wrapReducerOp`, `wrapRowOp`)
- Modify: `devtools/wasm/gen_ts_api.py:163-184` (`_factory_body`); regenerate `js/src/generated/ops.ts`
- Create: `js/src/shape.ts`; Modify: `js/src/index.ts`
- Test: `js/test/shape.test.ts`

**Interfaces:**
- Consumes: `shape::plan_layout`, `shape::input_views`, `shape::broadcast` (Task 1).
- Produces (WASM module `M`):
  - `M.shapedResult(specPtr: number, nIn: number) -> number[]` gives `(T, *A)`.
  - `M.evalShaped(op, specPtr: number, outPtr: number)` writes `(T, *A, nOut)`.
  - `M.layoutPlan(layout: string, shape: number[]) -> {out: string, source: number[], dropped: number[], shape: number[]}`.
  - A spec is a heap block of doubles: `[nArgs, then per argument: dataPtr, ndim, ...dims]`.
- Produces (TS): `wrapOp(M, src: RawOp | (() => RawOp))` (and the same for `wrapReducerOp`, `wrapRowOp`), plus `reshape(x: NdArray | Float64Array | number[], layout: string): NdArray`, exported from the package.

- [ ] **Step 1: Write the failing tests**

`js/test/shape.test.ts`:

```ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { ready, RollingMean, RollingCorr, Add, Sub, BollingerBands, BacktestPriceTarget, reshape } from "../src/index.ts";
import type { NdArray } from "../src/index.ts";

const T = 50, K = 3;
const X: NdArray = { data: Float64Array.from({ length: T * K }, (_, i) => Math.sin(i) * 10 + i / 7), shape: [T, K] };
const y = Float64Array.from({ length: T }, (_, t) => Math.cos(t) * 5 + t / 3);
const col = (a: NdArray, k: number, width = a.shape[1]) =>
  Float64Array.from({ length: a.shape[0] }, (_, t) => a.data[t * width + k]);

test("one-input op on a (T, K) NdArray runs each column", async () => {
  await ready();
  const out = RollingMean(5)(X) as NdArray;
  assert.deepEqual(out.shape, [T, K]);
  for (let k = 0; k < K; k++) assert.deepEqual(col(out, k), RollingMean(5)(col(X, k)));
});

test("a (T,) input broadcasts against (T, K)", async () => {
  await ready();
  const out = RollingCorr(10)(X, y) as NdArray;
  assert.deepEqual(out.shape, [T, K]);
  for (let k = 0; k < K; k++) assert.deepEqual(col(out, k), RollingCorr(10)(col(X, k), y));
});

test("a multi-output engine broadcasts one price", async () => {
  await ready();
  const target: NdArray = { data: Float64Array.from({ length: T * K }, (_, i) => (i % 5) - 2), shape: [T, K] };
  const out = BacktestPriceTarget()(target, y) as NdArray;
  assert.deepEqual(out.shape, [T, K, 4]);
});

test("a packed (T, K, 2) array is split along its last axis", async () => {
  await ready();
  const packed: NdArray = { data: new Float64Array(T * K * 2).map((_, i) => (i % 2 ? 3 : 1)), shape: [T, K, 2] };
  const out = Sub()(packed) as NdArray;
  assert.deepEqual(out.shape, [T, K]);
  assert.ok(out.data.every((v) => v === -2));
});

test("a size-1 field axis is consumed", async () => {
  await ready();
  const out = BollingerBands(5)({ data: X.data, shape: [T, K, 1] }) as NdArray;
  assert.deepEqual(out.shape, [T, K, 3]);
});

test("mismatched assets raise", async () => {
  await ready();
  assert.throws(() => Add()({ data: new Float64Array(T * 3), shape: [T, 3] }, { data: new Float64Array(T * 4), shape: [T, 4] }), /do not broadcast/);
});

test("an iterator of K-value events equals the (T, K) batch", async () => {
  await ready();
  const rows = Array.from({ length: T }, (_, t) => X.data.subarray(t * K, (t + 1) * K));
  const got = [...(RollingMean(5)(rows[Symbol.iterator]()) as Iterable<Float64Array>)];
  const want = RollingMean(5)(X) as NdArray;
  got.forEach((r, t) => assert.deepEqual(r, want.data.subarray(t * K, (t + 1) * K)));
});

test("an event shape change raises", async () => {
  await ready();
  const it = (RollingMean(3)([new Float64Array(3), new Float64Array(4)][Symbol.iterator]()) as Iterator<any>);
  it.next();
  assert.throws(() => it.next(), /same shape/);
});

test("reshape", async () => {
  await ready();
  assert.deepEqual(reshape(X, "TA").shape, [T, K, 1]);
  assert.equal(reshape(X, "TA").data, X.data);                 // same buffer
  const t = reshape({ data: Float64Array.from([1, 2, 3, 4, 5, 6]), shape: [2, 3] }, "AT->TA");
  assert.deepEqual(t.shape, [3, 2]);
  assert.deepEqual(Array.from(t.data), [1, 4, 2, 5, 3, 6]);
  assert.throws(() => reshape({ data: new Float64Array(8), shape: [2, 4] }, "TF->T"), /only a size-1 axis/);
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd js && npm run build:wasm && node --import tsx --test test/shape.test.ts`
Expected: FAIL (`reshape` is not exported; `RollingMean(5)(X)` throws `unsupported input`).

- [ ] **Step 3: Write `wasm/shape_bindings.cpp` and register it**

```cpp
// Shape-contract entry points for JavaScript: N-D and broadcast operator calls
// and layout plans, all decided by include/screamer/common/shape.h.
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <cstdint>
#include <string>
#include <vector>
#include "screamer/common/eval_op.h"
#include "screamer/common/shape.h"

namespace {

using screamer::shape::Shape;

// A call spec on the WASM heap, as doubles: [nArgs, then per argument: data
// pointer, ndim, dims...]. Each argument is a C-contiguous block of doubles.
struct Spec {
    std::vector<const double*> data;
    std::vector<Shape> shapes;
};

Spec read_spec(std::uintptr_t p) {
    const double* s = reinterpret_cast<const double*>(p);
    Spec r;
    std::size_t n = static_cast<std::size_t>(s[0]), k = 1;
    for (std::size_t i = 0; i < n; ++i) {
        r.data.push_back(reinterpret_cast<const double*>(static_cast<std::uintptr_t>(s[k++])));
        std::size_t nd = static_cast<std::size_t>(s[k++]);
        Shape sh;
        for (std::size_t d = 0; d < nd; ++d) sh.push_back(static_cast<std::size_t>(s[k++]));
        r.shapes.push_back(std::move(sh));
    }
    return r;
}

template <class V>
emscripten::val to_js(const std::vector<V>& v) {
    emscripten::val a = emscripten::val::array();
    for (const V& x : v) a.call<void>("push", static_cast<double>(x));
    return a;
}

// The (T, *A) result shape of an nIn-input call on these arguments.
emscripten::val shapedResult(std::uintptr_t spec, std::size_t n_in) {
    Spec s = read_spec(spec);
    return to_js(screamer::shape::broadcast(screamer::shape::input_views(s.shapes, n_in)).shape);
}

// Run op over every broadcast column, resetting it per column like a batch call;
// out receives (T, *A, n_out) doubles in row-major order.
void evalShaped(screamer::EvalOp& op, std::uintptr_t spec, std::uintptr_t outPtr) {
    Spec s = read_spec(spec);
    const std::size_t ni = op.n_in(), no = op.n_out();
    const screamer::shape::Plan plan =
        screamer::shape::broadcast(screamer::shape::input_views(s.shapes, ni));
    std::vector<const double*> data(ni);
    for (std::size_t i = 0; i < ni; ++i) data[i] = s.data[s.data.size() == 1 ? 0 : i];
    double* out = reinterpret_cast<double*>(outPtr);
    std::vector<double> in(ni);
    const std::size_t cols = plan.columns;
    for (std::size_t c = 0; c < cols; ++c) {
        op.reset();
        for (std::size_t t = 0; t < plan.T; ++t) {
            for (std::size_t i = 0; i < ni; ++i)
                in[i] = data[i][plan.offset[i * cols + c] + t * plan.time_stride[i]];
            op.eval(in.data(), out + (t * cols + c) * no);
        }
    }
    op.reset();
}

emscripten::val layoutPlan(std::string layout, emscripten::val shape) {
    Shape in;
    const unsigned n = shape["length"].as<unsigned>();
    for (unsigned i = 0; i < n; ++i) in.push_back(static_cast<std::size_t>(shape[i].as<double>()));
    const auto p = screamer::shape::plan_layout(layout, in);
    emscripten::val r = emscripten::val::object();
    r.set("out", p.out);
    r.set("source", to_js(p.source));
    r.set("dropped", to_js(p.dropped));
    r.set("shape", to_js(p.shape));
    return r;
}

}  // namespace

EMSCRIPTEN_BINDINGS(screamer_shape) {
    emscripten::function("shapedResult", &shapedResult);
    emscripten::function("evalShaped", &evalShaped, emscripten::allow_raw_pointers());
    emscripten::function("layoutPlan", &layoutPlan);
}
```

- [ ] **Step 4: Factories pass a constructor closure**

In `devtools/wasm/gen_ts_api.py` `_factory_body`, make every factory hand its wrapper a closure, so the runtime can build one op per asset:

```python
def _factory_body(name: str, params: list[dict], wrapper: str = "wrapOp") -> str:
    vecs = [p for p in params if p["is_vec"]]
    call_args = [f"_v{idx}" if p["is_vec"] else p["name"] for idx, p in enumerate(params)]
    lines = ["  const M = current();"]
    if not vecs:
        lines.append(f"  return {wrapper}(M, () => new M.{name}({', '.join(call_args)}));")
        return "\n".join(lines)
    lines.append(f"  return {wrapper}(M, () => {{")
    for idx, p in enumerate(params):
        if p["is_vec"]:
            v = f"_v{idx}"
            lines.append(f"    const {v} = new M.VectorDouble();")
            # A bare number is the one-entry vector (one value for all assets).
            lines.append(f"    for (const _x of (typeof {p['name']} === \"number\" ? [{p['name']}] : {p['name']})) {v}.push_back(_x);")
    lines.append(f"    const _op = new M.{name}({', '.join(call_args)});")
    for idx, p in enumerate(params):
        if p["is_vec"]:
            lines.append(f"    _v{idx}.delete();")
    lines.append("    return _op;")
    lines.append("  });")
    return "\n".join(lines)
```

Run `python devtools/wasm/gen_ts_api.py` to regenerate `js/src/generated/ops.ts` and `ops.d.ts`.

- [ ] **Step 5: Update `js/src/runtime.ts`**

Move `const isNd = ...` above `wrapOp`, and add this helper next to it:

```ts
type OpSource = RawOp | (() => RawOp);
const resolve = (src: OpSource): RawOp => (typeof src === "function" ? (src as () => RawOp)() : src);
```

In `wrapReducerOp(M, raw)` and `wrapRowOp(M, raw)`, rename the parameter to `src: OpSource` and start each body with `const raw = resolve(src);`. Change `wrapReducerOp`'s `const base = wrapOp(M, raw);` to keep passing the resolved `raw`.

Change `wrapOp`:

```ts
export function wrapOp(M: Screamer, src: OpSource): ScreamerOp {
  const make = typeof src === "function" ? (src as () => RawOp) : null;
  const raw = resolve(src);
  const nIn = raw.nIn(), nOut = raw.nOut();
  const inBuf = M.allocF64(nIn), outBuf = M.allocF64(nOut);
  let copies: RawOp[] = [];          // ops for assets 1..K-1 of an asset-event stream
  let disposed = false;
  const free = () => {
    if (disposed) return;
    disposed = true;
    M.freeBuf(inBuf); M.freeBuf(outBuf);
    copies.forEach((c) => c.delete());
    raw.delete();
  };
```

Keep `event` and `batchInto` unchanged and add:

```ts
  // An N-D or broadcast call: the C++ shape module splits a packed argument,
  // consumes a size-1 field axis, broadcasts the assets, and runs each column.
  function shaped(args: any[]): NdArray {
    const parts: NdArray[] = args.map((a) =>
      isNd(a) ? a : { data: Float64Array.from(a as ArrayLike<number>), shape: [(a as ArrayLike<number>).length] });
    const ptrs: number[] = [];
    try {
      const spec: number[] = [parts.length];
      for (const p of parts) {
        const ptr = M.allocF64(Math.max(1, p.data.length));
        ptrs.push(ptr);
        M.viewF64(ptr, p.data.length).set(p.data);
        spec.push(ptr, p.shape.length, ...p.shape);
      }
      const specPtr = M.allocF64(spec.length);
      ptrs.push(specPtr);
      M.viewF64(specPtr, spec.length).set(spec);
      const shape: number[] = M.shapedResult(specPtr, nIn);
      const n = shape.reduce((a, b) => a * b, 1) * nOut;
      const outPtr = M.allocF64(Math.max(1, n));
      ptrs.push(outPtr);
      M.evalShaped(raw, specPtr, outPtr);
      return { data: new Float64Array(M.viewF64(outPtr, n)), shape: nOut === 1 ? shape : [...shape, nOut] };
    } finally {
      ptrs.forEach((p) => M.freeBuf(p));
    }
  }

  // One event of an asset stream: K values, one per asset, each run by its own
  // op (raw for asset 0). The first event fixes K, as a (T, K) batch does.
  let assets = -1;                    // -1: not started, 0: scalar events, K: asset events
  const eventRow = (v: any): ArrayLike<number> => {
    if (!isNd(v)) return v as ArrayLike<number>;
    const s = v.shape;
    if (s.length === 1 || (s.length === 2 && s[1] === 1)) return v.data;
    throw new TypeError(`an event must hold one value per asset; got shape [${s}]`);
  };
  const assetEvent = (row: ArrayLike<number>): Float64Array | NdArray => {
    const k = row.length;
    const out = new Float64Array(k * nOut);
    for (let j = 0; j < k; j++) {
      M.viewF64(inBuf, 1)[0] = row[j];
      (j === 0 ? raw : copies[j - 1]).evalInto(inBuf, outBuf);
      out.set(M.viewF64(outBuf, nOut), j * nOut);
    }
    return nOut === 1 ? out : { data: out, shape: [k, nOut] };
  };
  const step = (v: any) => {
    if (typeof v === "number") {
      if (assets > 0) throw new TypeError(`every event must have the same shape: the first held ${assets} values, this one is a number`);
      assets = 0;
      return event([v]);
    }
    const row = eventRow(v);
    if (assets === -1) {
      if (row.length > 1 && !make) throw new TypeError("events with several assets need an op built by its factory");
      assets = row.length;
      raw.reset();
      while (copies.length < assets - 1) copies.push(make!());
      copies.forEach((c) => c.reset());
    } else if (row.length !== assets) {
      throw new TypeError(`every event must have the same shape: the first held ${assets === 0 ? "a number" : `${assets} values`}, this one holds ${row.length}`);
    }
    return assetEvent(row);
  };

  function* gen(src: Iterable<any>) {
    raw.reset();
    assets = -1;
    for (const v of src) yield step(v);
  }
  async function* agen(src: AsyncIterable<any>) {
    raw.reset();
    assets = -1;
    for await (const v of src) yield step(v);
  }
```

In `call`, change the one-input branch to check `if (isNd(a)) return shaped([a]);` right after the number check. Change the multi-input part to:

```ts
      if (args.length === nIn && args.every((x) => typeof x === "number")) return event(args);
      if (args.length === 1 && isNd(args[0])) return shaped(args);      // one packed (T, ..., nIn) array
      if (args.length === nIn && args.every((x) => isTyped(x) || isNumArr(x) || isNd(x))) {
        if (args.some(isNd)) return shaped(args);
        // ... the existing equal-length check and interleaved batch, unchanged
      }
```

Change `op.reset` to also reset the copies: `op.reset = () => { if (disposed) throw new Error("operation used after dispose()"); raw.reset(); copies.forEach((c) => c.reset()); };`.

The old `gen`/`agen` yielded `event([v])` for numbers. `step` keeps that for number events.

- [ ] **Step 6: `js/src/shape.ts` and the export**

```ts
import { current } from "./index.js";
import { normalizeError } from "./errors.js";
import type { NdArray } from "./ndarray.js";

/**
 * Reshape data by a layout string: `T` (time), `A` (asset), `F` (field), with
 * an optional `->` and the wanted result, as in Python's `screamer.reshape`.
 * Returns the same buffer with a new shape, or a copy when axes are reordered.
 */
export function reshape(x: NdArray | Float64Array | number[], layout: string): NdArray {
  const M = current();
  const nd: NdArray = (x as NdArray).shape
    ? (x as NdArray) : { data: Float64Array.from(x as ArrayLike<number>), shape: [(x as ArrayLike<number>).length] };
  let plan: { source: number[]; shape: number[] };
  try { plan = M.layoutPlan(layout, nd.shape); } catch (e) { throw normalizeError(e); }
  const kept = plan.source.filter((s) => s >= 0);
  if (kept.every((s, i) => i === 0 || s > kept[i - 1])) return { data: nd.data, shape: plan.shape };
  const inStride = nd.shape.map((_, d) => nd.shape.slice(d + 1).reduce((a, b) => a * b, 1));
  const stride = plan.source.map((s) => (s >= 0 ? inStride[s] : 0));
  const out = new Float64Array(nd.data.length);
  const idx = new Array(plan.shape.length).fill(0);
  for (let o = 0; o < out.length; o++) {
    let off = 0;
    for (let d = 0; d < idx.length; d++) off += idx[d] * stride[d];
    out[o] = nd.data[off];
    for (let d = idx.length - 1; d >= 0; d--) { if (++idx[d] < plan.shape[d]) break; idx[d] = 0; }
  }
  return { data: out, shape: plan.shape };
}
```

In `js/src/index.ts`, add `export { reshape } from "./shape.js";`.

The spec asks JS to say "assets" where it says "groups". In `wrapReducerOp`, rename the local `groups` variables to `assets`, and reword every error message: "expected an event of shape (assets, ${width})", "a flat event must hold a positive multiple of ${width} values", "asset ${g} is not a numeric array", "each asset must hold ${width} values", "every event must hold ${assets} assets", and "expected a batch of shape (events, assets, ${width})". Update the comments the same way. Then run `grep -rn "groups" js/test/reducer.test.ts` and update any assertion that matched the old wording.

- [ ] **Step 7: Build and run the JS suite**

Run: `cd js && npm run build:wasm && npm test`
Expected: all PASS, including the existing `runtime`, `reducer`, `cross_section`, `portfolio` and `lifecycle` tests (they exercise the factory change).

- [ ] **Step 8: Commit**

```bash
git add wasm/shape_bindings.cpp wasm/CMakeLists.txt devtools/wasm/gen_ts_api.py js/src js/test/shape.test.ts
git commit -m "feat(js): N-D and broadcast calls, asset events, reshape"
```

---

### Task 6: `CombineLatest` joins fields

**Files:**
- Modify: `include/screamer/streams/combine_latest.h`
- Modify: `bindings/bindings_streams.cpp` (`combine_latest_batch`)
- Modify: `include/screamer/dag/combine_latest_node.h`
- Modify: `include/screamer/dag/compiled_graph.h:126` and `:233-242`
- Modify: `screamer/streams.py` (`combine_latest`, `_combine_latest_zip_lazy`, `_combine_latest_asof_lazy`, `_combine_latest_lazy`, `CombineLatest`)
- Test: rewrite `tests/test_combine_latest_multicolumn.py`; add a case to `js/test/combinators.test.ts`

**Interfaces:**
- Consumes: `screamer.shape.reshape_array` (Task 2), `_b._broadcast_shape` (Task 1).
- Produces:
  - `streams::CombineLatest(std::vector<std::size_t> widths, bool when_all)`, `on_event(source, const double*)`, `set_width(source, w)`, `width(source)`, `total_width()`.
  - `dag::CombineLatestNode(std::vector<std::size_t> widths, bool when_all, Sink&, max_pending)`.
  - `combine_latest(*values, index=None, emit="when_all", layout=None)` and `CombineLatest(emit="when_all", layout=None)`.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_combine_latest_multicolumn.py`:

```python
"""CombineLatest concatenates fields and broadcasts assets (shape contract rule 5)."""
import numpy as np
import pytest

import screamer as s

T = 12
bars = np.arange(T * 4.0).reshape(T, 4)
sig = np.arange(T) * 10.0


def test_eager_concatenates_fields():
    values, index = s.CombineLatest()(bars, sig)
    assert index is None
    np.testing.assert_array_equal(values, np.column_stack([bars, sig]))


def test_eager_indexed_wide_as_of():
    v, i = s.CombineLatest()((bars, np.arange(T) * 2), (sig[:3], np.array([1, 5, 9])))
    assert v.shape[1] == 5
    row = list(i).index(10)
    np.testing.assert_array_equal(v[row], [*bars[5], sig[2]])


def test_eager_on_any_wide_nan_until_seen():
    v, _ = s.CombineLatest(emit="on_any")((bars, np.arange(T) + 5), (sig, np.arange(T)))
    assert v.shape == (T + 5, 5)
    assert np.isnan(v[0, :4]).all() and v[0, 4] == sig[0]


def test_layout_broadcasts_assets():
    panel = np.arange(T * 3.0).reshape(T, 3)
    v, _ = s.CombineLatest(layout=("TA", "T"))(panel, sig)
    assert v.shape == (T, 3, 2)
    np.testing.assert_array_equal(v[..., 0], panel)
    np.testing.assert_array_equal(v[..., 1], np.repeat(sig[:, None], 3, axis=1))


def test_layout_string_applies_to_every_input():
    a, b = np.zeros((T, 3)), np.ones((T, 3))
    v, _ = s.CombineLatest(layout="TA")(a, b)
    assert v.shape == (T, 3, 2)


def test_three_d_inputs_read_as_assets_and_fields():
    v, _ = s.CombineLatest()(np.zeros((T, 3, 4)), sig)
    assert v.shape == (T, 3, 5)


def test_layout_count_must_match():
    with pytest.raises(ValueError, match="one layout per input"):
        s.CombineLatest(layout=("TA",))(bars, sig)


def test_lazy_positional_wide():
    out = list(s.CombineLatest()(iter(bars), iter(sig)))
    np.testing.assert_array_equal(np.array([r for r, _ in out]), np.column_stack([bars, sig]))


def test_lazy_indexed_wide_matches_batch():
    idx_a, idx_b = np.arange(T) * 2, np.array([1, 5, 9])
    batch, bidx = s.CombineLatest()((bars, idx_a), (sig[:3], idx_b))
    lazy = list(s.CombineLatest()(iter(zip(bars, idx_a)), iter(zip(sig[:3], idx_b))))
    np.testing.assert_array_equal(np.array([r for r, _ in lazy]), batch)
    assert [i for _, i in lazy] == list(bidx)


def test_pipeline_joins_a_bar_node_and_a_signal():
    x, y = s.Input("x"), s.Input("y")
    p = s.Pipeline([x, y], [s.CombineLatest()(s.Resample(count=2, agg="ohlc")(x), y)])
    values, _ = p(np.arange(20.0), np.arange(20.0))
    assert values.shape[1] == 5
    np.testing.assert_array_equal(values[2, :4], [2.0, 3.0, 2.0, 3.0])


def test_pipeline_wide_input_when_all():
    x, y = s.Input("x"), s.Input("y")
    p = s.Pipeline([x, y], [s.CombineLatest()(x, y)])
    values, _ = p(bars, sig)
    np.testing.assert_array_equal(values, np.column_stack([bars, sig]))


def test_pipeline_on_any_wide_input_raises_clearly():
    x, y = s.Input("x"), s.Input("y")
    p = s.Pipeline([x, y], [s.CombineLatest(emit="on_any")(x, y)])
    with pytest.raises(ValueError, match="when_all"):
        p((bars, np.arange(T) + 5), (sig, np.arange(T)))


def test_pipeline_rejects_asset_layout():
    x, y = s.Input("x"), s.Input("y")
    with pytest.raises(ValueError, match="asset"):
        s.CombineLatest(layout=("TA", "T"))(x, y)
```

Add to `js/test/combinators.test.ts`, using the file's existing imports and helpers:

```ts
test("combineLatest joins a 4-column bar node and a signal", async () => {
  await ready();
  const x = Input("x"), y = Input("y");
  const p = new Pipeline([x, y], [combineLatest([resample(x, { count: 2, agg: "ohlc" }), y])]);
  const xs = Float64Array.from({ length: 20 }, (_, i) => i);
  const out: any = p.run(xs, xs);
  assert.equal(out.values.shape[1], 5);
});
```

Before writing it, read the top of `js/test/combinators.test.ts` and match its exact `Pipeline` call and result field names. If the pipeline returns `{ values, index }` under different names, use those.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_combine_latest_multicolumn.py -q`
Expected: FAIL (the eager path raises "one-column streams"; `layout` is an unexpected keyword).

- [ ] **Step 3: `streams::CombineLatest` with widths**

Replace the class in `include/screamer/streams/combine_latest.h`:

```cpp
// As-of latest-value join state for N sources fed a tagged event stream (in
// index order). Source i carries widths[i] values per event (its fields); the
// aligned row concatenates every source's latest fields in source order.
// on_event() updates the emitting source and returns whether a row is due:
//   when_all: only once every source has produced at least one event
//   on_any:   always (not-yet-seen sources read as NaN)
class CombineLatest {
public:
    CombineLatest(std::vector<std::size_t> widths, bool when_all)
        : widths_(std::move(widths)), seen_(widths_.size(), 0),
          seen_count_(0), when_all_(when_all) { layout(); }

    CombineLatest(std::size_t n, bool when_all)
        : CombineLatest(std::vector<std::size_t>(n, 1), when_all) {}

    bool on_event(std::uint32_t source, const double* values) {
        assert(source < widths_.size());
        if (!seen_[source]) { seen_[source] = 1; ++seen_count_; }
        std::copy(values, values + widths_[source], latest_.begin() + offset_[source]);
        return !(when_all_ && seen_count_ < widths_.size());
    }

    // A one-column source.
    bool on_event(std::uint32_t source, double value) {
        assert(widths_[source] == 1);
        return on_event(source, &value);
    }

    // Change the width of a source that has no value yet; the others keep theirs.
    void set_width(std::size_t source, std::size_t w) {
        if (seen_[source])
            throw std::logic_error("CombineLatest: a source's width is fixed by its first event");
        const std::vector<double> old = latest_;
        const std::vector<std::size_t> old_offset = offset_, old_width = widths_;
        widths_[source] = w;
        layout();
        for (std::size_t i = 0; i < widths_.size(); ++i)
            if (seen_[i])
                std::copy(old.begin() + old_offset[i], old.begin() + old_offset[i] + old_width[i],
                          latest_.begin() + offset_[i]);
    }

    std::size_t width(std::size_t source) const { return widths_[source]; }
    std::size_t total_width() const { return latest_.size(); }
    const std::vector<double>& latest() const { return latest_; }

    void reset() {
        std::fill(latest_.begin(), latest_.end(), std::numeric_limits<double>::quiet_NaN());
        std::fill(seen_.begin(), seen_.end(), static_cast<char>(0));
        seen_count_ = 0;
    }

private:
    void layout() {
        offset_.resize(widths_.size());
        std::size_t total = 0;
        for (std::size_t i = 0; i < widths_.size(); ++i) { offset_[i] = total; total += widths_[i]; }
        latest_.assign(total, std::numeric_limits<double>::quiet_NaN());
    }

    std::vector<std::size_t> widths_, offset_;
    std::vector<double> latest_;
    std::vector<char> seen_;
    std::size_t seen_count_;
    bool when_all_;
};
```

Add `#include <stdexcept>` and `#include <utility>`.

- [ ] **Step 4: Eager binding reads 2-D value arrays**

In `bindings/bindings_streams.cpp`, rewrite `combine_latest_batch<Index>`. Python now always passes `(T_i, w_i)` C-contiguous float64 arrays. The merge orders events by index alone, and each event carries its row position, so the row is read from its array:

```cpp
template <class Index>
static nb::object combine_latest_batch(nb::list index_arrays,
                                       nb::list value_arrays,
                                       bool when_all) {
    using In2D = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
    const std::size_t n = index_arrays.size();
    if (n == 0) throw std::runtime_error("combine_latest: needs at least one stream");
    if (value_arrays.size() != n) throw std::runtime_error("streams: index/values list length mismatch");

    // Merge on the index alone: each source's event value is its row position,
    // and the row's fields are read from that source's (rows, width) array.
    std::vector<In1D<Index>> indices;
    std::vector<In2D> vals;
    std::vector<std::vector<double>> positions(n);
    std::vector<std::unique_ptr<VectorSource<Index>>> sources;
    std::vector<Source<Index>*> child_ptrs;
    std::vector<std::size_t> widths;
    std::size_t total = 0, W = 0;
    for (std::size_t i = 0; i < n; ++i) {
        indices.push_back(nb::cast<In1D<Index>>(index_arrays[i]));
        vals.push_back(nb::cast<In2D>(value_arrays[i]));
        const std::size_t len = indices[i].shape(0);
        if (vals[i].shape(0) != len) throw std::runtime_error("streams: a child's index/values length differ");
        widths.push_back(vals[i].shape(1));
        W += widths.back();
        positions[i].resize(len);
        for (std::size_t r = 0; r < len; ++r) positions[i][r] = static_cast<double>(r);
        sources.push_back(std::make_unique<VectorSource<Index>>(indices[i].data(), positions[i].data(), len));
        child_ptrs.push_back(sources.back().get());
        total += len;
    }

    std::vector<Index> out_k;
    std::vector<double> out_v;
    out_k.reserve(total);
    out_v.reserve(total * W);

    // Coalesce to one row per distinct index, exactly as the graph
    // CombineLatestNode does.
    CombineLatest cl(widths, when_all);
    MergeSource<Index> merge(child_ptrs);
    bool has_buffered = false;
    Index buffered_index{};
    std::vector<double> buffered_row(W, 0.0);
    while (auto e = merge.next()) {
        const std::size_t row = static_cast<std::size_t>(e->value);
        if (cl.on_event(e->source, vals[e->source].data() + row * widths[e->source])) {
            const std::vector<double>& latest = cl.latest();
            if (has_buffered && e->index != buffered_index) {
                out_k.push_back(buffered_index);
                out_v.insert(out_v.end(), buffered_row.begin(), buffered_row.end());
            }
            buffered_index = e->index;
            std::copy(latest.begin(), latest.end(), buffered_row.begin());
            has_buffered = true;
        }
    }
    if (has_buffered) {
        out_k.push_back(buffered_index);
        out_v.insert(out_v.end(), buffered_row.begin(), buffered_row.end());
    }

    const std::size_t m = out_k.size();
    Index* rk = new Index[m ? m : 1];
    if (m) std::memcpy(rk, out_k.data(), m * sizeof(Index));
    double* rv = new double[(m * W) ? (m * W) : 1];
    if (m) std::memcpy(rv, out_v.data(), m * W * sizeof(double));
    return nb::make_tuple(owned_array(rk, {m}), owned_array(rv, {m, W}));
}
```

`CombineLatestPuller` keeps the `(n, when_all)` constructor and the one-value `on_event`.

- [ ] **Step 5: The graph node learns widths and concatenates**

In `include/screamer/dag/combine_latest_node.h`:

- Change the constructor to take `std::vector<std::size_t> widths` and keep the old signature delegating with `std::vector<std::size_t>(n, 1)`. Store `declared_ = widths`, `learned_.assign(n, false)`, and size `buffered_row_` to `cl_.total_width()`.
- Replace `Pending`'s `double value` with `std::size_t slot`. Add a per-port value pool: `std::vector<std::vector<double>> pool_` (one per port) and `std::vector<std::vector<std::size_t>> free_slots_`. A pending frame copies its `width` values into a free slot of its port's pool.
- Replace `on_port`:

```cpp
    void on_port(std::size_t i, const Frame<Index>& f) {
        if (!learned_[i]) {
            // A port's width is fixed by its first event. Before that, a row
            // emitted under emit="on_any" used the compile-time estimate.
            learned_[i] = true;
            if (f.width != cl_.width(i)) {
                if (emitted_any_)
                    throw std::invalid_argument(
                        "CombineLatest: input " + std::to_string(i) + " has " +
                        std::to_string(f.width) + "-column events, but rows were already "
                        "emitted with " + std::to_string(cl_.width(i)) + " column(s) for it. "
                        "With emit=\"on_any\" inside a Pipeline, a row can be due before every "
                        "input has arrived; use emit=\"when_all\".");
                cl_.set_width(i, f.width);
                buffered_row_.assign(cl_.total_width(), 0.0);
            }
        } else if (f.width != cl_.width(i)) {
            throw std::invalid_argument(
                "CombineLatest: input " + std::to_string(i) + " changed from " +
                std::to_string(cl_.width(i)) + " to " + std::to_string(f.width) + " columns");
        }
        if (wm_[i] < f.index) wm_[i] = f.index;
        pending_.push(Pending{f.index, static_cast<std::uint32_t>(i), next_seq_++, store(i, f.values, f.width)});
        if (pending_.size() > max_pending_)
            throw std::runtime_error(
                "CombineLatest reorder buffer overflow: an input has not advanced; "
                "a stream is stalled. Call advance(now) or check the feed.");
        release();
    }

    std::size_t store(std::size_t port, const double* values, std::size_t w) {
        std::size_t slot;
        if (!free_slots_[port].empty()) { slot = free_slots_[port].back(); free_slots_[port].pop_back(); }
        else { slot = pool_[port].size() / w; pool_[port].resize(pool_[port].size() + w); }
        std::copy(values, values + w, pool_[port].begin() + slot * w);
        return slot;
    }
```

- In `release()`, apply with `apply_ordered(p.port, p.index, &pool_[p.port][p.slot * cl_.width(p.port)])`, then `free_slots_[p.port].push_back(p.slot)`.
- `apply_ordered(std::size_t i, Index ev_index, const double* values)` calls `cl_.on_event(static_cast<std::uint32_t>(i), values)`. When that returns true, set `emitted_any_ = true` before buffering. All three `downstream_.push(Frame<Index>{..., buffered_row_.data(), n_})` calls use `buffered_row_.size()` as the width.
- `reset()` additionally does `cl_ = screamer::streams::CombineLatest(declared_, when_all_);`, `std::fill(learned_.begin(), learned_.end(), false); emitted_any_ = false; buffered_row_.assign(cl_.total_width(), 0.0);`, and clears every `pool_[p]` and `free_slots_[p]`. Store `when_all_` as a member, because `CombineLatest` does not expose it.
- Remove the old `f.width != 1` throw.

In `include/screamer/dag/compiled_graph.h`, change the width inference at line 126:

```cpp
            case NodeKind::CombineLatest: {
                std::size_t w = 0;
                for (auto in : nd.inputs) w += node_width[in];
                node_width[id] = w;
                break;
            }
```

At the construction (around line 233), pass the per-input estimates:

```cpp
            case NodeKind::CombineLatest: {
                std::vector<std::size_t> widths;
                for (auto in : ns.inputs) widths.push_back(node_width[in]);
                auto cn = std::make_shared<CombineLatestNode<std::int64_t>>(
                    widths, ns.when_all, *downstream, ns.max_pending);
```

- [ ] **Step 6: Python `combine_latest` with fields and `layout`**

In `screamer/streams.py`, add these helpers above `combine_latest`:

```python
def _as_fields(v, layout):
    """Read one input as (T, *A, F): rule 5 reads a 1-D or 2-D array as fields,
    a 3-D or wider array as (T, *A, F); a layout states another reading."""
    v = np.asarray(v, dtype=np.float64)
    if layout is not None:
        from .shape import reshape_array
        return reshape_array(v, layout)
    return v[:, None] if v.ndim == 1 else v


def _layouts(layout, n):
    if layout is None or isinstance(layout, str):
        return [layout] * n
    layout = list(layout)
    if len(layout) != n:
        raise ValueError(f"combine_latest: pass one layout per input ({n}), or one string for all")
    return layout


def _assemble(rows, fields, assets):
    """Split joined rows into each input's (*A_i, F_i) block, broadcast the
    assets and concatenate the fields (rule 5)."""
    T = rows.shape[0]
    parts, start = [], 0
    for f in fields:
        w = int(np.prod(f.shape[1:]))
        block = rows[:, start:start + w].reshape((T,) + f.shape[1:])
        parts.append(np.broadcast_to(block, (T,) + assets + f.shape[-1:]))
        start += w
    return np.concatenate(parts, axis=-1)


def _join_spec(heads, layouts):
    """(fields, assets, wide) from one sample per input of shape (T, ...)."""
    fields = [_as_fields(h, lay) for h, lay in zip(heads, layouts)]
    assets = tuple(_b._broadcast_shape([[1, *f.shape[1:-1]] for f in fields])[1:])
    wide = bool(assets) or any(f.shape[-1] != 1 for f in fields)
    return fields, assets, wide
```

Replace the eager tail of `combine_latest` (from `streams = _to_streams(values, index)` to the return):

```python
    streams = _to_streams(values, index)
    kind, idx, vals, positional = _streams_to_indexed(streams, "combine_latest")
    fields, assets, wide = _join_spec(vals, _layouts(layout, len(vals)))
    flat = [np.ascontiguousarray(f.reshape(f.shape[0], -1)) for f in fields]
    fn = _b._combine_latest_f64 if kind == "f64" else _b._combine_latest_i64
    out_index, rows = fn(idx, flat, emit == "when_all")   # coalesced in C++
    aligned = _assemble(rows, fields, assets) if assets else rows
    return aligned, (None if positional else out_index)
```

`_streams_to_indexed` calls `np.ascontiguousarray` on each value array, which keeps N-D shapes. With no assets, `rows` is already the concatenated fields. Add `layout=None` to the signature and the docstring: "``layout`` names each input's axes (one string per input, or one for all), as in :func:`reshape`; inputs with asset axes broadcast them and concatenate their fields. By default a 1-D or 2-D input is fields and a 3-D input is ``(T, A, F)``." Before the `is_node` check, add:

```python
    if any(is_node(v) for v in values):
        if any(lay and "A" in lay.split("->")[-1] for lay in _layouts(layout, len(values))):
            raise ValueError(
                "CombineLatest(layout=...) with an asset axis runs on arrays and iterators; "
                "streams inside a Pipeline carry fields only")
        return make_operator_node(CombineLatest, values, {"emit": emit})
```

Lazy paths. In `_combine_latest_lazy(values, emit, layout)`, after `_classify_lazy_sources`, peek the first event of each source to build the spec. `sources` are chained iterators, so re-chain the peeked heads:

```python
def _combine_latest_lazy(values, emit, layout=None):
    import itertools
    positional, sources, _ = _classify_lazy_sources(values, "combine_latest")
    heads = []
    for k, src in enumerate(sources):
        h = next(src, _EMPTY)
        heads.append(h)
        if h is not _EMPTY:
            sources[k] = itertools.chain([h], src)
    if any(h is _EMPTY for h in heads):
        return
    samples = [np.asarray(h if positional else h[0], dtype=np.float64)[None] for h in heads]
    fields, assets, wide = _join_spec(samples, _layouts(layout, len(samples)))
    if positional:
        rows = _combine_latest_zip_lazy(sources, wide)
    else:
        # Wide events enter the Pipeline as flat rows; the node concatenates them.
        rows = _combine_latest_asof_lazy(
            [(((np.ravel(np.asarray(v, dtype=np.float64)) if wide else v), i) for v, i in src)
             for src in sources], emit)
    for row, i in rows:
        if wide:
            row = np.asarray(row, dtype=np.float64)
            row = _assemble(row[None], fields, assets)[0] if assets else row
        yield row, i
```

`_combine_latest_zip_lazy(sources, wide)` keeps its stay-in-step check. It yields `tuple(float(x) for x in tup)` when not `wide`, and otherwise `np.concatenate([np.ravel(np.asarray(x, dtype=np.float64)) for x in tup])`. Pass `layout` from `combine_latest` into `_combine_latest_lazy(values, emit, layout)`.

Change the class:

```python
class CombineLatest:
    """As-of latest-value join.  Config-first form of :func:`combine_latest`.

    ``CombineLatest(emit="when_all", layout=None)(*values, index=None)`` is
    equivalent to ``combine_latest(*values, index=None, emit=..., layout=...)``.
    """

    def __init__(self, emit="when_all", layout=None):
        self._emit = emit
        self._layout = layout

    def __call__(self, *values, index=None):
        return combine_latest(*values, index=index, emit=self._emit, layout=self._layout)
```

- [ ] **Step 7: Build and run**

Run: `make install-dev && pytest tests/test_combine_latest_multicolumn.py tests/ -q -k "combine or dag or pipeline or stream" && cd js && npm run build:wasm && node --import tsx --test test/combinators.test.ts`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add include/screamer/streams/combine_latest.h bindings/bindings_streams.cpp include/screamer/dag/combine_latest_node.h include/screamer/dag/compiled_graph.h screamer/streams.py tests/test_combine_latest_multicolumn.py js/test/combinators.test.ts
git commit -m "feat: CombineLatest concatenates fields and broadcasts assets (layout=)"
```

---

### Task 7: `Resample` per column

**Files:**
- Modify: `include/screamer/dag/resample_params.h` (add `make_per_column_plan`)
- Modify: `include/screamer/dag/resample_node.h` (`push`, `reset`)
- Modify: `screamer/streams.py` (`resample` eager branch at `elif vals.ndim != 1`, and its lazy branch)
- Test: `tests/test_resample_columns.py`

**Interfaces:**
- Produces: `std::vector<ResamplePlanEntry> make_per_column_plan(ResampleAgg agg, std::size_t W)`. Python `Resample(agg=<builtin>)` on `(T, *A)` returns `(T', *A)`, or `(T', *A, 4)` for `"ohlc"`.

- [ ] **Step 1: Write the failing tests**

`tests/test_resample_columns.py`:

```python
"""Resample aggregates every column on its own (shape contract rule 4)."""
import numpy as np
import pytest

import screamer as s

rng = np.random.default_rng(2)
T, K = 60, 3
X = rng.normal(size=(T, K)).cumsum(axis=0)
idx = np.arange(T) * 10


def _per_column(agg, data=X):
    return [s.Resample(freq=50, agg=agg)((data[:, k], idx))[0] for k in range(data.shape[1])]


@pytest.mark.parametrize("agg", ["first", "last", "min", "max", "sum", "count", "mean"])
def test_single_value_aggs(agg):
    v, i = s.Resample(freq=50, agg=agg)((X, idx))
    assert v.shape == (len(i), K)
    for k, col in enumerate(_per_column(agg)):
        np.testing.assert_array_equal(v[:, k], col)


def test_ohlc_appends_a_field_axis():
    v, i = s.Resample(freq=50, agg="ohlc")((X, idx))
    assert v.shape == (len(i), K, 4)
    for k, col in enumerate(_per_column("ohlc")):
        np.testing.assert_array_equal(v[:, k], col)


def test_unit_field_consumed():
    v, _ = s.Resample(freq=50, agg="ohlc")((X[:, :, None], idx))
    assert v.shape[1:] == (K, 4)


def test_t_by_1_keeps_its_column():
    v, _ = s.Resample(count=5, agg="last")(X[:, :1])
    assert v.shape[1:] == (1,)


def test_count_mode_and_positional():
    v, _ = s.Resample(count=7, agg="max")(X)
    assert v.shape == (int(np.ceil(T / 7)), K)


def test_lazy_matches_batch():
    batch, bidx = s.Resample(freq=50, agg="ohlc")((X, idx))
    lazy = list(s.Resample(freq=50, agg="ohlc")(iter(zip(X, idx))))
    np.testing.assert_array_equal(np.array([v for v, _ in lazy]), batch)
    assert [i for _, i in lazy] == list(bidx)


def test_pipeline_wide_input():
    x = s.Input("x")
    p = s.Pipeline([x], [s.Resample(count=5, agg="last")(x)])
    v, _ = p(X)
    np.testing.assert_array_equal(v, X[4::5])


def test_functor_agg_takes_one_column():
    with pytest.raises(ValueError, match="one column"):
        s.Resample(count=5, agg=s.ExpandingSkew())(X)


def test_ohlc_bars_unchanged():
    bars = np.column_stack([X[:, 0]] * 4)
    v, _ = s.Resample(count=5, agg="ohlc_bars")(bars)
    assert v.shape[1] == 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_resample_columns.py -q`
Expected: FAIL with `ValueError: resample: expects a 1-D value stream`.

- [ ] **Step 3: The per-column plan in C++**

In `include/screamer/dag/resample_params.h`, after `make_ohlcv_bars_plan`:

```cpp
// Per-column plan for a builtin single-column agg on a W-column stream: every
// column is aggregated on its own (shape contract rule 4), and Ohlc gives four
// outputs per column, (open, high, low, close).
inline std::vector<ResamplePlanEntry> make_per_column_plan(ResampleAgg agg, std::size_t W) {
    std::vector<ResamplePlanEntry> plan;
    plan.reserve(agg == ResampleAgg::Ohlc ? 4 * W : W);
    for (std::size_t c = 0; c < W; ++c) {
        if (agg == ResampleAgg::Ohlc) {
            plan.push_back({ResampleAgg::First, c});
            plan.push_back({ResampleAgg::Max,   c});
            plan.push_back({ResampleAgg::Min,   c});
            plan.push_back({ResampleAgg::Last,  c});
        } else {
            plan.push_back({agg, c});
        }
    }
    return plan;
}
```

Check that `ResamplePlanEntry`'s field order is `{agg, input_col}` (as `make_ohlcv_bars_plan` uses it) and that `input_col` is `std::size_t`. Cast with `static_cast` if it is narrower.

In `ResampleNode::push`, replace the single-column `if (f.width != 1) throw ...` with:

```cpp
            if (f.width != 1) {
                // A wide stream under a single-column agg: aggregate every column
                // on its own. The plan lasts until reset(), so a reused Pipeline
                // can take a different width next call.
                p_.plan = make_per_column_plan(p_.agg, f.width);
                per_column_ = true;
                num_accums_ = f.width;
                out_width_  = p_.plan.size();
                out_.assign(out_width_, 0.0);
                last_emitted_.assign(out_width_, std::numeric_limits<double>::quiet_NaN());
                nan_row_.assign(out_width_, std::numeric_limits<double>::quiet_NaN());
                accums_.resize(num_accums_);
                reset_accums();
                if (p_.mode == ResampleMode::ByIndex) push_by_index_multi(f.index, f.values);
                else push_by_count_multi(f.index, f.values);
                return;
            }
```

At the start of `reset()`:

```cpp
        if (per_column_) {             // forget the width learned from the last stream
            p_.plan.clear();
            per_column_ = false;
            num_accums_ = 1;
            out_width_  = resample_output_width(p_);
            out_.assign(out_width_, 0.0);
            last_emitted_.assign(out_width_, std::numeric_limits<double>::quiet_NaN());
            nan_row_.assign(out_width_, std::numeric_limits<double>::quiet_NaN());
            accums_.resize(1);
        }
```

Add the member `bool per_column_ = false;` next to `out_width_`. Update the class comment's "Single-column mode" paragraph: "Single-column mode (plan empty): a width-1 stream uses one accumulator. A wider stream switches to a per-column plan on its first frame (`make_per_column_plan`) until reset()."

- [ ] **Step 4: Python eager and lazy**

In `screamer/streams.py`, define next to `_BAR_AGGS_NCOL`:

```python
_COLUMN_AGGS = frozenset(("first", "last", "min", "max", "sum", "count", "mean", "ohlc"))


def _column_shape(shape):
    """The (*A) part of a (T, *A) or (T, *A, 1) value array (rule 3)."""
    trailing = tuple(shape[1:])
    return trailing[:-1] if len(trailing) >= 2 and trailing[-1] == 1 else trailing
```

Replace the final `elif vals.ndim != 1: raise ValueError("resample: expects a 1-D value stream")` with:

```python
    elif vals.ndim != 1 and not (isinstance(agg, str) and agg in _COLUMN_AGGS):
        raise ValueError(
            f"resample: a functor agg aggregates one column; got shape {vals.shape}. "
            "Resample each column with its own functor, or use a builtin agg.")
    columns = _column_shape(vals.shape) if vals.ndim > 1 else None
    if columns is not None:
        vals = vals.reshape(len(vals), -1)
```

After `out_v, out_idx = _resample_via_cpp(...)` at the end of `resample`:

```python
    if columns is not None:
        out_v = np.asarray(out_v).reshape(
            (len(out_idx),) + columns + ((4,) if agg == "ohlc" else ()))
    return out_v, out_idx
```

Replace the lazy branch body (`if _is_lazy_stream(values): return _resample_via_cpp(values, ...)`) with `return _resample_lazy(values, every=every, count=count, agg=agg, origin=origin, label=label, fill=fill)` and add:

```python
def _resample_lazy(values, *, every, count, agg, origin, label, fill):
    """Lazy resample; events whose value is an array of assets are resampled per
    column and come back in the same shape (rule 4)."""
    import itertools
    it = iter(values)
    head = next(it, _EMPTY)
    if head is _EMPTY:
        return
    events = itertools.chain([head], it)
    v0 = np.asarray(head[0], dtype=np.float64)
    if v0.ndim == 0 or not (isinstance(agg, str) and agg in _COLUMN_AGGS):
        yield from _resample_via_cpp(events, every=every, count=count, threshold=None,
                                     agg=agg, origin=origin, label=label, fill=fill)
        return
    shape = _column_shape((1,) + v0.shape) + ((4,) if agg == "ohlc" else ())
    flat = ((np.ravel(np.asarray(v, dtype=np.float64)), i) for v, i in events)
    for v, i in _resample_via_cpp(flat, every=every, count=count, threshold=None,
                                  agg=agg, origin=origin, label=label, fill=fill):
        yield np.asarray(v, dtype=np.float64).reshape(shape), i
```

The lazy branch is now a generator function, so constructing it consumes nothing until the first `next()`, which keeps the lazy contract.

- [ ] **Step 5: Build and run**

Run: `make install-dev && pytest tests/test_resample_columns.py tests/ -q -k "resample"`
Expected: all PASS. If an existing test asserted that a 2-D input with a single-column agg raises "expects a 1-D value stream", change it to assert the per-column result.

- [ ] **Step 6: Commit**

```bash
git add include/screamer/dag/resample_params.h include/screamer/dag/resample_node.h screamer/streams.py tests/
git commit -m "feat: Resample aggregates each column; ohlc appends a field axis"
```

---

### Task 8: Remove the duplicated broadcasting from validation

**Files:**
- Modify: `screamer/validation.py:317-319, 349-350, 362-363, 842`
- Modify: `js/src/validation.ts` (`positionSingle`, `positionFamily`, `walkForward`)
- Modify: `docs/recipes/parameter-grid.md:43`, `docs/recipes/walk-forward.md:70`
- Test: the existing `tests/test_validation*.py`, `tests/test_walk_forward.py`, `js/test/validation.test.ts`, `js/test/walk_forward.test.ts`, plus one new JS test

**Interfaces:**
- Consumes: broadcasting in the Python dispatch (Task 3) and `shaped` in `wrapOp` (Task 5).

- [ ] **Step 1: Add a JS test that pins chunked engine calls**

In `js/test/validation.test.ts`, using the file's existing fixtures:

```ts
test("positionTest calls the engine once per chunk with a (T, k) block", async () => {
  await ready();
  const engine = BacktestPriceTarget();
  let calls = 0;
  const counted: any = (...a: any[]) => { calls++; return engine(...a); };
  counted.reset = () => engine.reset();
  const T = 300;
  const price = Float64Array.from({ length: T }, (_, t) => 100 + Math.sin(t / 9) * 5 + t / 50);
  const target = Float64Array.from({ length: T }, (_, t) => (Math.sin(t / 13) > 0 ? 1 : -1));
  const r = positionTest(counted, target, [price], { n: 50, seed: 1 });
  assert.equal(r.null.length > 0, true);
  assert.equal(calls, 2);        // the observed run and one chunk of 50
});
```

If `positionTest`'s signature in the file differs (for example, market passed as rest arguments), follow the existing tests' call form.

- [ ] **Step 2: Run it to verify it fails**

Run: `cd js && node --import tsx --test test/validation.test.ts`
Expected: FAIL (`calls` is 51).

- [ ] **Step 3: Python**

In `screamer/validation.py`:
- Lines 317-319: delete `cols = [...]` and call `out = np.asarray(engine(surr, *market), dtype=float)`.
- Lines 349-350: `observed_out = np.asarray(engine(target, *market), dtype=float)`.
- Lines 362-363: `out = np.asarray(engine(surr, *market), dtype=float)`.
- Line 842: `full = np.asarray(engine(targets, *prices), dtype=float).reshape(T, K, 4)`.

`market` and `prices` hold `(T,)` series (`_as_series`), which now broadcast inside the engine without being copied K times.

- [ ] **Step 4: JavaScript**

Add next to `column` in `js/src/validation.ts`:

```ts
// Column k of a (T, K, 4) engine output, as one run's (T, 4) output.
function run(out: NdArray, k: number): NdArray {
  const [T, K, F] = out.shape;
  const data = new Float64Array(T * F);
  for (let t = 0; t < T; t++) data.set(out.data.subarray((t * K + k) * F, (t * K + k + 1) * F), t * F);
  return { data, shape: [T, F] };
}
```

`positionSingle`: replace the inner per-column loop with one engine call per chunk:

```ts
      engine.reset();
      const out = engine({ data: block.data, shape: [T, k] }, ...market) as NdArray;
      for (let j = 0; j < k; j++) {
        const r = scorer.score(run(out, j));
        STAT_NAMES.forEach((name, i) => { nullStats[name][c0 + j] = r.stats[i]; });
        if (Number.isFinite(r.value)) nul.push(r.value);
      }
```

Delete the now unused `col` buffer. `positionFamily`: for each target `k`, call the engine once on `{ data: block.data, shape: [T, m] }` and score `run(out, j)` for each `j`, writing `nul[(c0 + j) * K + k]` as today. `walkForward`: replace `cols.map(...)` with one call on the `(T, K)` block:

```ts
    const block = new Float64Array(T * K);
    cols.forEach((x, k) => { for (let t = 0; t < T; t++) block[t * K + k] = x[t]; });
    engine.reset();
    const all = engine({ data: block, shape: [T, K] }, ...market) as NdArray;
    const full = cols.map((_, k) => run(all, k));
```

- [ ] **Step 5: Recipes**

`docs/recipes/parameter-grid.md:43` becomes `out = engine(target, o, h, l, c)`. `docs/recipes/walk-forward.md:70` becomes `full = np.asarray(engine(X, o, h, l, c))`. Remove any surrounding sentence that explains the manual broadcast, and add one plain sentence that the `(T,)` prices broadcast against the `(T, K)` targets.

- [ ] **Step 6: Run everything that touches validation**

Run: `pytest tests/ -q -k "validation or walk_forward or position or strategy" && cd js && node --import tsx --test test/validation.test.ts test/walk_forward.test.ts`
Expected: all PASS, with identical p-values and statistics (the seeds and surrogates are unchanged).

- [ ] **Step 7: Commit**

```bash
git add screamer/validation.py js/src/validation.ts js/test/validation.test.ts docs/recipes/parameter-grid.md docs/recipes/walk-forward.md
git commit -m "refactor: validation relies on engine broadcasting"
```

---

### Task 9: Contract tests in Python and JavaScript

**Files:**
- Create: `tests/test_shape_contract.py`
- Create: `devtools/wasm/gen_shape_oracle.py`; generate `wasm/smoke/shape_oracle.json`
- Create: `js/test/shape_contract.test.ts`

**Interfaces:**
- Consumes: `FUNCTORS`, `_instantiate`, `_inputs` from `tests/test_contract_compliance.py` (registry-driven; new operators enroll automatically).

- [ ] **Step 1: Write the Python contract module**

`tests/test_shape_contract.py`:

```python
"""The shape contract, asserted against every operator in the registry.

An array is (T, *A, F?): K assets give K independent series, a size-1 field
axis is consumed, separate inputs broadcast over assets, a packed array splits
along its last axis, and an iterator of events gives the batch result. See
docs/polymorphic_api.md.
"""
import numpy as np
import pytest

from tests.test_contract_compliance import FUNCTORS, _instantiate, _inputs

K = 3
NAMES = sorted(FUNCTORS)


def _panel(n_in, seed=7):
    cols = [_inputs(n_in, seed=seed + k) for k in range(K)]          # K lists of n_in (T,)
    return [np.stack([cols[k][i] for k in range(K)], axis=1) for i in range(n_in)]


def _out(name, arrays):
    return np.asarray(_instantiate(FUNCTORS[name])(*arrays), dtype=float)


@pytest.mark.parametrize("name", NAMES)
def test_assets_are_independent_series(name):
    n_in = int(FUNCTORS[name].get("inputs", 1))
    panel = _panel(n_in)
    got = _out(name, panel)
    for k in range(K):
        np.testing.assert_array_equal(got[:, k], _out(name, [p[:, k] for p in panel]))


@pytest.mark.parametrize("name", NAMES)
def test_unit_field_is_consumed(name):
    n_in = int(FUNCTORS[name].get("inputs", 1))
    panel = _panel(n_in)
    np.testing.assert_array_equal(_out(name, [p[:, :, None] for p in panel]), _out(name, panel))


@pytest.mark.parametrize("name", [n for n in NAMES if int(FUNCTORS[n].get("inputs", 1)) > 1])
def test_one_d_inputs_broadcast(name):
    n_in = int(FUNCTORS[name]["inputs"])
    panel = _panel(n_in)
    shared = [p[:, 0] for p in panel[1:]]
    want = _out(name, [panel[0]] + [np.repeat(s[:, None], K, axis=1) for s in shared])
    np.testing.assert_array_equal(_out(name, [panel[0]] + shared), want)
    np.testing.assert_array_equal(_out(name, [panel[0]] + [s[:, None] for s in shared]), want)


@pytest.mark.parametrize("name", [n for n in NAMES if int(FUNCTORS[n].get("inputs", 1)) > 1])
def test_incompatible_assets_raise(name):
    n_in = int(FUNCTORS[name]["inputs"])
    arrays = [np.ones((50, 3))] + [np.ones((50, 4))] * (n_in - 1)
    with pytest.raises(ValueError, match="do not broadcast"):
        _instantiate(FUNCTORS[name])(*arrays)


@pytest.mark.parametrize("name", [n for n in NAMES if int(FUNCTORS[n].get("inputs", 1)) > 1])
def test_packed_array_splits(name):
    panel = _panel(int(FUNCTORS[name]["inputs"]))
    packed = np.stack(panel, axis=-1)
    np.testing.assert_array_equal(np.asarray(_instantiate(FUNCTORS[name])(packed)), _out(name, panel))


@pytest.mark.parametrize("name", NAMES)
def test_iterator_of_events_equals_batch(name):
    n_in = int(FUNCTORS[name].get("inputs", 1))
    panel = _panel(n_in)
    op = _instantiate(FUNCTORS[name])
    events = list(op(*[iter(p) for p in panel]))
    np.testing.assert_array_equal(np.array(events, dtype=float), _out(name, panel))
```

If an operator has a stated structural reason it cannot meet one of these properties, add it to a `KNOWN_*` set with an `xfail(strict=True)` marker, following the pattern in `test_contract_compliance.py` (`_xfail_if_in`, `_params`). Write the reason in a comment beside the set.

- [ ] **Step 2: Run it**

Run: `pytest tests/test_shape_contract.py -q`
Expected: PASS. A failure here is a real contract break in Tasks 3 to 4. Fix the dispatch rather than adding an `xfail`, unless the operator is structurally exempt.

- [ ] **Step 3: The golden file for JavaScript**

`devtools/wasm/gen_shape_oracle.py`:

```python
"""Write wasm/smoke/shape_oracle.json: shape-contract cases computed by the
Python binding, which js/test/shape_contract.test.ts reproduces in WASM."""
import json
import os

import numpy as np

import screamer as s

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "wasm", "smoke", "shape_oracle.json")

T, K = 40, 3
rng = np.random.default_rng(11)
X = rng.normal(size=(T, K)).cumsum(axis=0) + 100
y = rng.normal(size=T).cumsum() + 100
tgt = np.sign(rng.normal(size=(T, K)))


def nd(a):
    a = np.asarray(a, dtype=float)
    return {"shape": list(a.shape), "data": [None if np.isnan(v) else float(v) for v in a.ravel()]}


CASES = [
    ("RollingMean", [5], [X]),
    ("RollingMean", [5], [X[:, :, None]]),
    ("BollingerBands", [5], [X]),
    ("RollingCorr", [10], [X, y]),
    ("RollingCorr", [10], [X, y[:, None]]),
    ("Sub", [], [np.stack([X, y[:, None] + 0 * X], axis=-1)]),
    ("BacktestPriceTarget", [], [tgt, y]),
]


def main():
    out = []
    for name, args, inputs in CASES:
        result = getattr(s, name)(*args)(*inputs)
        out.append({"name": name, "args": args, "inputs": [nd(a) for a in inputs], "expect": nd(result)})
    with open(OUT, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
```

Run: `python devtools/wasm/gen_shape_oracle.py`. Commit the JSON.

- [ ] **Step 4: The JS contract test**

`js/test/shape_contract.test.ts`:

```ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { ready } from "../src/index.ts";
import * as api from "../src/index.ts";
import type { NdArray } from "../src/index.ts";

const HERE = dirname(fileURLToPath(import.meta.url));
type Nd = { shape: number[]; data: Array<number | null> };
const cases: Array<{ name: string; args: number[]; inputs: Nd[]; expect: Nd }> =
  JSON.parse(readFileSync(join(HERE, "..", "..", "wasm", "smoke", "shape_oracle.json"), "utf8"));

const toNd = (x: Nd): NdArray => ({ shape: x.shape, data: Float64Array.from(x.data, (v) => (v === null ? NaN : v)) });

for (const c of cases) {
  test(`shape contract: ${c.name} on ${c.inputs.map((i) => `[${i.shape}]`).join(", ")}`, async () => {
    await ready();
    const op = (api as any)[c.name](...c.args);
    const got = op(...c.inputs.map((i) => (i.shape.length === 1 ? toNd(i).data : toNd(i)))) as NdArray | Float64Array;
    const data = got instanceof Float64Array ? got : got.data;
    if (!(got instanceof Float64Array)) assert.deepEqual(got.shape, c.expect.shape);
    const want = toNd(c.expect).data;
    assert.equal(data.length, want.length);
    for (let i = 0; i < want.length; i++) {
      if (Number.isNaN(want[i])) assert.ok(Number.isNaN(data[i]), `index ${i}`);
      else assert.ok(Math.abs(data[i] - want[i]) <= 1e-9, `index ${i}: ${data[i]} vs ${want[i]}`);
    }
  });
}
```

- [ ] **Step 5: Run both suites in full**

Run: `pytest -q && cd js && npm run build:wasm && npm test`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_shape_contract.py devtools/wasm/gen_shape_oracle.py wasm/smoke/shape_oracle.json js/test/shape_contract.test.ts
git commit -m "test: shape contract for every operator, with a JS golden file"
```

---

### Task 10: Docs and changelog

**Files:**
- Modify: `docs/polymorphic_api.md`, `docs/multistream.md`
- Create: `docs/functions_streams/reshape.md`
- Modify: `docs/recipes/multi-asset.md`, `docs/javascript.md`, `docs/portfolio_backtesting.md` (only where they broadcast by hand or split with `Select`)
- Modify: `CHANGELOG.md` (Unreleased)

- [ ] **Step 1: `docs/polymorphic_api.md`**

Add a section "Time, assets and fields" near the top, after the dispatch rule. It states:
- the contract `(T, *A, F?)` with one sentence per letter;
- the six rules as a numbered list, each with a one-line example (`RollingCorr(20)(X, y)` with `X (T, K)` and `y (T,)` gives `(T, K)`; `(T, K, 1)` reads as `(T, K)`; `(T, 1)` stays one asset);
- the two-place table of defaults for unlabeled 2-D data, copied from the spec;
- a short `layout` paragraph pointing to the `reshape` page.

Write it in the style guide's voice: lead with the point, introduce terms before using them, no em-dashes.

- [ ] **Step 2: `docs/multistream.md`**

In the `CombineLatest` section, replace the "one-column streams" restriction and the `Select` workaround with:
- a bar stream and a signal give `(T, 5)`;
- `layout=("TA", "T")` broadcasts a market series over a panel;
- the `emit="on_any"` limit inside a `Pipeline` (decision 3).

In the `Resample` section, add the per-column rule with `(T, X)` giving `(T', X, 4)` for `"ohlc"`.

- [ ] **Step 3: `docs/functions_streams/reshape.md`**

Copy the frontmatter structure of `docs/functions_streams/from_pandas.md`:

```yaml
---
name: reshape
title: reshape
kind: function
short: Name the axes of data with a layout string (T, A, F) and reshape it to match.
tags:
- reshape
- layout
- axes
- assets
- fields
- broadcast
see_also:
- from_pandas
- CombineLatest
topics:
- streams
covers:
- reshape
---
```

The body:
- one paragraph on what a layout is;
- the five examples from the spec as an executed code cell;
- the container table from the spec;
- the JS form `reshape(x, "TA")`.

Then run `python devtools/build_help_registry.py` (or the make target that regenerates `screamer/data/help.json`, as the other function pages do) and `python devtools/build_topic_pages.py`.

- [ ] **Step 4: Recipes and guides**

Run `grep -rn "broadcast_to\|np.repeat\|Select(k)\|\*ohlc.T" docs/*.md docs/recipes docs/notebooks/*.ipynb`. Change each hit that existed only to work around a missing broadcast or the one-column `CombineLatest` to the direct call. Leave a `Select` that picks a column for its own sake.

- [ ] **Step 5: Changelog**

Under `## Unreleased` in `CHANGELOG.md`:

```markdown
### Added
- Shape contract: every array reads as `(T, *A, F?)`, with time, assets and fields (see Polymorphic API).
- Multi-input operators broadcast over assets: `RollingCorr()(X, y)` with `X (T, K)` and `y (T,)`, and backtest engines with `(T, K)` targets and one `(T,)` price, in Python and JavaScript.
- A packed `(T, K, N)` array feeds an N-input operator.
- Iterators of asset events give the batch result, one operator state per asset (Python and JavaScript).
- `reshape(x, layout)` for arrays, `(values, index)` tuples, DataFrames, lists and iterators of events; `reshape` in JavaScript; `from_pandas(layout=)`.
- `CombineLatest` concatenates multi-column streams and broadcasts assets with `layout=`; inside a `Pipeline` it joins an OHLC node and a signal directly.
- `Resample` aggregates every column of `(T, X)` data; `agg="ohlc"` gives `(T', X, 4)`.
- JavaScript one-input operators accept `(T, K)` `NdArray` input.

### Changed
- An explicit trailing size-1 axis is consumed: `(T, K, 1)` input gives `(T, K)` output (was `(T, K, 1)`), and `(T, K, M)` for M-output operators (was `(T, K, 1, M)`).
- A shape mismatch between inputs now raises `ValueError` ("do not broadcast", "same length along time") instead of `TypeError`.
- `from_pandas` reads a DataFrame with two-level `(asset, field)` columns as `(T, A, F)` values.
- `position_test`, `strategy_test` and `walk_forward` no longer copy prices to K columns; JavaScript `positionTest` and `walkForward` run the engine once per chunk.
```

- [ ] **Step 6: Build the docs**

Run: `make docs`
Expected: a clean build with no new warnings. Every executed cell in the new and edited pages runs.

- [ ] **Step 7: Commit**

```bash
git add docs CHANGELOG.md screamer/data/help.json
git commit -m "docs: shape contract, reshape, multi-column CombineLatest and Resample"
```

---

## Final verification (after Task 10)

- [ ] `pytest -q`: every test passes.
- [ ] `cd js && npm run build:wasm && npm test`: every test passes.
- [ ] `make docs`: clean.
- [ ] The final report lists the deliberate phase-1 limits:
  - JS `combineLatest` `layout` comes with phase 2 (decision 6);
  - `Resample` with a functor `agg` takes one column (decision 5);
  - `emit="on_any"` with a wide `Input` in a `Pipeline` raises (decision 3);
  - async iterators in Python keep scalar events.
