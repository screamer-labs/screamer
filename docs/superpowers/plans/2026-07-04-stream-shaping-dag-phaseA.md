# Stream Shaping in the DAG — Phase A (`dropna` + `select`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `dropna` (drop NaN rows) and a new `select` (pick columns from a wide stream) usable as pure-C++ push-nodes inside a compiled `Dag`, plus eager + `_iter` forms for `select`.

**Architecture:** Two new `dag::Sink` push-nodes (`DropNaNode`, `SelectNode`) mirroring `FunctorNode`; two new `NodeKind`s wired by `CompiledGraph`; `add_dropna`/`add_select` on the builder + bindings; Python dispatch via the existing `is_node` → `make_combinator_node` pattern. `filter` with a Python predicate is rejected in graphs (no-lambda).

**Tech Stack:** C++17, pybind11, CMake (auto-globs `include/screamer/dag/*.h`, `bindings/*.cpp`, `src/**`), Python 3.11, pytest, numpy.

## Global Constraints

- **Causality:** all ops causal, no bfill/lookahead. Batch and streaming must give byte-identical results (guarded by the identity harness).
- **No Python in the C++ engine:** graph nodes are C++-only. `filter`'s Python predicate must NOT enter the graph.
- **NaN test uses the house helper** `screamer::isnan2(double)` from `include/screamer/common/float_info.h` (not `std::isnan`), matching every other functor.
- **Zero per-event allocation** in push-nodes: reuse buffers; `DropNaNode` forwards the incoming frame pointer unchanged when a row survives.
- **Build:** after any C++ change run `make install-dev` (not just `make build`) or Python imports a stale binding. Then run tests with `poetry run pytest`.
- **Do NOT** edit version files, `screamer/__init__.py`, or run any `make patch/minor/major`.
- **A "stream"** is `(keys_int64, values_float64)`; in a `Dag` a stream is a single `Node`. Eager single-stream ops take `(keys, values, …)`; the graph form takes one `Node` (detected by `is_node` on the first argument).

---

### Task 1: Eager `select` + `select_iter`

**Files:**
- Modify: `screamer/streams.py` (add `select`, `select_iter`)
- Test: `tests/test_streams_select.py` (create)

**Interfaces:**
- Produces: `select(keys, values, columns)` → `(keys, values_selected)`; `select_iter(events, columns)` generator. `columns` is an `int` or a sequence of `int` (non-negative). A scalar `int` yields a 1-D `values`; a list yields 2-D with columns in the given order. Keys and row count unchanged.

- [ ] **Step 1: Write the failing test**

Create `tests/test_streams_select.py`:

```python
import numpy as np
import pytest

from screamer.streams import select, select_iter


def _wide():
    keys = np.array([1, 2, 3], dtype=np.int64)
    values = np.array([[10.0, 11.0, 12.0],
                       [20.0, 21.0, 22.0],
                       [30.0, 31.0, 32.0]])
    return keys, values


def test_select_single_int_returns_1d():
    keys, values = _wide()
    k, v = select(keys, values, 1)
    np.testing.assert_array_equal(k, keys)
    assert v.ndim == 1
    np.testing.assert_array_equal(v, [11.0, 21.0, 31.0])


def test_select_list_preserves_order():
    keys, values = _wide()
    k, v = select(keys, values, [2, 0])
    assert v.shape == (3, 2)
    np.testing.assert_array_equal(v, [[12.0, 10.0], [22.0, 20.0], [32.0, 30.0]])
    np.testing.assert_array_equal(k, keys)


def test_select_out_of_range_raises():
    keys, values = _wide()
    with pytest.raises((ValueError, IndexError)):
        select(keys, values, 5)


def test_select_negative_index_raises():
    keys, values = _wide()
    with pytest.raises(ValueError):
        select(keys, values, -1)


def test_select_iter_matches_batch():
    keys, values = _wide()
    events = list(zip(keys.tolist(), values.tolist()))
    got = list(select_iter(events, [0, 2]))
    assert [k for k, _ in got] == [1, 2, 3]
    assert [list(v) for _, v in got] == [[10.0, 12.0], [20.0, 22.0], [30.0, 32.0]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_streams_select.py -q`
Expected: FAIL with `ImportError: cannot import name 'select'`.

- [ ] **Step 3: Implement `select` and `select_iter`**

In `screamer/streams.py`, add after the `split` function. First a small shared normalizer, then the two functions:

```python
def _normalize_columns(columns):
    """Validate columns (int or non-negative int sequence) -> (list_of_ints, scalar).

    Returns (cols, is_scalar). is_scalar is True when a bare int was given (the
    eager result is then 1-D). Negative indices are rejected explicitly.
    """
    scalar = np.ndim(columns) == 0
    cols = [int(columns)] if scalar else [int(c) for c in columns]
    for c in cols:
        if c < 0:
            raise ValueError(f"select: column index must be non-negative, got {c}")
    return cols, scalar


def select(keys, values, columns):
    """Pick column(s) from a wide (M, N) value stream.

    columns is an int (result is 1-D) or a sequence of ints (result is 2-D with
    those columns in order). Keys and row count are unchanged (shape op, not
    cardinality). Indices must be in range and non-negative.
    """
    keys = np.asarray(keys)
    values = np.asarray(values, dtype=np.float64)
    cols, scalar = _normalize_columns(columns)
    if values.ndim == 1:
        width = 1
    else:
        width = values.shape[1]
    for c in cols:
        if c >= width:
            raise ValueError(
                f"select: column {c} out of range for width {width}")
    if values.ndim == 1:
        # width 1: only column 0 is valid; result mirrors input
        picked = values if scalar else values.reshape(-1, 1)
    else:
        picked = values[:, cols[0]] if scalar else values[:, cols]
    return keys, picked


def select_iter(events, columns):
    """Streaming select over (key, value) tuples. value is scalar or sequence."""
    cols, scalar = _normalize_columns(columns)
    for key, value in events:
        arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
        for c in cols:
            if c >= arr.size:
                raise ValueError(
                    f"select_iter: column {c} out of range for width {arr.size}")
        if scalar:
            yield key, float(arr[cols[0]])
        else:
            yield key, [float(arr[c]) for c in cols]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_streams_select.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add screamer/streams.py tests/test_streams_select.py
git commit -m "feat(streams): add eager select + select_iter (column projection)"
```

---

### Task 2: `dropna` as a C++ graph node (+ `filter` graph guard)

**Files:**
- Create: `include/screamer/dag/dropna_node.h`
- Modify: `include/screamer/dag/graph.h` (NodeKind, NodeSpec fields, `add_dropna`)
- Modify: `include/screamer/dag/compiled_graph.h` (include, node-width computation, wiring case)
- Modify: `bindings/bindings_dag.cpp` (`PyGraphBuilder::add_dropna` + binding)
- Modify: `screamer/dag.py` (dispatch `"dropna"`)
- Modify: `screamer/streams.py` (`dropna`/`filter` detect `Node`)
- Test: `tests/test_dag_dropna.py` (create)

**Interfaces:**
- Consumes: `dag::Frame`/`Sink` (`frame.h`), `screamer::isnan2` (`float_info.h`), the `is_node`/`make_combinator_node` pattern (`dag.py`).
- Produces:
  - C++: `dag::DropNaNode<Key>(bool how_all, Sink<Key>& downstream)`.
  - `GraphBuilder::add_dropna(std::vector<std::size_t> inputs, bool how_all) -> std::size_t`.
  - `NodeKind::DropNa`; `NodeSpec.how_all` (bool).
  - `_GraphBuilder.add_dropna(inputs, how_all)` (binding).
  - Python: `dropna(stream, how="any")` returns a `Node` when `stream` is a `Node`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dag_dropna.py`:

```python
import numpy as np
import pytest

from screamer import Input, Dag
from screamer.streams import dropna


def _run_modes(dag, feed):
    """Return (batch, stream) results as (keys, values) for a single-output dag."""
    bk, bv = dag(feed)
    sk, sv = dag.stream(feed)
    return (bk, bv), (sk, sv)


def test_dropna_graph_matches_eager_any():
    keys = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    vals = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[dropna(x)])
    (bk, bv), (sk, sv) = _run_modes(dag, (keys, vals))
    ek, ev = dropna(keys, vals)          # eager oracle
    np.testing.assert_array_equal(bk, ek)
    np.testing.assert_array_equal(bv.reshape(-1), ev)
    np.testing.assert_array_equal(sk, ek)
    np.testing.assert_array_equal(sv.reshape(-1), ev)


def test_dropna_graph_all_dropped():
    keys = np.array([1, 2], dtype=np.int64)
    vals = np.array([np.nan, np.nan])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[dropna(x)])
    bk, bv = dag((keys, vals))
    assert len(bk) == 0


def test_dropna_graph_none_dropped():
    keys = np.array([1, 2, 3], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[dropna(x)])
    bk, bv = dag((keys, vals))
    np.testing.assert_array_equal(bk, keys)
    np.testing.assert_array_equal(bv.reshape(-1), vals)


def test_dropna_before_functor():
    from screamer import RollingMean
    keys = np.array([1, 2, 3, 4], dtype=np.int64)
    vals = np.array([2.0, np.nan, 4.0, 6.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[RollingMean(2)(dropna(x))])
    bk, bv = dag((keys, vals))
    sk, sv = dag.stream((keys, vals))
    np.testing.assert_array_equal(bk, sk)
    np.testing.assert_array_equal(bv, sv)
    # dropna removes the NaN row, leaving keys [1,3,4]; RollingMean(2) over [2,4,6]
    np.testing.assert_array_equal(bk, [1, 3, 4])


def test_filter_rejected_in_graph():
    x = Input("x")
    from screamer.streams import filter as sfilter
    with pytest.raises(ValueError, match="not supported"):
        sfilter(x, lambda r: r > 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_dag_dropna.py -q`
Expected: FAIL — `dropna(x)` on a `Node` currently runs the eager path and raises (or the `Dag` rejects the combinator).

- [ ] **Step 3a: Add the `DropNaNode` push-node**

Create `include/screamer/dag/dropna_node.h`:

```cpp
#ifndef SCREAMER_DAG_DROPNA_NODE_H
#define SCREAMER_DAG_DROPNA_NODE_H

#include <cstddef>
#include "screamer/common/float_info.h"
#include "screamer/dag/frame.h"

namespace screamer { namespace dag {

// Drops events whose values are NaN. how_all=false ("any"): drop if any value
// is NaN. how_all=true ("all"): drop only if every value is NaN (an empty frame
// is never dropped). Cardinality-reducing; forwards the surviving frame pointer
// unchanged (zero per-event allocation).
template <class Key>
class DropNaNode : public Sink<Key> {
public:
    DropNaNode(bool how_all, Sink<Key>& downstream)
        : how_all_(how_all), downstream_(downstream) {}

    void push(const Frame<Key>& f) override {
        bool any_nan = false;
        bool all_nan = f.width > 0;
        for (std::size_t i = 0; i < f.width; ++i) {
            if (screamer::isnan2(f.values[i])) any_nan = true;
            else                               all_nan = false;
        }
        bool drop = how_all_ ? all_nan : any_nan;
        if (!drop) downstream_.push(f);
    }

    void flush() override { downstream_.flush(); }

private:
    bool how_all_;
    Sink<Key>& downstream_;
};

}} // namespace screamer::dag
#endif
```

- [ ] **Step 3b: Extend the graph spec + builder**

In `include/screamer/dag/graph.h`:

Add `DropNa` to the enum:
```cpp
enum class NodeKind { Input, Functor, CombineLatest, DropNa };
```

Add a `how_all` field to `NodeSpec` (after `when_all`):
```cpp
struct NodeSpec {
    NodeKind kind;
    EvalOp* op = nullptr;                 // Functor only
    bool when_all = true;                 // CombineLatest only
    bool how_all = false;                 // DropNa only
    std::vector<std::size_t> inputs;      // producer node ids (edges into this node)
};
```

Add the builder method (after `add_combine_latest`):
```cpp
std::size_t add_dropna(std::vector<std::size_t> inputs, bool how_all) {
    spec_.nodes.push_back(NodeSpec{NodeKind::DropNa, nullptr, true, how_all,
                                   std::move(inputs)});
    return spec_.nodes.size() - 1;
}
```
NOTE: the existing `add_functor`/`add_combine_latest`/`add_input` push `NodeSpec{...}` with 4 positional fields (`kind, op, when_all, inputs`). Adding the `how_all` field means those three initializers need the extra field. Update them to include `how_all` (`false`) in position:
```cpp
std::size_t add_input() {
    spec_.nodes.push_back(NodeSpec{NodeKind::Input, nullptr, true, false, {}});
    std::size_t id = spec_.nodes.size() - 1;
    spec_.input_ids.push_back(id);
    return id;
}
std::size_t add_functor(EvalOp* op, std::vector<std::size_t> inputs) {
    spec_.nodes.push_back(NodeSpec{NodeKind::Functor, op, true, false, std::move(inputs)});
    return spec_.nodes.size() - 1;
}
std::size_t add_combine_latest(std::vector<std::size_t> inputs, bool when_all) {
    spec_.nodes.push_back(NodeSpec{NodeKind::CombineLatest, nullptr, when_all, false,
                                   std::move(inputs)});
    return spec_.nodes.size() - 1;
}
```

- [ ] **Step 3c: Wire `DropNa` in the compiler + refactor node-width**

In `include/screamer/dag/compiled_graph.h`:

Add the include near the other dag includes:
```cpp
#include "screamer/dag/dropna_node.h"
```

Replace the early per-output width switch (the block computing `output_widths_` at lines ~72–91, BEFORE the topo sort) with a single computation AFTER the topo sort, so producer widths are known when a pass-through node (DropNa) needs its input's width. Concretely:

1. DELETE the existing `output_widths_.resize(num_out); for (...) { switch(node.kind) ... }` block near the top of the constructor.

2. AFTER `std::reverse(topo.begin(), topo.end());` and the cycle check, insert a full node-width pass (producers-first = reverse of the consumers-first `topo`):
```cpp
// Width of every node's emitted frame. Producers-first (reverse of the
// consumers-first topo) so a pass-through node can read its input's width.
std::vector<std::size_t> node_width(n, 1);
for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    std::size_t id = *it;
    const auto& nd = s.nodes[id];
    switch (nd.kind) {
    case NodeKind::Input:         node_width[id] = 1; break;
    case NodeKind::Functor:       node_width[id] = nd.op->n_out(); break;
    case NodeKind::CombineLatest: node_width[id] = nd.inputs.size(); break;
    case NodeKind::DropNa:        node_width[id] = node_width[nd.inputs[0]]; break;
    }
}
output_widths_.resize(num_out);
for (std::size_t o = 0; o < num_out; ++o)
    output_widths_[o] = node_width[s.output_ids[o]];
```

3. In the wiring `switch (ns.kind)` (consumers-first loop), add the `DropNa` case (alongside `Functor`/`CombineLatest`). DropNa is stateless (no reset registration) and single-input (returns its own sink for any slot, like `Functor`):
```cpp
case NodeKind::DropNa: {
    auto dn = std::make_shared<DropNaNode<std::int64_t>>(ns.how_all, *downstream);
    node_input_sink[id] = [ptr = dn.get()](std::size_t) -> Sink<std::int64_t>* {
        return ptr;
    };
    owned_.push_back(dn);
    break;
}
```

- [ ] **Step 3d: Bind `add_dropna`**

In `bindings/bindings_dag.cpp`, in `struct PyGraphBuilder` (after `add_combine_latest`):
```cpp
std::size_t add_dropna(std::vector<std::size_t> inputs, bool how_all) {
    return builder.add_dropna(std::move(inputs), how_all);
}
```
And in the `py::class_<PyGraphBuilder>` definitions (after the `add_combine_latest` `.def`):
```cpp
.def("add_dropna", [](PyGraphBuilder& b,
                      std::vector<std::size_t> inputs, bool how_all) {
    return b.add_dropna(std::move(inputs), how_all);
}, py::arg("inputs"), py::arg("how_all") = false)
```

- [ ] **Step 3e: Dispatch `dropna` (and reject `filter`) in Python**

In `screamer/dag.py`, in `_compile_cpp`'s `build()`, replace the combinator branch:
```cpp
            elif isinstance(op, tuple) and op[0] == "combinator":
                fn, kwargs = op[1], op[2]
                if getattr(fn, "__name__", "") != "combine_latest":
                    raise ValueError(
                        f"{fn.__name__} is not supported as a DAG graph node")
                inp = [build(i) for i in node.inputs]
                nid = gb.add_combine_latest(inp, kwargs.get("emit") == "when_all")
```
with:
```python
            elif isinstance(op, tuple) and op[0] == "combinator":
                fn, kwargs = op[1], op[2]
                name = getattr(fn, "__name__", "")
                inp = [build(i) for i in node.inputs]
                if name == "combine_latest":
                    nid = gb.add_combine_latest(inp, kwargs.get("emit") == "when_all")
                elif name == "dropna":
                    nid = gb.add_dropna(inp, kwargs.get("how") == "all")
                else:
                    raise ValueError(
                        f"{name} is not supported as a DAG graph node")
```

In `screamer/streams.py`, make `dropna` detect a `Node` first argument (add at the very top of the function body, before the `how` validation):
```python
def dropna(keys, values=None, how="any"):
    if is_node(keys):
        return make_combinator_node(dropna, (keys,), {"how": how})
    ...  # existing eager body unchanged (keys, values arrays)
```
And make `filter` reject a `Node` (add at the top of `filter`):
```python
def filter(keys, values, predicate=None):
    if is_node(keys):
        raise ValueError(
            "filter is not supported as a DAG graph node: the graph engine has "
            "no Python predicates (no lambda). Use dropna for NaN removal.")
    ...  # existing eager body unchanged
```
(Adding `values=None`/`predicate=None` defaults keeps eager callers working — they always pass the second argument — while allowing the single-`Node` graph call form.)

- [ ] **Step 4: Build and run the tests**

```bash
make install-dev
poetry run pytest tests/test_dag_dropna.py -q
```
Expected: PASS (5 passed).

- [ ] **Step 5: Regression + commit**

```bash
poetry run pytest tests/test_streams_dropna.py tests/test_dag_identity.py -q
git add include/screamer/dag/dropna_node.h include/screamer/dag/graph.h include/screamer/dag/compiled_graph.h bindings/bindings_dag.cpp screamer/dag.py screamer/streams.py tests/test_dag_dropna.py
git commit -m "feat(dag): dropna as a C++ push-node; reject filter (no lambda)"
```
Expected: regressions pass.

---

### Task 3: `select` as a C++ graph node

**Files:**
- Create: `include/screamer/dag/select_node.h`
- Modify: `include/screamer/dag/graph.h` (NodeKind, NodeSpec `columns`, `add_select`)
- Modify: `include/screamer/dag/compiled_graph.h` (include, node-width case, wiring case)
- Modify: `bindings/bindings_dag.cpp` (`add_select` + binding)
- Modify: `screamer/dag.py` (dispatch `"select"`, `_normalize_columns` import/use)
- Modify: `screamer/streams.py` (`select` detects `Node`)
- Test: `tests/test_dag_select.py` (create)

**Interfaces:**
- Consumes: `dag::Frame`/`Sink`, the node-width + wiring pattern from Task 2, `_normalize_columns` from Task 1.
- Produces:
  - C++: `dag::SelectNode<Key>(std::vector<std::size_t> columns, Sink<Key>& downstream)`.
  - `GraphBuilder::add_select(std::vector<std::size_t> inputs, std::vector<std::size_t> columns) -> std::size_t`.
  - `NodeKind::Select`; `NodeSpec.columns`.
  - `_GraphBuilder.add_select(inputs, columns)` (binding).
  - Python: `select(stream, columns)` returns a `Node` when `stream` is a `Node`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dag_select.py`:

```python
import numpy as np
import pytest

from screamer import Input, Dag, RollingMean
from screamer.streams import select, combine_latest


def test_select_column_from_combine_latest():
    ak = np.array([1, 2, 3], dtype=np.int64); av = np.array([10.0, 20.0, 30.0])
    bk = np.array([1, 2, 3], dtype=np.int64); bv = np.array([1.0, 2.0, 3.0])
    a, b = Input("a"), Input("b")
    # combine_latest(a, b) is width-2; select column 0 -> a's latest
    dag = Dag(inputs=[a, b], outputs=[select(combine_latest(a, b), 0)])
    bk_, bv_ = dag((ak, av), (bk, bv))
    sk_, sv_ = dag.stream((ak, av), (bk, bv))
    np.testing.assert_array_equal(bk_, sk_)
    np.testing.assert_array_equal(bv_, sv_)
    # column 0 tracks a's value at each aligned row
    np.testing.assert_array_equal(bv_.reshape(-1), [10.0, 20.0, 30.0])


def test_select_two_columns_reorder():
    ak = np.array([1, 2], dtype=np.int64); av = np.array([10.0, 20.0])
    bk = np.array([1, 2], dtype=np.int64); bv = np.array([1.0, 2.0])
    a, b = Input("a"), Input("b")
    dag = Dag(inputs=[a, b], outputs=[select(combine_latest(a, b), [1, 0])])
    bk_, bv_ = dag((ak, av), (bk, bv))
    sk_, sv_ = dag.stream((ak, av), (bk, bv))
    np.testing.assert_array_equal(bv_, sv_)
    # [1,0] swaps -> column0 = b, column1 = a
    np.testing.assert_array_equal(bv_, [[1.0, 10.0], [2.0, 20.0]])


def test_select_feeds_functor():
    ak = np.array([1, 2, 3], dtype=np.int64); av = np.array([2.0, 4.0, 6.0])
    bk = np.array([1, 2, 3], dtype=np.int64); bv = np.array([0.0, 0.0, 0.0])
    a, b = Input("a"), Input("b")
    # select a's column then smooth it
    dag = Dag(inputs=[a, b], outputs=[RollingMean(2)(select(combine_latest(a, b), 0))])
    bk_, bv_ = dag((ak, av), (bk, bv))
    sk_, sv_ = dag.stream((ak, av), (bk, bv))
    np.testing.assert_array_equal(bk_, sk_)
    np.testing.assert_array_equal(bv_, sv_)


def test_select_out_of_range_errors():
    ak = np.array([1], dtype=np.int64); av = np.array([10.0])
    bk = np.array([1], dtype=np.int64); bv = np.array([1.0])
    a, b = Input("a"), Input("b")
    dag = Dag(inputs=[a, b], outputs=[select(combine_latest(a, b), 5)])
    with pytest.raises(Exception):
        dag((ak, av), (bk, bv))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_dag_select.py -q`
Expected: FAIL — `select(node, ...)` runs the eager path / the `Dag` rejects `select`.

- [ ] **Step 3a: Add the `SelectNode` push-node**

Create `include/screamer/dag/select_node.h`:

```cpp
#ifndef SCREAMER_DAG_SELECT_NODE_H
#define SCREAMER_DAG_SELECT_NODE_H

#include <cstddef>
#include <stdexcept>
#include <vector>
#include "screamer/dag/frame.h"

namespace screamer { namespace dag {

// Projects selected columns out of a wide frame, emitting a width-M frame
// (M = columns.size()) with the columns in the given order. Keys pass through;
// row count unchanged. Reuses one output buffer (zero per-event allocation).
template <class Key>
class SelectNode : public Sink<Key> {
public:
    SelectNode(std::vector<std::size_t> columns, Sink<Key>& downstream)
        : columns_(std::move(columns)), downstream_(downstream),
          out_(columns_.size()) {}

    void push(const Frame<Key>& f) override {
        for (std::size_t j = 0; j < columns_.size(); ++j) {
            if (columns_[j] >= f.width)
                throw std::runtime_error(
                    "dag::SelectNode: column index out of range for frame width");
            out_[j] = f.values[columns_[j]];
        }
        downstream_.push(Frame<Key>{f.key, out_.data(), out_.size()});
    }

    void flush() override { downstream_.flush(); }

private:
    std::vector<std::size_t> columns_;
    Sink<Key>& downstream_;
    std::vector<double> out_;   // reused every event
};

}} // namespace screamer::dag
#endif
```

- [ ] **Step 3b: Extend the graph spec + builder**

In `include/screamer/dag/graph.h`:

Add `Select` to the enum:
```cpp
enum class NodeKind { Input, Functor, CombineLatest, DropNa, Select };
```

Add a `columns` field to `NodeSpec` (after `how_all`):
```cpp
    bool how_all = false;                 // DropNa only
    std::vector<std::size_t> columns;     // Select only
    std::vector<std::size_t> inputs;      // producer node ids (edges into this node)
```

Add the builder method (after `add_dropna`):
```cpp
std::size_t add_select(std::vector<std::size_t> inputs,
                       std::vector<std::size_t> columns) {
    NodeSpec ns{NodeKind::Select, nullptr, true, false, std::move(columns),
                std::move(inputs)};
    spec_.nodes.push_back(std::move(ns));
    return spec_.nodes.size() - 1;
}
```
NOTE: adding the `columns` field shifts the aggregate initializer positions. Update the other `add_*` methods' `NodeSpec{...}` initializers to include an empty `columns` (`{}`) in position, e.g.:
```cpp
spec_.nodes.push_back(NodeSpec{NodeKind::Input, nullptr, true, false, {}, {}});
spec_.nodes.push_back(NodeSpec{NodeKind::Functor, op, true, false, {}, std::move(inputs)});
spec_.nodes.push_back(NodeSpec{NodeKind::CombineLatest, nullptr, when_all, false, {}, std::move(inputs)});
spec_.nodes.push_back(NodeSpec{NodeKind::DropNa, nullptr, true, how_all, {}, std::move(inputs)});
```

- [ ] **Step 3c: Wire `Select` in the compiler**

In `include/screamer/dag/compiled_graph.h`:

Add the include:
```cpp
#include "screamer/dag/select_node.h"
```

Add the `Select` case to the node-width pass (its width is the number of columns):
```cpp
    case NodeKind::Select:        node_width[id] = nd.columns.size(); break;
```

Add the `Select` wiring case (stateless, single-input, like `DropNa`):
```cpp
case NodeKind::Select: {
    auto sn = std::make_shared<SelectNode<std::int64_t>>(ns.columns, *downstream);
    node_input_sink[id] = [ptr = sn.get()](std::size_t) -> Sink<std::int64_t>* {
        return ptr;
    };
    owned_.push_back(sn);
    break;
}
```

- [ ] **Step 3d: Bind `add_select`**

In `bindings/bindings_dag.cpp`, in `struct PyGraphBuilder` (after `add_dropna`):
```cpp
std::size_t add_select(std::vector<std::size_t> inputs,
                       std::vector<std::size_t> columns) {
    return builder.add_select(std::move(inputs), std::move(columns));
}
```
And in the `py::class_<PyGraphBuilder>` definitions:
```cpp
.def("add_select", [](PyGraphBuilder& b,
                      std::vector<std::size_t> inputs,
                      std::vector<std::size_t> columns) {
    return b.add_select(std::move(inputs), std::move(columns));
}, py::arg("inputs"), py::arg("columns"))
```

- [ ] **Step 3e: Dispatch `select` in Python**

In `screamer/dag.py`, add a `select` branch to `build()`'s combinator dispatch (after the `dropna` branch). Import `_normalize_columns` from streams at the top of the function or module:
```python
                elif name == "select":
                    from .streams import _normalize_columns
                    cols, _ = _normalize_columns(kwargs["columns"])
                    nid = gb.add_select(inp, cols)
```

In `screamer/streams.py`, make `select` detect a `Node` first argument (add at the top of the function body, before `np.asarray`):
```python
def select(keys, values=None, columns=None):
    if is_node(keys):
        # graph form: select(stream, columns) — columns may be the 2nd
        # positional (the `values` slot) or the `columns` keyword.
        cols = values if columns is None else columns
        if cols is None:
            raise ValueError("select: columns is required")
        return make_combinator_node(select, (keys,), {"columns": cols})
    if columns is None:
        raise ValueError("select: columns is required")
    ...  # existing eager body from Task 1, unchanged
```
(The eager body from Task 1 already validates via `_normalize_columns`; keep it. The `values=None`/`columns=None` defaults let the graph form `select(node, [0,2])` bind `[0,2]` into the `values` slot.)

- [ ] **Step 4: Build and run the tests**

```bash
make install-dev
poetry run pytest tests/test_dag_select.py tests/test_streams_select.py -q
```
Expected: PASS.

- [ ] **Step 5: Full identity regression + commit**

```bash
poetry run pytest tests/test_dag_identity.py tests/test_dag_dropna.py -q
git add include/screamer/dag/select_node.h include/screamer/dag/graph.h include/screamer/dag/compiled_graph.h bindings/bindings_dag.cpp screamer/dag.py screamer/streams.py tests/test_dag_select.py
git commit -m "feat(dag): select (column projection) as a C++ push-node"
```
Expected: identity matrix + dropna tests still green.

---

## Self-Review

**Spec coverage:** dropna-in-graph (Task 2), select eager + `_iter` (Task 1) + select-in-graph (Task 3), filter graph-rejection (Task 2), batch==stream identity (Tasks 2 & 3 tests). All spec items covered.

**Type consistency:** `add_dropna(inputs, how_all: bool)`, `add_select(inputs, columns: vector<size_t>)`, `DropNaNode(bool, Sink&)`, `SelectNode(vector<size_t>, Sink&)`, `NodeKind::{DropNa,Select}`, `NodeSpec.{how_all, columns}` used consistently across graph.h / compiled_graph.h / bindings / dag.py. The `NodeSpec` aggregate-initializer field order is called out explicitly in both Task 2 (add `how_all`) and Task 3 (add `columns`) so earlier `add_*` initializers are updated in lockstep.

**Placeholder scan:** none — every step carries full code.

**Cross-task note:** Task 2 refactors node-width computation to run after the topo sort (needed so `DropNa` can read its input's width); Task 3 only adds a `Select` case to that same pass. `filter`'s eager signature gains a `predicate=None` default (Task 2) but eager callers always pass it, so behavior is unchanged.
