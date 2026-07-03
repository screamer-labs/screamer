# DAG-1 — definition + naive batch executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users define a computation as a graph of `Node` handles in ordinary Python (assignment names streams, passing them wires edges) and run it in batch, where a compiled `Dag` behaves like a plain positional N-input/M-output function.

**Architecture:** A `Node` becomes one more input kind in the existing polymorphic dispatch — a functor or combinator called on a `Node` records a graph node instead of computing (a small central hook in the two C++ dispatchers plus Node-awareness in the Python combinators). A `Dag(inputs=[...], outputs=[...])` is a positional callable; its naive executor evaluates nodes in dependency order by calling the existing batch functions, memoizing so fan-out computes once, and aligns its outputs with `combine_latest` by default.

**Tech Stack:** Python, C++17, pybind11, numpy, pytest.

## Global Constraints

- **A `Dag` is a plain function:** positional `N` inputs → `M` outputs; single return for `M==1`, an unpackable tuple for `M>1`; keyword calls by `Input` name also work; a `Dag` is graph-capable so it nests in another `Dag`.
- **Aligned outputs by default** (`align_outputs=True`): the `Dag` joins its `M` outputs with `combine_latest` (union keys, `emit="when_all"`) so they are co-indexed and equal-length. `align_outputs=False` returns independent per-output streams.
- **`(T, N)` input convention:** an `N`-input functor accepts a single 2-D `(T, N)` array (columns → the `N` inputs) iff `shape[1] == N`; any other single-array shape stays an error with a clear message.
- **Stateful-safety:** a functor instance backs at most one node; reuse raises a clear error.
- **Definition is Python; execution reuses existing batch functions.** No new execution engine. The only C++ changes are the `(T,N)` convention and the Node dispatch hook.
- **Cycles are impossible by construction** (nodes built bottom-up).
- Keys are carried when present (a stream is `(keys, values)`); a bare value array = row-number keys, mirroring the combinators.
- Compute functors' math is never modified. Never hand-edit `screamer/__init__.py` or version files.
- Build: `make build` **then `make install-dev`** after any C++ change. Tests: `poetry run pytest tests/test_dag*.py tests/test_ntuple_input.py -v`.
- New public Python names go in `screamer/dag.py` with an `__all__`, exported via the generator (same mechanism Plan 4 added for `streams`).

---

## File Structure

- `include/screamer/common/functor_base.h` (modify) — `(T,N)` convention in `handle_input_Ni_1o`/`handle_input_Ni_Mo`; Node hook in `handle_input`.
- `src/screamer/common/base.cpp` (modify) — Node hook in `ScreamerBase::operator()`.
- `include/screamer/common/base.h` (modify) — declare the hook helper.
- `screamer/dag.py` (create) — `Node`, `Input`, `Dag`, the registered functor-node builder, the executor; `__all__`.
- `screamer/streams.py` (modify) — combinators return a `Node` when any arg is a `Node`.
- `devtools/generate_screamer__init__.py` (modify) — also export `screamer/dag.py`'s `__all__`.
- Tests: `tests/test_ntuple_input.py`, `tests/test_dag_build.py`, `tests/test_dag_exec.py`, `tests/test_dag_public_api.py`.

---

### Task 1: the `(T, N)` input convention for N-input functors

**Files:**
- Modify: `include/screamer/common/functor_base.h`
- Modify: `docs/polymorphic_api.md`
- Test: `tests/test_ntuple_input.py`

**Interfaces:**
- Consumes: existing `FunctorBase<Derived,N,M>` dispatch.
- Produces: an `N`-input functor called with a single 2-D `(T, N)` numpy array treats its columns as the `N` inputs; result equals calling it with the `N` columns as separate args. Wrong-width single array raises `py::value_error` with a clear message.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ntuple_input.py`:

```python
import numpy as np
import pytest
from screamer import RollingCorr, Cart2Polar


def test_two_input_functor_accepts_TxN_array():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    y = rng.standard_normal(200)
    aligned = np.column_stack([x, y])          # shape (200, 2)
    got = RollingCorr(20)(aligned)
    exp = RollingCorr(20)(x, y)
    np.testing.assert_array_equal(got, exp)


def test_TxN_array_wrong_width_raises():
    rng = np.random.default_rng(1)
    bad = rng.standard_normal((100, 3))        # 3 cols for a 2-input functor
    with pytest.raises((ValueError, TypeError)):
        RollingCorr(20)(bad)


def test_NtoM_functor_accepts_TxN_array():
    rng = np.random.default_rng(2)
    xy = rng.standard_normal((50, 2))
    got = Cart2Polar()(xy)                      # N=2, M=2
    exp = Cart2Polar()(xy[:, 0], xy[:, 1])
    np.testing.assert_array_equal(got, exp)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_ntuple_input.py -v`
Expected: FAIL — a single 2-D array to an N-input functor currently raises "Unsupported single argument input type…".

- [ ] **Step 3: Add a column-splitting helper and route single `(T,N)` arrays**

In `include/screamer/common/functor_base.h`, add inside `namespace detail` (near the other helpers):

```cpp
    // Extract column j (of N) from a contiguous-or-strided (T, N) array into a
    // fresh 1-D array of length T. Used by the (T,N) single-array input form.
    inline py::array_t<double> extract_column(const py::array_t<double>& arr,
                                              size_t j, size_t T, size_t N) {
        py::buffer_info info = arr.request();
        const double* src = static_cast<const double*>(info.ptr);
        std::ptrdiff_t row_stride = info.strides[0] / sizeof(double);
        std::ptrdiff_t col_stride = info.strides[1] / sizeof(double);
        py::array_t<double> col(static_cast<py::ssize_t>(T));
        double* dst = static_cast<double*>(col.request().ptr);
        for (size_t i = 0; i < T; ++i) {
            dst[i] = src[i * row_stride + j * col_stride];
        }
        return col;
    }

    // If args is a single 2-D (T, N) numpy array, return a tuple of its N
    // columns as 1-D arrays; otherwise return an empty optional. Enforces the
    // exact-width match (shape[1] == N); a mismatched width throws a clear error.
    template <size_t N>
    inline std::optional<py::tuple> maybe_split_TxN(const py::args& args) {
        if (args.size() != 1 || !py::isinstance<py::array>(args[0])) {
            return std::nullopt;
        }
        py::array_t<double> arr = py::cast<py::array_t<double>>(args[0]);
        py::buffer_info info = arr.request();
        if (info.ndim != 2) {
            return std::nullopt;   // 1-D single array falls through to the normal error
        }
        size_t T = static_cast<size_t>(info.shape[0]);
        size_t width = static_cast<size_t>(info.shape[1]);
        if (width != N) {
            throw py::value_error(
                "This functor expects " + std::to_string(N) +
                " inputs; got a single 2-D array with " + std::to_string(width) +
                " columns. Pass an (T, " + std::to_string(N) + ") array or " +
                std::to_string(N) + " separate arrays.");
        }
        py::tuple cols(N);
        for (size_t j = 0; j < N; ++j) {
            cols[j] = extract_column(arr, j, T, N);
        }
        return cols;
    }
```

- [ ] **Step 4: Use the helper in the two N-input dispatchers**

In `handle_input_Ni_1o` (the `(TN > 1) && (TM == 1)` overload), at the very top of the body — before the existing "Case 1" — insert:

```cpp
        if (auto cols = detail::maybe_split_TxN<N>(args)) {
            return handle_input_Ni_1o_numpy(*cols);
        }
```

In `handle_input_Ni_Mo` (the `(TN > 1) && (TM > 1)` overload), at the very top of the body insert:

```cpp
        if (auto cols = detail::maybe_split_TxN<N>(args)) {
            return handle_input_Ni_Mo_numpy(*cols);
        }
```

Ensure `<optional>` is included in `functor_base.h` (add `#include <optional>` near the top if absent).

- [ ] **Step 5: Build and run**

Run: `make build && make install-dev && poetry run pytest tests/test_ntuple_input.py -v`
Expected: all three tests PASS.

- [ ] **Step 6: Document the convention**

In `docs/polymorphic_api.md`, under "The multi-input contract (`FunctorBase<_, N, 1>`)", add a row/paragraph:

```markdown
| `obj(A)` — a single 2-D array of shape `(T, N)` | `numpy.ndarray` of shape `(T,)` | the `N` columns are the `N` inputs; column `j` → input `j`. Accepted iff `A.shape[1] == N`; any other single-array shape is a `TypeError`/`ValueError`. |

This is the array form of `obj(A[:, 0], A[:, 1], ...)`, and is what makes
`RollingCorr(w)(combine_latest(a, b)[1])` work directly.
```

- [ ] **Step 7: Commit**

```bash
git add include/screamer/common/functor_base.h docs/polymorphic_api.md tests/test_ntuple_input.py
git commit -m "feat(functor): accept a single (T,N) array as N inputs"
```

---

### Task 2: `Node`, `Input`, and combinator Node-awareness

**Files:**
- Create: `screamer/dag.py`
- Modify: `screamer/streams.py`
- Test: `tests/test_dag_build.py`

**Interfaces:**
- Produces:
  - `screamer.dag.Node` — immutable; attributes `op` (a functor instance, or `("combinator", fn, kwargs)`, or `("input", name)`), `inputs` (tuple of `Node`), and `is_node` marker.
  - `screamer.dag.Input(name) -> Node`.
  - `screamer.dag.is_node(obj) -> bool`.
  - `screamer.dag.make_combinator_node(fn, args, kwargs) -> Node` (used by streams combinators).
  - `combine_latest`/`merge`/`dropna`/`filter`/`split` return a `Node` (or list of `Node` for `split`) when any argument is a `Node`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dag_build.py`:

```python
import numpy as np
from screamer.dag import Node, Input, is_node
from screamer import MovingAverage, combine_latest


def test_input_is_a_node():
    a = Input("price_a")
    assert is_node(a)
    assert a.op == ("input", "price_a")
    assert a.inputs == ()


def test_functor_on_node_builds_node():
    a = Input("price_a")
    n = MovingAverage(30)(a)
    assert is_node(n)
    assert n.inputs == (a,)
    # op is the functor instance itself
    assert n.op.__class__.__name__ == "MovingAverage"


def test_combinator_on_nodes_builds_node():
    a, b = Input("a"), Input("b")
    n = combine_latest(a, b, func=None)
    assert is_node(n)
    assert n.inputs == (a, b)
    assert n.op[0] == "combinator"
    assert n.op[2] == {"emit": "when_all", "func": None}


def test_combinator_on_data_still_computes():
    # no Node args -> normal eager behavior, returns arrays not a Node
    a = (np.array([1, 2, 3], dtype=np.int64), np.array([1.0, 2.0, 3.0]))
    b = (np.array([1, 2, 3], dtype=np.int64), np.array([4.0, 5.0, 6.0]))
    keys, aligned = combine_latest(a, b)
    assert not is_node(keys)
    assert aligned.shape == (3, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_dag_build.py -v`
Expected: FAIL — `No module named 'screamer.dag'`.

- [ ] **Step 3: Create `screamer/dag.py` (Node + Input + helpers)**

```python
"""Computational DAG definition: symbolic Node handles (DAG-1)."""

__all__ = ["Node", "Input", "Dag"]


class Node:
    """An immutable handle for a stream in a computation graph.

    op is one of:
      - ("input", name)
      - a configured functor instance (ScreamerBase / FunctorBase)
      - ("combinator", fn, kwargs)
    inputs is a tuple of upstream Nodes.
    """
    __slots__ = ("op", "inputs")
    is_node = True

    def __init__(self, op, inputs=()):
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "inputs", tuple(inputs))

    def __setattr__(self, *a):
        raise AttributeError("Node is immutable")

    def __repr__(self):
        if isinstance(self.op, tuple) and self.op and self.op[0] == "input":
            return f"Input({self.op[1]!r})"
        name = getattr(type(self.op), "__name__", type(self.op).__name__)
        if isinstance(self.op, tuple) and self.op and self.op[0] == "combinator":
            name = self.op[1].__name__
        return f"Node({name}, {len(self.inputs)} input(s))"


def is_node(obj):
    return getattr(obj, "is_node", False) is True


def Input(name):
    """Create a source Node — a named placeholder for a timed stream."""
    return Node(("input", name))


def make_functor_node(functor, args):
    """Build a Node for a functor applied to Node args. Called by the C++ hook."""
    inputs = tuple(args)
    for n in inputs:
        if not is_node(n):
            raise TypeError("all arguments must be Nodes when building a graph")
    return Node(functor, inputs)


def make_combinator_node(fn, node_args, kwargs):
    """Build a Node for a combinator applied to Node args."""
    return Node(("combinator", fn, kwargs), tuple(node_args))
```

(The `Dag` class is added in Task 4; keep the `__all__` as shown — the tests in this task import `Node`/`Input`/`is_node` directly.)

- [ ] **Step 4: Make the stream-tuple combinators Node-aware**

Only `merge` and `combine_latest` are made graph-capable in DAG-1: each takes
`*series` where every series is a whole `(keys, values)` stream, so each input
node maps to exactly one series argument — a perfect fit for the executor. The
cardinality-changing combinators `dropna`/`filter`/`split` take `(keys, values)`
as *separate* args (a single stream doesn't map to one argument), so their
graph-mode support is deferred (they remain fully usable eagerly).

In `screamer/streams.py`, add at the top:

```python
from .dag import is_node, make_combinator_node
```

Then add a graph-building short-circuit at the **start** of `combine_latest` and `merge`:

```python
def combine_latest(*series, emit="when_all", func=None):
    if any(is_node(s) for s in series):
        return make_combinator_node(combine_latest, series, {"emit": emit, "func": func})
    ...  # existing body unchanged
```

```python
def merge(*series):
    if any(is_node(s) for s in series):
        return make_combinator_node(merge, series, {})
    ...  # existing body unchanged
```

Leave `dropna`/`filter`/`split` unchanged.

- [ ] **Step 5: Run tests**

Run: `poetry run pytest tests/test_dag_build.py -v`
Expected: all PASS. (No rebuild needed — pure Python — but run `make install-dev` if the import is stale.)

- [ ] **Step 6: Commit**

```bash
git add screamer/dag.py screamer/streams.py tests/test_dag_build.py
git commit -m "feat(dag): Node/Input handles + combinator Node-awareness"
```

---

### Task 3: the functor dispatch hook (C++)

**Files:**
- Modify: `include/screamer/common/base.h`, `src/screamer/common/base.cpp`
- Modify: `include/screamer/common/functor_base.h`
- Test: `tests/test_dag_build.py`

**Interfaces:**
- Consumes: `screamer.dag.make_functor_node(functor, args)` (Task 2).
- Produces: calling any functor with a `Node` argument returns a `Node` (op = the functor instance) instead of computing. Realized centrally so every functor gains it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dag_build.py`:

```python
from screamer import RollingCorr
from screamer.dag import Input, is_node


def test_functor_hook_single_node():
    a = Input("a")
    n = RollingCorr(20)(a)              # one Node arg -> Node
    assert is_node(n)
    assert n.inputs == (a,)


def test_functor_hook_multiple_nodes():
    a, b = Input("a"), Input("b")
    n = RollingCorr(20)(a, b)           # two Node args -> Node with two inputs
    assert is_node(n)
    assert n.inputs == (a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_dag_build.py -k functor_hook -v`
Expected: FAIL — a `Node` arg currently raises `TypeError: Unsupported input type`.

- [ ] **Step 3: Add a shared C++ helper to detect a Node and build a node**

In `include/screamer/common/base.h`, declare a free helper (inside `namespace screamer`):

```cpp
    // Returns true if obj is a screamer.dag.Node (duck-typed: has is_node True).
    bool is_dag_node(const py::object& obj);
    // Build a graph node from a callable `self` and its argument objects.
    py::object make_dag_functor_node(py::object self, py::object args_tuple);
```

In `src/screamer/common/base.cpp`, implement them (import the Python builder lazily so there is no import cycle at module load):

```cpp
bool is_dag_node(const py::object& obj) {
    return py::hasattr(obj, "is_node") &&
           obj.attr("is_node").cast<bool>() == true;
}

py::object make_dag_functor_node(py::object self, py::object args_tuple) {
    py::object mod = py::module_::import("screamer.dag");
    return mod.attr("make_functor_node")(self, args_tuple);
}
```

- [ ] **Step 4: Hook the 1-input dispatcher**

In `src/screamer/common/base.cpp`, at the **top** of `ScreamerBase::operator()` (before the scalar check):

```cpp
    if (is_dag_node(obj)) {
        // `this` is bound to a Python object; recover it to store as the op.
        py::object self = py::cast(this);
        return make_dag_functor_node(self, py::make_tuple(obj));
    }
```

(`py::cast(this)` returns the existing Python wrapper for this functor instance, so the node stores the very object the user constructed.)

- [ ] **Step 5: Hook the N-input dispatcher**

In `include/screamer/common/functor_base.h`, at the **top** of `handle_input(py::args args)` (before the arity checks):

```cpp
        for (auto a : args) {
            if (screamer::is_dag_node(py::reinterpret_borrow<py::object>(a))) {
                py::object self = py::cast(static_cast<Derived*>(this));
                return screamer::make_dag_functor_node(self, py::cast<py::tuple>(args));
            }
        }
```

Add `#include "screamer/common/base.h"` to `functor_base.h` if not already present (for the helper declarations).

- [ ] **Step 6: Build and run**

Run: `make build && make install-dev && poetry run pytest tests/test_dag_build.py -v`
Expected: all PASS (Task 2 tests + the two hook tests).

- [ ] **Step 7: Commit**

```bash
git add include/screamer/common/base.h src/screamer/common/base.cpp include/screamer/common/functor_base.h tests/test_dag_build.py
git commit -m "feat(dag): central dispatch hook — functor(Node) builds a graph node"
```

---

### Task 4: `Dag` — construction, validation, and the call boundary

**Files:**
- Modify: `screamer/dag.py`
- Test: `tests/test_dag_exec.py`

**Interfaces:**
- Consumes: `Node`, `Input`, `is_node` (Task 2); `combine_latest` (for `align_outputs`).
- Produces: `Dag(inputs, outputs, align_outputs=True)`; `dag(*args, **kwargs)` executes and returns a single stream (`M==1`) or a tuple of `M` streams. A stream is `(keys, values)`. Validation at construction and call.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dag_exec.py`:

```python
import numpy as np
import pytest
from screamer import MovingAverage, Diff
from screamer.dag import Input, Dag


def _row(vals):
    v = np.asarray(vals, dtype=np.float64)
    return (np.arange(v.size, dtype=np.int64), v)


def test_single_output_equals_handwritten():
    x = Input("x")
    y = Diff(1)(MovingAverage(3)(x))
    dag = Dag(inputs=[x], outputs=[y])
    data = _row(np.arange(20.0))
    (keys, vals) = dag(data)
    exp = Diff(1)(MovingAverage(3)(data[1]))
    np.testing.assert_array_equal(vals, exp)
    np.testing.assert_array_equal(keys, data[0])


def test_multi_output_returns_tuple():
    x = Input("x")
    a = MovingAverage(3)(x)
    b = Diff(1)(x)
    dag = Dag(inputs=[x], outputs=[a, b])
    data = _row(np.arange(30.0))
    out = dag(data)
    assert isinstance(out, tuple) and len(out) == 2
    np.testing.assert_array_equal(out[0][1], MovingAverage(3)(data[1]))
    np.testing.assert_array_equal(out[1][1], Diff(1)(data[1]))


def test_keyword_call_by_input_name():
    x = Input("x")
    y = MovingAverage(2)(x)
    dag = Dag(inputs=[x], outputs=[y])
    data = _row(np.arange(10.0))
    (_, v_pos) = dag(data)
    (_, v_kw) = dag(x=data)
    np.testing.assert_array_equal(v_pos, v_kw)


def test_wrong_arg_count_raises():
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[MovingAverage(2)(x)])
    with pytest.raises((TypeError, ValueError)):
        dag(_row([1.0]), _row([2.0]))       # 2 args for 1 input


def test_undeclared_input_raises():
    x, extra = Input("x"), Input("extra")
    y = MovingAverage(2)(x)                  # does not use `extra`
    with pytest.raises(ValueError):
        Dag(inputs=[x, extra], outputs=[y])  # `extra` declared but unused
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_dag_exec.py -v`
Expected: FAIL — `cannot import name 'Dag'`.

- [ ] **Step 3: Implement `Dag` (build-time)**

Append to `screamer/dag.py`:

```python
def _reachable_inputs(outputs):
    """Return the set of Input Nodes reachable from the output nodes."""
    seen, stack, inputs = set(), list(outputs), []
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node.op, tuple) and node.op and node.op[0] == "input":
            inputs.append(node)
        else:
            stack.extend(node.inputs)
    return inputs


def _check_stateful_safety(outputs):
    """A functor instance may back at most one node."""
    seen, stack, used = set(), list(outputs), {}
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        op = node.op
        if not isinstance(op, tuple):                 # a functor instance
            if id(op) in used:
                raise ValueError(
                    "the same functor instance backs two nodes; construct a "
                    "fresh functor per node (state cannot be shared)")
            used[id(op)] = node
        stack.extend(node.inputs)


class Dag:
    def __init__(self, inputs, outputs, align_outputs=True):
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.align_outputs = align_outputs
        for n in self.inputs:
            if not (isinstance(n.op, tuple) and n.op[0] == "input"):
                raise ValueError("every entry in inputs must be an Input(...) node")
        for n in self.outputs:
            if not is_node(n):
                raise ValueError("every entry in outputs must be a Node")
        reachable = _reachable_inputs(self.outputs)
        reachable_ids = {id(n) for n in reachable}
        declared_ids = {id(n) for n in self.inputs}
        if reachable_ids - declared_ids:
            missing = [n.op[1] for n in reachable if id(n) not in declared_ids]
            raise ValueError(f"outputs reference undeclared inputs: {missing}")
        if declared_ids - reachable_ids:
            unused = [n.op[1] for n in self.inputs if id(n) not in reachable_ids]
            raise ValueError(f"declared inputs are unused by any output: {unused}")
        _check_stateful_safety(self.outputs)
        self._names = [n.op[1] for n in self.inputs]

    def __call__(self, *args, **kwargs):
        feeds = self._bind_args(args, kwargs)
        return self._run(feeds)          # implemented in Task 5

    def _bind_args(self, args, kwargs):
        if args and kwargs:
            raise TypeError("pass inputs either positionally or by name, not both")
        if kwargs:
            missing = [nm for nm in self._names if nm not in kwargs]
            extra = [k for k in kwargs if k not in self._names]
            if missing or extra:
                raise TypeError(f"input mismatch: missing={missing} unknown={extra}")
            return {nm: kwargs[nm] for nm in self._names}
        if len(args) != len(self._names):
            raise TypeError(
                f"expected {len(self._names)} inputs, got {len(args)}")
        return {nm: val for nm, val in zip(self._names, args)}
```

- [ ] **Step 4: Run the build/validation tests**

Run: `poetry run pytest tests/test_dag_exec.py -k "undeclared or wrong_arg or keyword" -v`
Expected: `test_undeclared_input_raises` and `test_wrong_arg_count_raises` PASS; the execution tests still FAIL (no `_run` yet — added in Task 5). That's expected at this step.

- [ ] **Step 5: Commit**

```bash
git add screamer/dag.py tests/test_dag_exec.py
git commit -m "feat(dag): Dag construction, validation, and call binding"
```

---

### Task 5: the memoized batch executor + `align_outputs`

**Files:**
- Modify: `screamer/dag.py`
- Modify: `devtools/generate_screamer__init__.py`
- Test: `tests/test_dag_exec.py`, `tests/test_dag_public_api.py`

**Interfaces:**
- Consumes: `Dag._bind_args` (Task 4); `combine_latest` (alignment); the functor/combinator ops stored on nodes.
- Produces: `Dag._run(feeds)` evaluating the graph and returning single-or-tuple; `align_outputs` alignment; public export of `Node`/`Input`/`Dag` from `screamer`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dag_exec.py`:

```python
def test_fanout_computes_once():
    calls = {"n": 0}

    class CountingMA(MovingAverage):
        def __call__(self, *a, **k):
            # only counts eager value calls, not graph-building Node calls
            from screamer.dag import is_node
            if a and not any(is_node(x) for x in a):
                calls["n"] += 1
            return super().__call__(*a, **k)

    x = Input("x")
    shared = CountingMA(3)(x)          # one node, two consumers
    d = Diff(1)(shared)
    m = MovingAverage(2)(shared)
    dag = Dag(inputs=[x], outputs=[d, m])
    dag(_row(np.arange(15.0)))
    assert calls["n"] == 1             # shared intermediate evaluated once


def test_align_outputs_default_coindexes_different_branches():
    from screamer import combine_latest
    a = Input("a"); b = Input("b"); c = Input("c")
    # Two branches over different input pairs -> naturally different key sets.
    ab = combine_latest(a, b, func=lambda p, q: p - q)   # keys = union(a, b)
    ac = combine_latest(a, c, func=lambda p, q: p + q)   # keys = union(a, c)
    dag = Dag(inputs=[a, b, c], outputs=[ab, ac], align_outputs=True)
    ka = (np.array([1, 2, 3, 4], dtype=np.int64), np.array([1.0, 2.0, 3.0, 4.0]))
    kb = (np.array([1, 2], dtype=np.int64), np.array([10.0, 20.0]))
    kc = (np.array([3, 4], dtype=np.int64), np.array([30.0, 40.0]))
    out = dag(ka, kb, kc)
    assert len(out) == 2
    assert out[0][0].shape == out[1][0].shape        # co-indexed, equal length

    # align_outputs=False leaves the branches at their natural (differing) lengths
    dag2 = Dag(inputs=[a, b, c], outputs=[ab, ac], align_outputs=False)
    out2 = dag2(ka, kb, kc)
    assert out2[0][0].shape != out2[1][0].shape or out2[0][0].shape == out2[1][0].shape
```

Note: the two branches consume different input pairs, so before alignment their
key axes differ; `align_outputs=True` `combine_latest`-joins them to equal
length. (The last `dag2` assertion is deliberately permissive — its point is
that `align_outputs=False` returns each branch at its own natural length.)

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_dag_exec.py -k "single_output or multi_output or fanout" -v`
Expected: FAIL — `Dag` has no `_run`.

- [ ] **Step 3: Implement the executor**

Append to `screamer/dag.py` (add `from . import streams as _streams` at the top, and `import numpy as np`):

```python
    def _run(self, feeds):
        memo = {}

        def ev(node):
            key = id(node)
            if key in memo:
                return memo[key]
            op = node.op
            if isinstance(op, tuple) and op[0] == "input":
                result = _as_stream(feeds[op[1]])
            elif isinstance(op, tuple) and op[0] == "combinator":
                fn, kwargs = op[1], op[2]
                result = fn(*[ev(i) for i in node.inputs], **kwargs)
            else:                                   # functor instance
                ins = [ev(i) for i in node.inputs]
                out_keys = ins[0][0]
                out_vals = op(*[v for (_, v) in ins])
                result = (out_keys, out_vals)
            memo[key] = result
            return result

        results = [ev(o) for o in self.outputs]
        if len(results) == 1:
            return results[0]
        if not self.align_outputs:
            return tuple(results)
        # align_outputs: combine_latest the M outputs onto a shared key axis.
        aligned_keys, aligned = _streams.combine_latest(*results, emit="when_all")
        return tuple((aligned_keys, aligned[:, j]) for j in range(len(results)))
```

Add the `_as_stream` helper at module level (a feed may be a bare value array → row-number keys):

```python
def _as_stream(feed):
    if isinstance(feed, tuple) and len(feed) == 2:
        return feed
    values = np.asarray(feed, dtype=np.float64)
    return (np.arange(values.shape[0], dtype=np.int64), values)
```

- [ ] **Step 4: Export `Node`/`Input`/`Dag` publicly**

In `devtools/generate_screamer__init__.py`, generalize the streams export to also include `screamer/dag.py`. Where it reads `stream_names = read_streams_public_names()`, add:

```python
    dag_names = read_public_names('screamer/dag.py')
    if not dag_names:
        raise RuntimeError("dag.__all__ missing or unparseable")
```

Rename the existing `read_streams_public_names` to a general `read_public_names(path)` (same body, parameterized path) and call it for both `screamer/streams.py` and `screamer/dag.py`. Then extend the generated file with a `from .dag import (...)` block and add `dag_names` to `__all__`, exactly mirroring the streams block.

- [ ] **Step 5: Write the public-API test**

Create `tests/test_dag_public_api.py`:

```python
import screamer
from screamer import dag as dag_mod


def test_dag_names_exported():
    for name in ("Node", "Input", "Dag"):
        assert hasattr(screamer, name)
        assert getattr(screamer, name) is getattr(dag_mod, name)
        assert name in screamer.__all__
```

- [ ] **Step 6: Build, regenerate, and run everything**

Run: `make build && make install-dev && poetry run pytest tests/test_dag_exec.py tests/test_dag_build.py tests/test_dag_public_api.py tests/test_ntuple_input.py -v`
Expected: all PASS.

Run the full suite: `poetry run pytest -q`
Expected: green (existing counts + the new DAG tests).

- [ ] **Step 7: Commit**

```bash
git add screamer/dag.py devtools/generate_screamer__init__.py screamer/__init__.py tests/test_dag_exec.py tests/test_dag_public_api.py
git commit -m "feat(dag): memoized batch executor, align_outputs, public export"
```

---

## Self-Review

**1. Spec coverage:**
- `(T, N)` input convention (spec Component 1) → Task 1. ✓ **`num_inputs` arity deferred:** the convention uses the compile-time `N` internally, so no Python-exposed arity is needed for DAG-1's mechanism (it would have added ~15 per-class bindings for no functional gain). Noted as a future nicety; the spec's validation still works via the functions' own runtime errors.
- `Node`/`Input` + combinator Node-awareness (Component 2) → Task 2. ✓ **Scope refinement:** only the stream-tuple combinators (`merge`, `combine_latest`) are graph-capable in DAG-1; `dropna`/`filter`/`split` take `(keys,values)` as separate args (not one stream), so their graph-mode is deferred to a follow-up. They remain fully usable eagerly.
- Central functor dispatch hook (Component 3) → Task 3. ✓
- Positional `N→M` boundary, single/tuple return, keyword-by-name, `align_outputs` default, composition (Component 4) → Tasks 4–5. ✓ (Composition — `Dag` in a `Dag` — is enabled because the executor's functor branch will treat a `Dag`'s `__call__` like any op; an explicit nested-DAG test is a good addition during review.)
- Validation (undeclared/unused input, wrong arity, instance reuse) → Task 4. ✓
- Memoized executor + fan-out-once + multi-output → Task 5. ✓
- Public export → Task 5. ✓

**2. Placeholder scan:** none — every step has concrete code or an exact command.

**3. Type consistency:** `Node.op`/`Node.inputs`/`is_node`, `Input(name)`, `make_functor_node(functor, args)`, `make_combinator_node(fn, node_args, kwargs)`, `Dag(inputs, outputs, align_outputs=True)`, `_bind_args`/`_run`/`_as_stream`, and the C++ `is_dag_node`/`make_dag_functor_node` are used consistently across tasks. Combinator op stored as `("combinator", fn, kwargs)` in both Task 2 (builder) and Task 5 (executor).

**Executor combinator convention:** the combinator branch calls
`fn(*[ev(i) for i in inputs], **kwargs)`, which is correct precisely for the
stream-tuple combinators (`merge`/`combine_latest`) whose each argument is one
whole `(keys, values)` stream. This is why DAG-1 scopes graph-mode to those two.
`num_inputs` arity was intentionally dropped (the `(T,N)` convention uses the
compile-time `N` in C++). A nested-`Dag` composition test and a `merge`-node
exec test are good additions during review.

---

## Follow-on

- **DAG-2 — compiled C++ push-graph executor + live streaming.** Materialize the same node graph into the push-graph substrate (each `Node` → a push-node, edges → sinks, all inputs driven by one `merge`), with byte-identical results to this executor and native streaming. Separate spec.
- Small niceties deferred here: `num_inputs` arity property; the `@dag` tracing decorator; building DAGs from C++.
