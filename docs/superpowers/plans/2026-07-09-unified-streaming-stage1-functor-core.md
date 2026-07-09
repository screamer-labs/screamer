# Unified Streaming, Stage 1: functor callable core (lazy iterables) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every screamer functor type-propagate uniformly, so a lazy iterable in
gives a lazy iterator out for N-in-M-out functors just as it already does for
1-in-1-out functors, backed by a single generic lazy iterator.

**Architecture:** Introduce one `LazyEvalIterator` that streams any `EvalOp` (pulls
`n_in()` values per step, calls `eval`, yields `n_out()` outputs lazily). Route both the
1-in-1-out path (`base.cpp`) and the N-in-M-out paths (`functor_base.h`) through it,
retiring the two existing single-purpose iterators (`LazyIterator`, `FunctorIterator`).
No numeric results change; only the return type for lazy (generator) inputs changes from
an eager list to a lazy iterator.

**Tech Stack:** C++17, pybind11, numpy, pytest. Build with `make install-dev`.

## Global Constraints

- Batch results are unchanged; the acceptance oracle is the current batch output. Every
  functor must satisfy `batch == eager-list == lazy-iterator` for the same input.
- All numeric logic stays in C++; Python is a thin shim (no compute loops added in Python).
- After any C++ change run `make install-dev` (not just `make build`), or Python imports a
  stale binding.
- Never edit version files; do not bump versions.
- No em-dashes and no ` -- ` in comments, docstrings, or prose; use commas, colons,
  parentheses, or separate sentences.
- Keep the C++ pure-EvalOp so a future WASM binding inherits it.

## Definitions (current state, verified)

- `ScreamerBase::operator()` in `src/screamer/common/base.cpp` (lines 17 to 50) dispatches
  a 1-in-1-out functor: scalar to scalar; numpy array or list to a batch array; a lazy
  iterable to a `LazyIterator` (already lazy); a `Node` to a graph node.
- `FunctorBase<Derived, N, M>` in `include/screamer/common/functor_base.h` dispatches
  N-in-M-out functors. Its iterable branches (`handle_input_Ni_1o` around line 520 to 570,
  `handle_input_Ni_Mo` around line 666, `handle_input_1i_Mo` around line 363) build an
  **eager** `std::vector<ResultTuple>` and return it as a list. This is the inconsistency.
- Every functor is an `EvalOp` (`include/screamer/common/eval_op.h`): `n_in()`, `n_out()`,
  `eval(const double* in, double* out)`, `reset()`.
- Concrete test functors: `CumSum` (1-in-1-out, `ScreamerBase`), `Add`/`Sub`
  (`FunctorBase<_, 2, 1>`, `include/screamer/arithmetic.h`), `BollingerBands`
  (`FunctorBase<_, 1, 3>`, `include/screamer/bollinger_bands.h`).

## File Structure

- `include/screamer/common/lazy_eval_iterator.h` (create): the one generic lazy iterator
  over `EvalOp`.
- `bindings/bindings_core.cpp` (modify): bind `LazyEvalIterator`; remove the `LazyIterator`
  binding once unused.
- `src/screamer/common/base.cpp` (modify): return a `LazyEvalIterator` from the
  1-in-1-out iterable branch.
- `include/screamer/common/functor_base.h` (modify): return a `LazyEvalIterator` from the
  three N-in-M-out iterable branches instead of eager lists.
- `include/screamer/common/iterator.h` and `include/screamer/common/functor_iterator.h`
  (delete at the end): the two retired iterators.
- `bindings/bindings_myfunctors.cpp` (modify): drop the `bind_functor_iterator` call.
- `tests/test_lazy_streaming.py` (create): the acceptance tests.

## Interfaces

- Produces (used by all later stages): `screamer.LazyEvalIterator`, a Python iterator that
  wraps one `EvalOp` and one or more input iterables and yields, per event, a scalar when
  `n_out() == 1` or a tuple of length `n_out()` when `n_out() > 1`. Constructed only from
  C++ dispatch, never directly by users.
- Consumes: `EvalOp` (`n_in`, `n_out`, `eval`, and the Python wrapper for keep-alive).

---

### Task 1: `LazyEvalIterator` over `EvalOp`

**Files:**
- Create: `include/screamer/common/lazy_eval_iterator.h`
- Modify: `bindings/bindings_core.cpp` (bind the class)
- Test: `tests/test_lazy_streaming.py`

**Interfaces:**
- Produces: a bound `screamer.LazyEvalIterator` with `__iter__` and `__next__`.
- Consumes: `EvalOp&` plus the functor's `py::object` wrapper (keep-alive), and either
  N separate input iterables or one iterable whose items are N-tuples.

- [ ] **Step 1: Write the failing test** (tests/test_lazy_streaming.py)

```python
import types
import numpy as np
import pytest
from screamer import CumSum


def test_lazy_iterator_is_lazy_and_correct():
    # A generator input must yield a lazy iterator, not a list, and match batch.
    def gen():
        for x in [1.0, 2.0, 3.0, 4.0]:
            yield x

    out = CumSum()(gen())
    assert not isinstance(out, list)
    assert hasattr(out, "__next__")          # it is an iterator
    assert list(out) == [1.0, 3.0, 6.0, 10.0]

    # Laziness: the input generator is consumed one item at a time.
    pulled = []
    def spy():
        for x in [1.0, 2.0, 3.0]:
            pulled.append(x)
            yield x
    it = CumSum()(spy())
    assert pulled == []                      # nothing consumed yet
    next(it)
    assert pulled == [1.0]                    # exactly one pulled
```

- [ ] **Step 2: Run to verify it fails**

Run: `make install-dev && python -m pytest tests/test_lazy_streaming.py::test_lazy_iterator_is_lazy_and_correct -x -q`
Expected: today `CumSum()(gen())` returns a `LazyIterator` (this specific test may pass for
the 1-in-1-out case). If it passes, keep it as a regression guard and proceed; the new
class replaces `LazyIterator` in Task 2. The genuinely-failing behavior is exercised in
Task 3 (N-in-M-out). Record the result.

- [ ] **Step 3: Write `include/screamer/common/lazy_eval_iterator.h`**

```cpp
#ifndef SCREAMER_LAZY_EVAL_ITERATOR_H
#define SCREAMER_LAZY_EVAL_ITERATOR_H

#include <vector>
#include <pybind11/pybind11.h>
#include "screamer/common/eval_op.h"

namespace py = pybind11;

namespace screamer {

// One lazy iterator for any EvalOp. Holds the functor's Python wrapper (keep-alive)
// and the input source(s). Per __next__ it collects n_in() input values, calls
// eval() once, and yields a scalar (n_out()==1) or a tuple of n_out() floats.
//
// Two input shapes are supported, matching the C++ dispatch:
//   - `sources` holds n_in() separate iterators (one value pulled from each);
//   - `sources` holds exactly one iterator whose items are n_in()-tuples (unpacked),
//     used for the "one iterable of tuples" call form.
class LazyEvalIterator {
public:
    LazyEvalIterator(py::object op_owner, std::vector<py::object> iterables)
        : op_owner_(std::move(op_owner)),
          op_(op_owner_.cast<EvalOp&>()),
          n_in_(op_.n_in()), n_out_(op_.n_out()),
          in_(op_.n_in()), out_(op_.n_out()) {
        for (auto& it : iterables) iters_.push_back(py::iter(it));
        unpack_tuples_ = (iters_.size() == 1 && n_in_ > 1);
        in_.resize(n_in_);
        out_.resize(n_out_);
    }

    LazyEvalIterator& __iter__() { return *this; }

    py::object __next__() {
        if (unpack_tuples_) {
            py::object item = next_or_stop(iters_[0]);          // an n_in-tuple
            py::sequence seq = py::cast<py::sequence>(item);
            if (py::len(seq) != static_cast<py::ssize_t>(n_in_))
                throw py::value_error("LazyEvalIterator: tuple size does not match n_in");
            for (std::size_t i = 0; i < n_in_; ++i)
                in_[i] = seq[i].cast<double>();
        } else {
            for (std::size_t i = 0; i < n_in_; ++i)
                in_[i] = next_or_stop(iters_[i]).cast<double>();
        }
        op_.eval(in_.data(), out_.data());
        if (n_out_ == 1) return py::float_(out_[0]);
        py::tuple t(n_out_);
        for (std::size_t i = 0; i < n_out_; ++i) t[i] = py::float_(out_[i]);
        return std::move(t);
    }

private:
    static py::object next_or_stop(py::iterator& it) {
        if (it == py::iterator()) throw py::stop_iteration();   // default-constructed sentinel
        py::object v = py::reinterpret_borrow<py::object>(*it);
        ++it;
        return v;
    }

    py::object op_owner_;                 // keeps the functor wrapper alive
    EvalOp& op_;
    std::size_t n_in_, n_out_;
    std::vector<py::iterator> iters_;
    std::vector<double> in_, out_;
    bool unpack_tuples_ = false;
};

}  // namespace screamer
#endif
```

- [ ] **Step 4: Bind it** in `bindings/bindings_core.cpp` (next to the existing
`LazyIterator` binding at line 31):

```cpp
#include "screamer/common/lazy_eval_iterator.h"
// ... inside the module init:
py::class_<screamer::LazyEvalIterator>(m, "LazyEvalIterator")
    .def("__iter__", &screamer::LazyEvalIterator::__iter__,
         py::return_value_policy::reference_internal)
    .def("__next__", &screamer::LazyEvalIterator::__next__);
```

- [ ] **Step 5: Build and run**

Run: `make install-dev && python -m pytest tests/test_lazy_streaming.py -x -q`
Expected: PASS (the class exists and is bound; the 1-in-1-out test passes because Task 2
wires it, but the class itself must compile and import cleanly now).

- [ ] **Step 6: Commit**

```bash
git add include/screamer/common/lazy_eval_iterator.h bindings/bindings_core.cpp tests/test_lazy_streaming.py
git commit -m "feat(stream): add generic LazyEvalIterator over EvalOp"
```

---

### Task 2: route the 1-in-1-out iterable path through `LazyEvalIterator`

**Files:**
- Modify: `src/screamer/common/base.cpp:37-40` (the iterable branch of `ScreamerBase::operator()`)
- Test: `tests/test_lazy_streaming.py`

**Interfaces:**
- Consumes: `LazyEvalIterator` from Task 1.
- Produces: `ScreamerBase::operator()` returns a `LazyEvalIterator` for a lazy iterable.

- [ ] **Step 1: Add a batch-equals-lazy test** (append to tests/test_lazy_streaming.py)

```python
def test_1i1o_batch_equals_lazy():
    from screamer import CumSum
    xs = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]
    batch = CumSum()(np.array(xs))                 # array in, array out
    lazy = list(CumSum()(x for x in xs))           # generator in, lazy iterator out
    np.testing.assert_allclose(np.asarray(lazy), batch)
```

- [ ] **Step 2: Run to verify current behavior**

Run: `make install-dev && python -m pytest tests/test_lazy_streaming.py::test_1i1o_batch_equals_lazy -x -q`
Expected: PASS today (the 1-in-1-out path is already lazy via `LazyIterator`). This test is
the regression guard for the swap in Step 3.

- [ ] **Step 3: Replace `LazyIterator` with `LazyEvalIterator`** in
`src/screamer/common/base.cpp`. The current branch (around line 37) is:

```cpp
if (py::isinstance<py::iterable>(obj)) {
    return py::cast(LazyIterator(obj.cast<py::iterable>(), py::cast(this)));
}
```

Change it to (add `#include "screamer/common/lazy_eval_iterator.h"` at the top, remove the
`#include "screamer/common/iterator.h"`):

```cpp
if (py::isinstance<py::iterable>(obj)) {
    std::vector<py::object> sources{obj};       // a single iterable of scalars (n_in==1)
    return py::cast(LazyEvalIterator(py::cast(this), std::move(sources)));
}
```

- [ ] **Step 4: Build and run**

Run: `make install-dev && python -m pytest tests/test_lazy_streaming.py -x -q`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/screamer/common/base.cpp tests/test_lazy_streaming.py
git commit -m "refactor(stream): 1-in-1-out functors stream via LazyEvalIterator"
```

---

### Task 3: make N-in-M-out functor iterables lazy

**Files:**
- Modify: `include/screamer/common/functor_base.h` (the three iterable branches:
  `handle_input_1i_Mo` ~363, `handle_input_Ni_1o` ~520-570, `handle_input_Ni_Mo` ~666-720)
- Test: `tests/test_lazy_streaming.py`

**Interfaces:**
- Consumes: `LazyEvalIterator` from Task 1.
- Produces: N-in-M-out functors return a `LazyEvalIterator` for lazy iterable inputs.

- [ ] **Step 1: Write the failing tests** (append to tests/test_lazy_streaming.py)

```python
def test_Ni1o_lazy_separate_iterables():
    from screamer import Add
    a = [1.0, 2.0, 3.0]
    b = [10.0, 20.0, 30.0]
    batch = Add()(np.array(a), np.array(b))                 # two arrays -> array
    out = Add()((x for x in a), (y for y in b))             # two generators -> lazy iter
    assert hasattr(out, "__next__") and not isinstance(out, list)
    np.testing.assert_allclose(np.asarray(list(out)), batch)


def test_1iMo_lazy_multi_output_tuples():
    from screamer import BollingerBands
    xs = [10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 12.0, 15.0]
    batch = BollingerBands(5)(np.array(xs))                 # array -> 2D (rows, 3)
    out = BollingerBands(5)(x for x in xs)                  # generator -> lazy iter of 3-tuples
    assert hasattr(out, "__next__") and not isinstance(out, list)
    rows = list(out)
    assert all(isinstance(r, tuple) and len(r) == 3 for r in rows)
    np.testing.assert_allclose(np.asarray(rows), batch, equal_nan=True)
```

- [ ] **Step 2: Run to verify they fail**

Run: `make install-dev && python -m pytest tests/test_lazy_streaming.py -x -q`
Expected: FAIL on the two new tests: today N-in-M-out returns an eager `list`, so
`hasattr(out, "__next__")` is false and `isinstance(out, list)` is true.

- [ ] **Step 3: Replace the eager returns with `LazyEvalIterator`** in
`include/screamer/common/functor_base.h`. Add `#include
"screamer/common/lazy_eval_iterator.h"` at the top. In each of the three iterable branches
(the ones that currently build `std::vector<ResultTuple> results` and loop to fill it),
replace the loop and the `return py::cast(results);` with a lazy return. For the N separate
iterables branch (`handle_input_Ni_1o` / `handle_input_Ni_Mo`, the "all iterable" case):

```cpp
if (all_iterable) {
    std::vector<py::object> sources;
    for (auto input : inputs) sources.push_back(py::reinterpret_borrow<py::object>(input));
    return py::cast(LazyEvalIterator(py::cast(this), std::move(sources)));
}
```

For the single-iterable-of-N-tuples branch (`handle_input_Ni_1o` / `handle_input_Ni_Mo`
Case 1, and `handle_input_1i_Mo` Case 3), pass the single iterable as the only source, so
`LazyEvalIterator` unpacks each item (`unpack_tuples_` is true when `n_in > 1`; for
`handle_input_1i_Mo`, `n_in == 1` so items are scalars):

```cpp
std::vector<py::object> sources{ py::reinterpret_borrow<py::object>(input) };
return py::cast(LazyEvalIterator(py::cast(this), std::move(sources)));
```

Keep the scalar and numpy-array branches unchanged (they stay eager: scalar to
scalar/tuple, array to array). Only the iterable branches change.

- [ ] **Step 4: Build and run**

Run: `make install-dev && python -m pytest tests/test_lazy_streaming.py -x -q`
Expected: PASS (all tests, including the two new lazy N-in-M-out ones).

- [ ] **Step 5: Commit**

```bash
git add include/screamer/common/functor_base.h tests/test_lazy_streaming.py
git commit -m "refactor(stream): N-in-M-out functors stream via LazyEvalIterator"
```

---

### Task 4: retire `LazyIterator` and `FunctorIterator`

**Files:**
- Delete: `include/screamer/common/iterator.h`, `include/screamer/common/functor_iterator.h`
- Modify: `bindings/bindings_core.cpp` (remove the `LazyIterator` class binding),
  `bindings/bindings_myfunctors.cpp` (remove the `bind_functor_iterator` call and include)
- Test: `tests/test_lazy_streaming.py` (full suite as the regression check)

**Interfaces:**
- Consumes: nothing new.
- Produces: a single lazy-iterator concept (`LazyEvalIterator`) across the whole functor
  layer.

- [ ] **Step 1: Grep for remaining uses**

Run: `grep -rn "LazyIterator\|FunctorIterator\|functor_iterator.h\|common/iterator.h" include/ src/ bindings/`
Expected: after Tasks 2 and 3, the only hits are the class binding in `bindings_core.cpp`,
the `bind_functor_iterator` call and include in `bindings_myfunctors.cpp`, and the two
header files themselves. If any other file still uses them, stop and handle it before
deleting.

- [ ] **Step 2: Remove the bindings and includes**

In `bindings/bindings_core.cpp` delete the `py::class_<screamer::LazyIterator>` block and
its `#include "screamer/common/iterator.h"`. In `bindings/bindings_myfunctors.cpp` delete
the `bind_functor_iterator<...>(...)` call(s) and the `#include
"screamer/common/functor_iterator.h"`.

- [ ] **Step 3: Delete the two headers**

```bash
git rm include/screamer/common/iterator.h include/screamer/common/functor_iterator.h
```

- [ ] **Step 4: Build and run the full suite**

Run: `make install-dev && python -m pytest -q`
Expected: PASS. Baseline before this stage is the current suite count (3916 passed plus the
2 pre-existing `test_oscillators_hlc.py::TestBOP` failures, which are unrelated). This stage
adds `tests/test_lazy_streaming.py` and must add zero new failures.

- [ ] **Step 5: Commit**

```bash
git add bindings/bindings_core.cpp bindings/bindings_myfunctors.cpp
git commit -m "refactor(stream): retire LazyIterator and FunctorIterator (one lazy iterator)"
```

---

## Self-review notes

- **Spec coverage (Stage 1 slice):** type propagation for functors (value/array/lazy-
  iterator in gives same out) is delivered by Tasks 2 and 3; the single reused streaming
  concept is delivered by Tasks 1 and 4. Multi-input tuple/dict unpacking beyond the
  already-existing positional and single-tuple forms, as-of alignment, operators, `Dag`,
  and `resample` are later stages (out of scope here, by design).
- **Not in this stage:** dict/`**kwargs` unpacking (functors are positional and unnamed, so
  it lands with named `Dag` inputs in Stage 3); as-of alignment (index-level, Stage 3);
  scalar push cardinality for window operators (operators are Stage 2).
- **Oracle:** every lazy test asserts equality against the batch (array) output of the same
  functor, so `batch == lazy` is enforced per functor.
- **Line numbers** in this plan are approximate; the implementer confirms the exact branch
  in each `handle_input_*` by matching the described "build `std::vector<ResultTuple>` then
  return a list" shape.

## Next stages (separate plans)

2. Stream operators (`resample`, `dropna`, `filter`, `merge`, `combine_latest`) exposed as
   callables with the one dispatch, backed by the C++ engine (retire the `*_iter` Python
   generators).
3. `Dag` as a function: `dag(iterables)` gives a lazy iterator, `dag(arrays)` batch; retire
   `stream()`/`live()`; as-of alignment as the multi-input default; dict/kwargs unpacking of
   named inputs.
4. `resample` re-signature: single `freq=` (count / index span / timedelta), `(bar_label,
   bar_value)` tuple output, `(index, NaN)` heartbeats.
5. Retire `Stream` and any remaining bespoke surfaces; migration table.
6. Docs and notebooks rewritten onto the one surface.
