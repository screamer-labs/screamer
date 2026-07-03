# DAG-2a — uniform EvalOp interface + arithmetic functors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every functor — 1-input `ScreamerBase` and N-input/M-output `FunctorBase` alike — one uniform C++ op interface (`EvalOp`) so the DAG engine can hold and drive any functor behind a single pointer, and add the binary arithmetic functors (`Add`/`Sub`/`Mul`/`Div`) that become a graph's C++-only reduction vocabulary.

**Architecture:** `EvalOp` is a tiny abstract base (`n_in`/`n_out`/`eval`/`reset`) that both `ScreamerBase` and `FunctorBase<_,N,M>` inherit and implement in terms of their existing `process_scalar`/`call`. It is bound to Python so every functor (registered under it) exposes `num_inputs`/`num_outputs`, and the engine (DAG-2b) can recover an `EvalOp*` from any functor object generically. Arithmetic functors follow the existing 2→1 pattern (`Atan2`/`Hypot`).

**Tech Stack:** C++17, pybind11, numpy, pytest.

## Global Constraints

- **One interface, not two node families:** `EvalOp` is the single abstraction the DAG engine holds; no per-arity duplication. `ScreamerBase` and `FunctorBase` both derive from `EvalOp`.
- **`eval` is defined in terms of existing methods** — `process_scalar` (1→1) / `call` (N→M); no algorithm is reimplemented, and per-event `eval` equals the batch array path (foundation identity).
- **Hot path untouched:** adding `EvalOp` must not add any per-call cost to eager scalar/array functor calls (it only adds virtuals, called by the engine, not by `operator()`/`handle_input`).
- **Arithmetic functors are C++-only** (`Add`/`Sub`/`Mul`/`Div`, 2→1) — the graph reduction vocabulary replacing DAG-1 Python `func=` lambdas.
- Compute functors' math is never modified. Never hand-edit `screamer/__init__.py` or version files.
- Build: `make build` **then `make install-dev`**. Tests: `poetry run pytest tests/test_evalop.py tests/test_arithmetic.py -v`.

---

## File Structure

- `include/screamer/common/eval_op.h` (create) — the `EvalOp` abstract interface.
- `include/screamer/common/base.h` (modify) — `ScreamerBase : public EvalOp` + `eval`/`n_in`/`n_out` decls.
- `src/screamer/common/base.cpp` (modify) — `ScreamerBase::eval`/`n_in`/`n_out` impls.
- `include/screamer/common/functor_base.h` (modify) — `FunctorBase<_,N,M> : public EvalOp` + `eval`/`n_in`/`n_out`.
- `bindings/bindings_core.cpp` (modify) — bind `EvalOp` (num_inputs/num_outputs + `_eval_op` test helper); `ScreamerBase` derives from it.
- `bindings/*.cpp` (modify) — register every `FunctorBase`-derived functor's `py::class_` under `EvalOp`.
- `include/screamer/arithmetic.h` (create) — `Add`/`Sub`/`Mul`/`Div`.
- `bindings/bindings_math.cpp` (modify) — bind the four arithmetic functors.
- Tests: `tests/test_evalop.py`, `tests/test_arithmetic.py`.

---

### Task 1: the `EvalOp` interface (1-input side)

**Files:**
- Create: `include/screamer/common/eval_op.h`
- Modify: `include/screamer/common/base.h`, `src/screamer/common/base.cpp`
- Modify: `bindings/bindings_core.cpp`
- Test: `tests/test_evalop.py`

**Interfaces:**
- Produces:
  - C++: `struct EvalOp { virtual size_t n_in() const=0; virtual size_t n_out() const=0; virtual void eval(const double* in, double* out)=0; virtual void reset()=0; virtual ~EvalOp()=default; };`
  - `ScreamerBase : public EvalOp` with `n_in()=1`, `n_out()=1`, `eval(in,out){out[0]=process_scalar(in[0]);}`.
  - Python: every `ScreamerBase` functor exposes `.num_inputs`/`.num_outputs` (both `1`); `screamer_bindings._eval_op(op, [inputs]) -> [outputs]` calls `eval` once.

- [ ] **Step 1: Write the failing test**

Create `tests/test_evalop.py`:

```python
import numpy as np
from screamer import RollingMean
from screamer import screamer_bindings as _b


def test_screamerbase_arity():
    f = RollingMean(3)
    assert f.num_inputs == 1
    assert f.num_outputs == 1


def test_eval_matches_process_scalar():
    f = RollingMean(3)
    g = RollingMean(3)
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    # eval one value at a time == process_scalar one value at a time
    got = [_b._eval_op(f, [x])[0] for x in xs]
    exp = [g.process_scalar(x) for x in xs]
    np.testing.assert_array_equal(got, exp)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_evalop.py -v`
Expected: FAIL — `RollingMean` has no `num_inputs`; `_eval_op` undefined.

- [ ] **Step 3: Create the `EvalOp` interface**

`include/screamer/common/eval_op.h`:

```cpp
#ifndef SCREAMER_EVAL_OP_H
#define SCREAMER_EVAL_OP_H

#include <cstddef>

namespace screamer {

// The uniform op interface the DAG engine holds for ANY functor, regardless of
// arity. ScreamerBase (1->1) and FunctorBase<_,N,M> (N->M) both implement it in
// terms of their existing process_scalar/call. eval() processes exactly one
// event: it reads n_in() inputs from `in` and writes n_out() outputs to `out`.
struct EvalOp {
    virtual ~EvalOp() = default;
    virtual std::size_t n_in() const = 0;
    virtual std::size_t n_out() const = 0;
    virtual void eval(const double* in, double* out) = 0;
    virtual void reset() = 0;
};

} // namespace screamer
#endif
```

- [ ] **Step 4: Make `ScreamerBase` implement `EvalOp`**

In `include/screamer/common/base.h`, include the interface and derive:

```cpp
#include "screamer/common/eval_op.h"
```

Change the class declaration to `class ScreamerBase : public EvalOp {` and add these overrides in the `public:` section (keep the existing `reset`/`process_scalar` — `reset` already satisfies `EvalOp::reset`):

```cpp
    std::size_t n_in() const override { return 1; }
    std::size_t n_out() const override { return 1; }
    void eval(const double* in, double* out) override { out[0] = process_scalar(in[0]); }
```

(The existing `virtual void reset() {}` now overrides `EvalOp::reset` — that's fine.)

- [ ] **Step 5: Bind `EvalOp` and register `ScreamerBase` under it**

In `bindings/bindings_core.cpp`, add `#include "screamer/common/eval_op.h"` and, inside `init_bindings_core`, before the `ScreamerBase` binding:

```cpp
    py::class_<screamer::EvalOp>(m, "EvalOp")
        .def_property_readonly("num_inputs", &screamer::EvalOp::n_in)
        .def_property_readonly("num_outputs", &screamer::EvalOp::n_out);

    // Test/engine helper: run one event through an op.
    m.def("_eval_op", [](screamer::EvalOp& op, const std::vector<double>& in) {
        std::vector<double> out(op.n_out());
        op.eval(in.data(), out.data());
        return out;
    });
```

Change the `ScreamerBase` binding line from `py::class_<screamer::ScreamerBase>(m, "ScreamerBase")` to:

```cpp
    py::class_<screamer::ScreamerBase, screamer::EvalOp>(m, "ScreamerBase")
```

(Ensure `<vector>` is included in `bindings_core.cpp`.)

- [ ] **Step 6: Build and run**

Run: `make build && make install-dev && poetry run pytest tests/test_evalop.py -v`
Expected: both tests PASS. `num_inputs`/`num_outputs` now exist on every `ScreamerBase` functor (they inherit `EvalOp`).

- [ ] **Step 7: Commit**

```bash
git add include/screamer/common/eval_op.h include/screamer/common/base.h src/screamer/common/base.cpp bindings/bindings_core.cpp tests/test_evalop.py
git commit -m "feat(core): EvalOp uniform op interface; ScreamerBase implements it"
```

---

### Task 2: `FunctorBase` implements `EvalOp` (N-input/M-output side)

**Files:**
- Modify: `include/screamer/common/functor_base.h`
- Modify: every `bindings/*.cpp` that binds a `FunctorBase`-derived functor
- Test: `tests/test_evalop.py`

**Interfaces:**
- Consumes: `EvalOp` (Task 1).
- Produces: `FunctorBase<_,N,M> : public EvalOp` with `n_in()=N`, `n_out()=M`, `eval` calling `call({in...})` and writing `M` outputs. Every N-input/M-output functor exposes `.num_inputs`/`.num_outputs`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evalop.py`:

```python
from screamer import RollingCorr, Cart2Polar, BollingerBands


def test_functorbase_arity():
    assert RollingCorr(20).num_inputs == 2 and RollingCorr(20).num_outputs == 1
    assert Cart2Polar().num_inputs == 2 and Cart2Polar().num_outputs == 2
    assert BollingerBands(20).num_inputs == 1 and BollingerBands(20).num_outputs == 3


def test_eval_matches_call_2in_1out():
    f = RollingCorr(10)
    g = RollingCorr(10)
    xs = np.random.default_rng(0).standard_normal(30)
    ys = np.random.default_rng(1).standard_normal(30)
    got = [_b._eval_op(f, [x, y])[0] for x, y in zip(xs, ys)]
    exp = [g(x, y) for x, y in zip(xs, ys)]
    np.testing.assert_array_equal(got, exp)


def test_eval_matches_call_2in_2out():
    f = Cart2Polar()
    out = _b._eval_op(f, [3.0, 4.0])
    exp = Cart2Polar()(3.0, 4.0)               # tuple (r, theta)
    np.testing.assert_array_equal(out, list(exp))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_evalop.py::test_functorbase_arity -v`
Expected: FAIL — `RollingCorr` has no `num_inputs` (not yet registered under `EvalOp`).

- [ ] **Step 3: Make `FunctorBase` implement `EvalOp`**

In `include/screamer/common/functor_base.h`, add `#include "screamer/common/eval_op.h"` and change the class to derive from `EvalOp`:

```cpp
template <class Derived, size_t N, size_t M>
class FunctorBase : public EvalOp {
```

Add these overrides in the `public:` section (near `call`/`reset`; `reset` is already virtual and satisfies `EvalOp::reset`):

```cpp
    std::size_t n_in() const override { return N; }
    std::size_t n_out() const override { return M; }
    void eval(const double* in, double* out) override {
        InputArray inputs;
        for (std::size_t i = 0; i < N; ++i) inputs[i] = in[i];
        if constexpr (M == 1) {
            out[0] = call(inputs);
        } else {
            detail::write_tuple_to_memory(out, call(inputs));
        }
    }
```

(`detail::write_tuple_to_memory` already exists in this file and is used by the numpy handlers.)

- [ ] **Step 4: Register every `FunctorBase` functor under `EvalOp`**

Find every binding of a `FunctorBase`-derived functor — grep: `grep -rn "py::class_<screamer::" bindings/ | grep -v "ScreamerBase"`. These are the classes whose `py::class_<X>(...)` has **no** second (base) template argument (1-input functors already list `, ScreamerBase` and inherit `EvalOp` transitively — leave them). For each N-input/M-output functor (e.g. `RollingCorr`, `RollingCov`, `RollingBeta`, `RollingSpread`, `Cart2Polar`, `Polar2Cart`, `Hypot`, `Atan2`, `BollingerBands`, `RollingMinMax`, `MyFunctor11`, `MyFunctor22`, and any other with no bound base), change:

```cpp
    py::class_<screamer::RollingCorr>(m, "RollingCorr")
```
to:
```cpp
    py::class_<screamer::RollingCorr, screamer::EvalOp>(m, "RollingCorr")
```

Add `#include "screamer/common/eval_op.h"` to each binding file you touch. Do NOT change the 1-input functors (already `, ScreamerBase`).

- [ ] **Step 5: Build and run**

Run: `make build && make install-dev && poetry run pytest tests/test_evalop.py -v`
Expected: all `test_evalop.py` tests PASS (1-input from Task 1 + N-input/M-output here).

- [ ] **Step 6: Full-suite guard**

Run: `poetry run pytest -q`
Expected: green (the base-class change must not regress any functor).

- [ ] **Step 7: Commit**

```bash
git add include/screamer/common/functor_base.h bindings/ tests/test_evalop.py
git commit -m "feat(core): FunctorBase implements EvalOp; register N->M functors under it"
```

---

### Task 3: binary arithmetic functors `Add`/`Sub`/`Mul`/`Div`

**Files:**
- Create: `include/screamer/arithmetic.h`
- Modify: `bindings/bindings_math.cpp`
- Test: `tests/test_arithmetic.py`

**Interfaces:**
- Consumes: `FunctorBase<_,2,1>`, `EvalOp` (Tasks 1–2), the `(T,N)` input convention (already in `main`).
- Produces: `Add`/`Sub`/`Mul`/`Div` — 2-input→1-output functors, exported from `screamer`, each a `FunctorBase<_,2,1>` registered under `EvalOp`. `Sub()(a, b) == a - b` element-wise; `Sub()(aligned_2col) == aligned[:,0] - aligned[:,1]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_arithmetic.py`:

```python
import numpy as np
from screamer import Add, Sub, Mul, Div, combine_latest


def test_arithmetic_scalar():
    assert Add()(2.0, 3.0) == 5.0
    assert Sub()(5.0, 2.0) == 3.0
    assert Mul()(4.0, 2.0) == 8.0
    assert Div()(6.0, 3.0) == 2.0


def test_arithmetic_arrays():
    a = np.arange(1.0, 6.0)
    b = np.arange(10.0, 15.0)
    np.testing.assert_array_equal(Add()(a, b), a + b)
    np.testing.assert_array_equal(Sub()(a, b), a - b)
    np.testing.assert_array_equal(Mul()(a, b), a * b)
    np.testing.assert_array_equal(Div()(a, b), a / b)


def test_sub_over_aligned_columns_TxN():
    # the graph spread idiom: align two series, then a C++ Sub over the columns.
    ka = (np.array([1, 2, 3], dtype=np.int64), np.array([10.0, 30.0, 50.0]))
    kb = (np.array([1, 2, 3], dtype=np.int64), np.array([1.0, 2.0, 3.0]))
    _, aligned = combine_latest(ka, kb)           # (T, 2)
    spread = Sub()(aligned)                        # (T,N) convention -> columns as inputs
    np.testing.assert_array_equal(spread, aligned[:, 0] - aligned[:, 1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_arithmetic.py -v`
Expected: FAIL — `cannot import name 'Add'`.

- [ ] **Step 3: Create the arithmetic functors**

`include/screamer/arithmetic.h`:

```cpp
#ifndef SCREAMER_ARITHMETIC_H
#define SCREAMER_ARITHMETIC_H

#include "screamer/common/functor_base.h"

namespace screamer {

// Binary elementwise arithmetic: 2 inputs -> 1 output. Stateless. These are the
// C++-only reduction vocabulary for computation DAGs (e.g. a price spread is
// Sub()(combine_latest(a, b))).
class Add : public FunctorBase<Add, 2, 1> {
public:
    ResultTuple call(const InputArray& in) override { return in[0] + in[1]; }
};

class Sub : public FunctorBase<Sub, 2, 1> {
public:
    ResultTuple call(const InputArray& in) override { return in[0] - in[1]; }
};

class Mul : public FunctorBase<Mul, 2, 1> {
public:
    ResultTuple call(const InputArray& in) override { return in[0] * in[1]; }
};

class Div : public FunctorBase<Div, 2, 1> {
public:
    ResultTuple call(const InputArray& in) override { return in[0] / in[1]; }
};

} // namespace screamer
#endif
```

- [ ] **Step 4: Bind the functors**

In `bindings/bindings_math.cpp`, add `#include "screamer/arithmetic.h"` and `#include "screamer/common/eval_op.h"`, then near the other 2-input math functors (`Hypot`/`Atan2`) add:

```cpp
    py::class_<screamer::Add, screamer::EvalOp>(m, "Add")
        .def(py::init<>())
        .def("__call__", &screamer::Add::handle_input)
        .def("reset", &screamer::Add::reset, "Reset to the initial state.");

    py::class_<screamer::Sub, screamer::EvalOp>(m, "Sub")
        .def(py::init<>())
        .def("__call__", &screamer::Sub::handle_input)
        .def("reset", &screamer::Sub::reset, "Reset to the initial state.");

    py::class_<screamer::Mul, screamer::EvalOp>(m, "Mul")
        .def(py::init<>())
        .def("__call__", &screamer::Mul::handle_input)
        .def("reset", &screamer::Mul::reset, "Reset to the initial state.");

    py::class_<screamer::Div, screamer::EvalOp>(m, "Div")
        .def(py::init<>())
        .def("__call__", &screamer::Div::handle_input)
        .def("reset", &screamer::Div::reset, "Reset to the initial state.");
```

- [ ] **Step 5: Build, regenerate, run**

Run: `make build && make install-dev && poetry run pytest tests/test_arithmetic.py -v`
Expected: all PASS. `make build` regenerates `__init__.py`, which now exports `Add`/`Sub`/`Mul`/`Div` (they start with uppercase, so the generator picks them up automatically).

- [ ] **Step 6: Full-suite guard**

Run: `poetry run pytest -q`
Expected: green.

- [ ] **Step 7: Commit**

```bash
git add include/screamer/arithmetic.h bindings/bindings_math.cpp screamer/__init__.py tests/test_arithmetic.py
git commit -m "feat(math): Add/Sub/Mul/Div binary functors (graph reduction vocabulary)"
```

---

## Self-Review

**1. Spec coverage (DAG-2a portion):**
- `EvalOp` interface (`n_in`/`n_out`/`eval`/`reset`) → Task 1 (`eval_op.h`). ✓
- `ScreamerBase` implements it → Task 1; `FunctorBase` implements it → Task 2. ✓
- `eval` defined via existing `process_scalar`/`call`; per-event equals batch (tested by `_eval_op` vs `__call__`) → Tasks 1–2. ✓
- Graph can hold any functor behind one pointer → both bases derive `EvalOp`; all functors registered under it (Tasks 1–2) so `py::cast<EvalOp*>` works generically (used by the DAG-2b engine). ✓
- Arithmetic functors `Add`/`Sub`/`Mul`/`Div` + reduction idiom → Task 3. ✓
- Hot path untouched: `eval` is a new virtual called only by the engine/`_eval_op`, never by `operator()`/`handle_input` — no per-call cost on eager scalar/array paths. ✓ (Reviewer should confirm no eager path calls `eval`.)
- Deferred (correctly absent): the graph repr/builder/compiler/drivers (DAG-2b), cardinality push-nodes (DAG-2c).

**2. Placeholder scan:** none — every step has concrete code or an exact command. Task 2 Step 4 uses a grep to enumerate functors rather than listing all ~15 verbatim; the change per class is shown exactly.

**3. Type consistency:** `EvalOp` (`n_in`/`n_out`/`eval`/`reset`), `num_inputs`/`num_outputs` (bound property names), `_eval_op(op, in)->out`, and the `FunctorBase<_,2,1>` arithmetic classes are consistent across tasks. `write_tuple_to_memory` is the existing helper reused for M>1 `eval`.

---

## Follow-on

- **DAG-2b** — the engine: C++ graph representation + builder, generalized `FunctorNode` driving an `EvalOp`, `CombineLatestNode`, fan-out, compiler, batch-replay + streaming drivers, `align_outputs`; thin `dag.py` cutover; DAG-1 `_run` → test oracle.
- **DAG-2c** (later) — `dropna`/`filter`/`split` as C++ push-nodes.
