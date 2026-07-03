# DAG-2b-2 — wide-edge push interface + FunctorNode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the DAG engine's core push-graph plumbing — the wide-edge `Frame`, the `Sink` push interface, and a `FunctorNode` that drives any `EvalOp` at any width — and prove a hand-wired `source → FunctorNode → collector` produces output byte-identical to calling the functor eagerly.

**Architecture:** A new `screamer::dag` C++ subsystem, distinct from the `streams` combinator layer. Its edges carry a *value span* (`Frame{key, const double* values, width}`) pointing at the emitter's reused buffer — zero per-event allocation. `FunctorNode` wraps one `EvalOp` (any arity, from DAG-2a/2b-1), evaluating one frame at a time into its own reused output buffer and emitting a frame downstream. This increment hand-wires and tests the plumbing; the graph builder/compiler and combinator nodes come in later increments.

**Tech Stack:** C++17, pybind11, numpy, pytest.

## Global Constraints

- **`dag` is its own subsystem** (namespace `screamer::dag`), separate from `streams`. It reuses the operator cores (`EvalOp`) but has its own wide-edge plumbing — one plumbing does not serve two jobs.
- **Wide-edge, zero per-event allocation:** `Frame::values` points at the *emitter's* reused buffer, valid only during the synchronous `push` call. Nodes own their output buffer and reuse it every event.
- **One responsibility per type:** `Frame` (an edge event), `Sink` (receive a frame), `FunctorNode` (drive one `EvalOp`). No type does two jobs.
- **Width contract:** a `FunctorNode` requires `frame.width == op.n_in()`; it emits `width == op.n_out()`. A mismatch throws with a clear message.
- **Byte-identity to eager:** per-event `eval` through the node equals the eager functor over the array (foundation identity). No reset inside the driver (streaming semantics); tests use fresh functor instances.
- Compute functors' math never modified. Never hand-edit `screamer/__init__.py` or version files.
- Build: `make build` **then `make install-dev`**. Tests: `poetry run pytest tests/test_dag_engine.py -v`.

---

## File Structure

- `include/screamer/dag/frame.h` (create) — `Frame<Key>` + `Sink<Key>`.
- `include/screamer/dag/functor_node.h` (create) — `FunctorNode<Key>`.
- `include/screamer/dag/collector.h` (create) — `Collector<Key>` (terminal sink → output buffer).
- `include/screamer/dag/driver.h` (create) — `replay_batch<Key>` (arrays → frames → sink).
- `bindings/bindings_dag.cpp` (create) — `_run_functor_batch` test entry; `init_bindings_dag`.
- `bindings/bindings.cpp` (modify) — register `init_bindings_dag`.
- `tests/test_dag_engine.py` (create) — byte-identity tests (1→1, 2→1, 2→2).

---

### Task 1: `Frame`/`Sink`/`FunctorNode` + hand-wired batch, byte-identical to eager

**Files:**
- Create: `include/screamer/dag/frame.h`, `include/screamer/dag/functor_node.h`, `include/screamer/dag/collector.h`, `include/screamer/dag/driver.h`
- Create: `bindings/bindings_dag.cpp`
- Modify: `bindings/bindings.cpp`
- Test: `tests/test_dag_engine.py`

**Interfaces:**
- Consumes: `screamer::EvalOp` (`n_in`/`n_out`/`eval`, from DAG-2a; every functor registered under it, DAG-2b-1).
- Produces:
  - C++: `dag::Frame<Key>{Key key; const double* values; size_t width;}`; `dag::Sink<Key>::push(const Frame<Key>&)`/`flush()`; `dag::FunctorNode<Key>(EvalOp&, Sink<Key>&)`; `dag::Collector<Key>(double* out, size_t width)`; `dag::replay_batch<Key>(const Key*, const double*, size_t T, size_t width, Sink<Key>&)`.
  - Python: `screamer_bindings._run_functor_batch(op, keys_int64, values) -> np.ndarray` of shape `(T, op.num_outputs)`; `values` is `(T,)` (width 1) or `(T, W)` (width W == op.num_inputs).

- [ ] **Step 1: Write the failing test**

Create `tests/test_dag_engine.py`:

```python
import numpy as np
from screamer import RollingMean, Sub, Cart2Polar
from screamer import screamer_bindings as _b


def test_engine_1in_1out_matches_eager():
    x = np.random.default_rng(0).standard_normal(200)
    keys = np.arange(x.size, dtype=np.int64)
    out = _b._run_functor_batch(RollingMean(5), keys, x)     # width 1 -> 1
    assert out.shape == (200, 1)
    np.testing.assert_array_equal(out.reshape(-1), RollingMean(5)(x))


def test_engine_2in_1out_aligned_matches_eager():
    a = np.random.default_rng(1).standard_normal(200)
    b = np.random.default_rng(2).standard_normal(200)
    aligned = np.ascontiguousarray(np.column_stack([a, b]))   # (200, 2)
    keys = np.arange(200, dtype=np.int64)
    out = _b._run_functor_batch(Sub(), keys, aligned)         # width 2 -> 1
    np.testing.assert_array_equal(out.reshape(-1), a - b)


def test_engine_2in_2out_matches_eager():
    xy = np.ascontiguousarray(np.random.default_rng(3).standard_normal((50, 2)))
    keys = np.arange(50, dtype=np.int64)
    out = _b._run_functor_batch(Cart2Polar(), keys, xy)       # width 2 -> 2
    assert out.shape == (50, 2)
    exp = Cart2Polar()(xy[:, 0], xy[:, 1])                    # (50, 2)
    np.testing.assert_array_equal(out, exp)


def test_engine_width_mismatch_raises():
    x = np.random.default_rng(4).standard_normal(10)          # width 1
    keys = np.arange(10, dtype=np.int64)
    try:
        _b._run_functor_batch(Sub(), keys, x)                 # Sub needs width 2
        assert False, "expected a width-mismatch error"
    except Exception:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_dag_engine.py -v`
Expected: FAIL — `_run_functor_batch` undefined.

- [ ] **Step 3: Create the frame + sink**

`include/screamer/dag/frame.h`:

```cpp
#ifndef SCREAMER_DAG_FRAME_H
#define SCREAMER_DAG_FRAME_H

#include <cstddef>

namespace screamer { namespace dag {

// One event on a graph edge. `values` points at the EMITTER's reused buffer and
// is valid only for the duration of the synchronous push() call — a consumer
// reads it immediately and does not retain the pointer. `width` is the number
// of doubles (1 for a normal functor, N for an aligned combine_latest, M for a
// multi-output functor).
template <class Key>
struct Frame {
    Key key;
    const double* values;
    std::size_t width;
};

// Receives frames. One method, one job.
template <class Key>
struct Sink {
    virtual ~Sink() = default;
    virtual void push(const Frame<Key>& f) = 0;
    virtual void flush() {}
};

}} // namespace screamer::dag
#endif
```

- [ ] **Step 4: Create the functor node**

`include/screamer/dag/functor_node.h`:

```cpp
#ifndef SCREAMER_DAG_FUNCTOR_NODE_H
#define SCREAMER_DAG_FUNCTOR_NODE_H

#include <stdexcept>
#include <vector>
#include "screamer/common/eval_op.h"
#include "screamer/dag/frame.h"

namespace screamer { namespace dag {

// Drives exactly one EvalOp. On each frame it evaluates op into its OWN reused
// output buffer and emits a frame downstream. Shape-preserving: key passes
// through; output width is op.n_out().
template <class Key>
class FunctorNode : public Sink<Key> {
public:
    FunctorNode(EvalOp& op, Sink<Key>& downstream)
        : op_(op), downstream_(downstream), out_(op.n_out()) {}

    void push(const Frame<Key>& f) override {
        if (f.width != op_.n_in()) {
            throw std::runtime_error(
                "dag::FunctorNode: frame width does not match op n_in");
        }
        op_.eval(f.values, out_.data());
        downstream_.push(Frame<Key>{f.key, out_.data(), out_.size()});
    }

    void flush() override { downstream_.flush(); }

private:
    EvalOp& op_;
    Sink<Key>& downstream_;
    std::vector<double> out_;   // reused every event; zero per-event allocation
};

}} // namespace screamer::dag
#endif
```

- [ ] **Step 5: Create the collector + driver**

`include/screamer/dag/collector.h`:

```cpp
#ifndef SCREAMER_DAG_COLLECTOR_H
#define SCREAMER_DAG_COLLECTOR_H

#include <cstddef>
#include "screamer/dag/frame.h"

namespace screamer { namespace dag {

// Terminal sink: writes each frame's `width` values into a row-major (T, width)
// output buffer.
template <class Key>
class Collector : public Sink<Key> {
public:
    Collector(double* out, std::size_t width) : out_(out), width_(width), n_(0) {}

    void push(const Frame<Key>& f) override {
        for (std::size_t j = 0; j < f.width; ++j) out_[n_ * width_ + j] = f.values[j];
        ++n_;
    }

    std::size_t count() const { return n_; }

private:
    double* out_;
    std::size_t width_;
    std::size_t n_;
};

}} // namespace screamer::dag
#endif
```

`include/screamer/dag/driver.h`:

```cpp
#ifndef SCREAMER_DAG_DRIVER_H
#define SCREAMER_DAG_DRIVER_H

#include <cstddef>
#include "screamer/dag/frame.h"

namespace screamer { namespace dag {

// Batch driver: replay a row-major (T, width) value buffer as T frames (row i is
// values + i*width, pointing directly into the caller's buffer), pushing each
// into the sink. No copies; no per-event allocation.
template <class Key>
void replay_batch(const Key* keys, const double* values,
                  std::size_t T, std::size_t width, Sink<Key>& sink) {
    for (std::size_t i = 0; i < T; ++i) {
        sink.push(Frame<Key>{keys[i], values + i * width, width});
    }
    sink.flush();
}

}} // namespace screamer::dag
#endif
```

- [ ] **Step 6: Create the binding entry**

`bindings/bindings_dag.cpp`:

```cpp
#include <cstdint>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "screamer/common/eval_op.h"
#include "screamer/dag/frame.h"
#include "screamer/dag/functor_node.h"
#include "screamer/dag/collector.h"
#include "screamer/dag/driver.h"

namespace py = pybind11;
using namespace screamer;

// Hand-wire source -> FunctorNode(op) -> collector and run it in batch.
// `values` is (T,) [width 1] or (T, W) [width W]; returns (T, op.n_out()).
static py::array_t<double> run_functor_batch(
        EvalOp& op,
        py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> keys,
        py::array_t<double, py::array::c_style | py::array::forcecast> values) {
    auto vinfo = values.request();
    std::size_t T = static_cast<std::size_t>(vinfo.shape[0]);
    std::size_t width = (vinfo.ndim == 1)
        ? 1u : static_cast<std::size_t>(vinfo.shape[1]);
    if (width != op.n_in()) {
        throw std::runtime_error(
            "run_functor_batch: input width does not match op num_inputs");
    }
    std::size_t out_w = op.n_out();

    py::array_t<double> out({static_cast<py::ssize_t>(T),
                             static_cast<py::ssize_t>(out_w)});

    dag::Collector<std::int64_t> collector(
        static_cast<double*>(out.request().ptr), out_w);
    dag::FunctorNode<std::int64_t> node(op, collector);
    dag::replay_batch<std::int64_t>(
        static_cast<const std::int64_t*>(keys.request().ptr),
        static_cast<const double*>(vinfo.ptr), T, width, node);
    return out;
}

void init_bindings_dag(py::module& m) {
    m.def("_run_functor_batch", &run_functor_batch,
          py::arg("op"), py::arg("keys"), py::arg("values"));
}
```

- [ ] **Step 7: Register the submodule**

In `bindings/bindings.cpp`, add the declaration and call (alongside the others):

```cpp
void init_bindings_dag(py::module& m);   // <-- add near the other declarations
```
```cpp
    init_bindings_dag(m);                 // <-- add inside PYBIND11_MODULE
```

- [ ] **Step 8: Build and run**

Run: `make build && make install-dev && poetry run pytest tests/test_dag_engine.py -v`
Expected: all four tests PASS (1→1, 2→1, 2→2 byte-identical; width-mismatch raises).

- [ ] **Step 9: Full-suite guard**

Run: `poetry run pytest -q`
Expected: green (the new subsystem is additive).

- [ ] **Step 10: Commit**

```bash
git add include/screamer/dag bindings/bindings_dag.cpp bindings/bindings.cpp tests/test_dag_engine.py
git commit -m "feat(dag): wide-edge Frame/Sink + FunctorNode; hand-wired batch == eager"
```

---

## Self-Review

**1. Spec coverage (DAG-2b-2):**
- Wide-edge `Frame` (value span, emitter-owned buffer) + `Sink` push interface → `frame.h`. ✓
- `FunctorNode` drives any `EvalOp` at any width; reused output buffer (zero per-event alloc); width contract enforced → `functor_node.h`, tested (1→1, 2→1, 2→2, mismatch). ✓
- Byte-identity to eager → tests use `assert_array_equal` on fresh instances (no reset in the driver = streaming semantics). ✓
- `dag` is its own subsystem, reuses `EvalOp`, own plumbing → `screamer::dag` namespace. ✓
- Deferred (correctly absent): graph builder/compiler (2b-4), `CombineLatestNode`/fan-out (2b-3), drivers/streaming/cutover (2b-5).

**2. Placeholder scan:** none — every step has concrete code or an exact command.

**3. Type consistency:** `Frame<Key>{key,values,width}`, `Sink<Key>::push/flush`, `FunctorNode<Key>(EvalOp&, Sink<Key>&)`, `Collector<Key>(double*, width)`, `replay_batch<Key>(keys,values,T,width,Sink&)`, and `_run_functor_batch(op, keys, values)->(T, n_out)` are consistent across the files. `n_in`/`n_out` are the `EvalOp` methods from DAG-2a.

---

## Follow-on (DAG-2b, remaining)

- **DAG-2b-3** — `CombineLatestNode` (push fan-in, N input ports, reuses the `CombineLatest` operator, emits a width-N frame) + fan-out/broadcast (a node with multiple downstream sinks).
- **DAG-2b-4** — C++ graph representation + builder + compiler (graph → wired push-graph).
- **DAG-2b-5** — batch-replay + live-streaming drivers + `align_outputs` + thin `dag.py` cutover + DAG-1 `_run` → test oracle.
