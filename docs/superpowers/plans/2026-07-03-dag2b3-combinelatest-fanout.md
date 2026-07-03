# DAG-2b-3 — CombineLatestNode + fan-out Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the DAG engine's aligning fan-in node (`CombineLatestNode`, reusing the existing `CombineLatest` operator) and fan-out (`Broadcast`), and prove align→compute end-to-end in the push graph — `CombineLatestNode → FunctorNode(Sub)` equals the eager `Sub()(combine_latest(a, b))`.

**Architecture:** `CombineLatestNode` exposes N single-value input *ports* (tiny per-index `Sink`s) and, on any port event, updates the reused `CombineLatest` operator and — when warm — emits one **width-N** frame (the aligned row) to its downstream. `Broadcast` is a `Sink` that forwards each frame to many downstreams. Inputs are merged in key order by reusing `streams::MergeSource`, so the node sees events in the same order the batch `combine_latest` does — giving byte-identity. Each type keeps one responsibility.

**Tech Stack:** C++17, pybind11, numpy, pytest.

## Global Constraints

- **Reuse, don't reimplement:** `CombineLatestNode` wraps the existing `screamer::streams::CombineLatest` operator (`on_event`/`latest`); it does not re-derive alignment. Input merging reuses `streams::MergeSource`.
- **Wide-edge, zero per-event allocation:** the emitted aligned frame points at the operator's own `latest()` buffer (stable during the synchronous push). No allocation on the push path.
- **One responsibility per type:** `CombineLatestNode` (align N ports → width-N frame), a nested `Port` (route one input to its index), `Broadcast` (fan one frame to many).
- **Ports carry single values:** each `combine_latest` input is a single-value stream; a port requires `frame.width == 1`.
- **Byte-identity:** `CombineLatestNode` output == the batch `streams.combine_latest` (`emit="when_all"`/`"on_any"`); `CombineLatestNode → FunctorNode(Sub)` == eager `Sub()(combine_latest(a,b))`.
- Compute functors' math never modified. Never hand-edit `screamer/__init__.py` or version files.
- Build: `make build` **then `make install-dev`**. Tests: `poetry run pytest tests/test_dag_combine.py -v`.

---

## File Structure

- `include/screamer/dag/combine_latest_node.h` (create) — `CombineLatestNode<Key>` + nested `Port`.
- `include/screamer/dag/broadcast.h` (create) — `Broadcast<Key>`.
- `bindings/bindings_dag.cpp` (modify) — `_run_combine_latest_batch` and `_run_combine_then_sub_batch` test entries.
- `tests/test_dag_combine.py` (create) — align byte-identity, the align→compute idiom, fan-out.

---

### Task 1: `CombineLatestNode` (aligning fan-in), byte-identical to batch combine_latest

**Files:**
- Create: `include/screamer/dag/combine_latest_node.h`
- Modify: `bindings/bindings_dag.cpp`
- Test: `tests/test_dag_combine.py`

**Interfaces:**
- Consumes: `dag::Frame`/`dag::Sink` (2b-2); `streams::CombineLatest` operator; `streams::MergeSource` + `streams::VectorSource` (foundation) for merging inputs in key order.
- Produces:
  - C++: `dag::CombineLatestNode<Key>(size_t n, bool when_all, Sink<Key>& downstream)`; `Sink<Key>& port(size_t i)`; on any port event (width-1 frame) it updates the operator and, when it fires, emits a width-`n` frame `{key, latest.data(), n}`.
  - Python: `screamer_bindings._run_combine_latest_batch(keys_list, values_list, when_all) -> (out_keys int64, out_aligned (M, N))`, matching `streams.combine_latest`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dag_combine.py`:

```python
import numpy as np
import pytest
from screamer import combine_latest
from screamer import screamer_bindings as _b


def _keys_vals(pairs):
    k = np.array([p[0] for p in pairs], dtype=np.int64)
    v = np.array([p[1] for p in pairs], dtype=np.float64)
    return k, v


@pytest.mark.parametrize("when_all", [True, False])
def test_combine_latest_node_matches_batch(when_all):
    rng = np.random.default_rng(0)
    a_k = np.sort(rng.integers(0, 500, size=120)).astype(np.int64)
    a_v = rng.standard_normal(120)
    b_k = np.sort(rng.integers(0, 500, size=90)).astype(np.int64)
    b_v = rng.standard_normal(90)

    out_k, out_a = _b._run_combine_latest_batch([a_k, b_k], [a_v, b_v], when_all)
    exp_k, exp_a = combine_latest((a_k, a_v), (b_k, b_v),
                                  emit="when_all" if when_all else "on_any")
    np.testing.assert_array_equal(out_k, exp_k)
    np.testing.assert_array_equal(out_a, exp_a)      # NaN==NaN under assert_array_equal


def test_combine_latest_node_three_inputs():
    rng = np.random.default_rng(1)
    series = []
    for _ in range(3):
        k = np.sort(rng.integers(0, 300, size=60)).astype(np.int64)
        v = rng.standard_normal(60)
        series.append((k, v))
    out_k, out_a = _b._run_combine_latest_batch([s[0] for s in series],
                                                [s[1] for s in series], True)
    exp_k, exp_a = combine_latest(*series)
    np.testing.assert_array_equal(out_k, exp_k)
    np.testing.assert_array_equal(out_a, exp_a)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_dag_combine.py -v`
Expected: FAIL — `_run_combine_latest_batch` undefined.

- [ ] **Step 3: Create `CombineLatestNode`**

`include/screamer/dag/combine_latest_node.h`:

```cpp
#ifndef SCREAMER_DAG_COMBINE_LATEST_NODE_H
#define SCREAMER_DAG_COMBINE_LATEST_NODE_H

#include <cassert>
#include <cstddef>
#include <vector>
#include "screamer/dag/frame.h"
#include "screamer/streams/combine_latest.h"

namespace screamer { namespace dag {

// Aligning fan-in node. Exposes N single-value input ports; on any port event it
// updates the reused CombineLatest operator and, when it fires (per when_all),
// emits ONE width-N frame carrying the aligned latest values. The emitted frame
// points at the operator's own latest() buffer (stable during the push).
template <class Key>
class CombineLatestNode {
public:
    CombineLatestNode(std::size_t n, bool when_all, Sink<Key>& downstream)
        : cl_(n, when_all), downstream_(downstream), n_(n) {
        ports_.reserve(n);
        for (std::size_t i = 0; i < n; ++i) ports_.emplace_back(*this, i);
    }

    // Non-movable/copyable: the ports hold a reference back to this node.
    CombineLatestNode(const CombineLatestNode&) = delete;
    CombineLatestNode& operator=(const CombineLatestNode&) = delete;

    Sink<Key>& port(std::size_t i) { return ports_[i]; }

private:
    void on_port(std::size_t i, const Frame<Key>& f) {
        assert(f.width == 1);
        if (cl_.on_event(static_cast<std::uint32_t>(i), f.values[0])) {
            const std::vector<double>& row = cl_.latest();
            downstream_.push(Frame<Key>{f.key, row.data(), n_});
        }
    }

    // A single input port: routes an event to its owning node with its index.
    struct Port : Sink<Key> {
        CombineLatestNode& node;
        std::size_t idx;
        Port(CombineLatestNode& n, std::size_t i) : node(n), idx(i) {}
        void push(const Frame<Key>& f) override { node.on_port(idx, f); }
    };
    friend struct Port;

    screamer::streams::CombineLatest cl_;   // reused operator (no re-derivation)
    Sink<Key>& downstream_;
    std::size_t n_;
    std::vector<Port> ports_;
};

}} // namespace screamer::dag
#endif
```

- [ ] **Step 4: Add the binding entry**

In `bindings/bindings_dag.cpp`, add includes:

```cpp
#include <vector>
#include "screamer/dag/combine_latest_node.h"
#include "screamer/streams/vector_source.h"
#include "screamer/streams/merge_source.h"
#include "screamer/streams/collector_sink.h"   // not needed; remove if unused
```

Add the helper (merge the N inputs in key order via `streams::MergeSource`, driving the node's ports; collect the emitted aligned frames):

```cpp
static py::tuple run_combine_latest_batch(py::list key_arrays,
                                          py::list value_arrays,
                                          bool when_all) {
    std::size_t n = key_arrays.size();
    // Build a VectorSource per input and a MergeSource over them (key order).
    std::vector<py::array_t<std::int64_t>> keys;
    std::vector<py::array_t<double>> vals;
    std::vector<std::unique_ptr<streams::VectorSource<std::int64_t>>> srcs;
    std::vector<streams::Source<std::int64_t>*> child_ptrs;
    std::size_t total = 0;
    for (std::size_t i = 0; i < n; ++i) {
        keys.push_back(py::cast<py::array_t<std::int64_t>>(key_arrays[i]));
        vals.push_back(py::cast<py::array_t<double>>(value_arrays[i]));
        auto ki = keys[i].request(); auto vi = vals[i].request();
        std::size_t len = static_cast<std::size_t>(ki.shape[0]);
        total += len;
        srcs.push_back(std::make_unique<streams::VectorSource<std::int64_t>>(
            static_cast<const std::int64_t*>(ki.ptr),
            static_cast<const double*>(vi.ptr), len));
        child_ptrs.push_back(srcs.back().get());
    }
    streams::MergeSource<std::int64_t> merge(child_ptrs);

    // Collect emitted width-n frames into growable buffers (M <= total).
    std::vector<std::int64_t> out_k;
    std::vector<double> out_v;
    out_k.reserve(total);
    out_v.reserve(total * n);

    struct Gather : dag::Sink<std::int64_t> {
        std::vector<std::int64_t>& k; std::vector<double>& v;
        Gather(std::vector<std::int64_t>& kk, std::vector<double>& vv) : k(kk), v(vv) {}
        void push(const dag::Frame<std::int64_t>& f) override {
            k.push_back(f.key);
            v.insert(v.end(), f.values, f.values + f.width);
        }
    } gather(out_k, out_v);

    dag::CombineLatestNode<std::int64_t> node(n, when_all, gather);
    double one;
    while (auto e = merge.next()) {
        one = e->value;
        dag::Frame<std::int64_t> f{e->key, &one, 1};
        node.port(e->source).push(f);
    }

    std::size_t m = out_k.size();
    py::array_t<std::int64_t> rk(static_cast<py::ssize_t>(m));
    if (m) std::memcpy(rk.request().ptr, out_k.data(), m * sizeof(std::int64_t));
    py::array_t<double> rv({static_cast<py::ssize_t>(m), static_cast<py::ssize_t>(n)});
    if (m) std::memcpy(rv.request().ptr, out_v.data(), m * n * sizeof(double));
    return py::make_tuple(rk, rv);
}
```

Register in `init_bindings_dag`:

```cpp
    m.def("_run_combine_latest_batch", &run_combine_latest_batch,
          py::arg("key_arrays"), py::arg("value_arrays"), py::arg("when_all"));
```

Ensure `<cstring>`, `<memory>`, `<vector>` are included in `bindings_dag.cpp`.

- [ ] **Step 5: Build and run**

Run: `make build && make install-dev && poetry run pytest tests/test_dag_combine.py -v`
Expected: all PASS — the node's aligned output equals the batch `combine_latest` for `when_all`, `on_any`, and 3 inputs.

- [ ] **Step 6: Full-suite guard**

Run: `poetry run pytest -q`
Expected: green.

- [ ] **Step 7: Commit**

```bash
git add include/screamer/dag/combine_latest_node.h bindings/bindings_dag.cpp tests/test_dag_combine.py
git commit -m "feat(dag): CombineLatestNode (aligning fan-in) == batch combine_latest"
```

---

### Task 2: `Broadcast` fan-out + the align→compute idiom

**Files:**
- Create: `include/screamer/dag/broadcast.h`
- Modify: `bindings/bindings_dag.cpp`
- Test: `tests/test_dag_combine.py`

**Interfaces:**
- Consumes: `CombineLatestNode` (Task 1), `FunctorNode` (2b-2), `Sub` (arithmetic, DAG-2a).
- Produces:
  - C++: `dag::Broadcast<Key>` — `void add(Sink<Key>&)`, `push` forwards to all, `flush` forwards to all.
  - Python: `screamer_bindings._run_combine_then_sub_batch(keys_list, values_list, when_all) -> (out_keys, out_spread)` — wires `CombineLatestNode(2) → FunctorNode(Sub) → collector`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dag_combine.py`:

```python
from screamer import Sub


def test_combine_then_sub_matches_eager():
    # align a,b then a C++ Sub over the width-2 frame == eager Sub()(combine_latest)
    rng = np.random.default_rng(2)
    a_k = np.sort(rng.integers(0, 500, size=100)).astype(np.int64)
    a_v = rng.standard_normal(100)
    b_k = np.sort(rng.integers(0, 500, size=100)).astype(np.int64)
    b_v = rng.standard_normal(100)

    out_k, spread = _b._run_combine_then_sub_batch([a_k, b_k], [a_v, b_v], True)
    exp_k, aligned = combine_latest((a_k, a_v), (b_k, b_v))     # when_all
    np.testing.assert_array_equal(out_k, exp_k)
    np.testing.assert_array_equal(spread.reshape(-1), aligned[:, 0] - aligned[:, 1])


def test_broadcast_fans_out():
    # A width-2 aligned frame delivered to two collectors is identical.
    rng = np.random.default_rng(3)
    a_k = np.sort(rng.integers(0, 200, size=50)).astype(np.int64)
    a_v = rng.standard_normal(50)
    b_k = np.sort(rng.integers(0, 200, size=50)).astype(np.int64)
    b_v = rng.standard_normal(50)
    (k1, a1), (k2, a2) = _b._run_combine_latest_fanout([a_k, b_k], [a_v, b_v], True)
    np.testing.assert_array_equal(k1, k2)
    np.testing.assert_array_equal(a1, a2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_dag_combine.py -k "then_sub or fans_out" -v`
Expected: FAIL — the two entries are undefined.

- [ ] **Step 3: Create `Broadcast`**

`include/screamer/dag/broadcast.h`:

```cpp
#ifndef SCREAMER_DAG_BROADCAST_H
#define SCREAMER_DAG_BROADCAST_H

#include <vector>
#include "screamer/dag/frame.h"

namespace screamer { namespace dag {

// Fan-out: forwards each frame to every registered downstream sink. One job.
template <class Key>
class Broadcast : public Sink<Key> {
public:
    void add(Sink<Key>& s) { sinks_.push_back(&s); }
    void push(const Frame<Key>& f) override { for (auto* s : sinks_) s->push(f); }
    void flush() override { for (auto* s : sinks_) s->flush(); }
private:
    std::vector<Sink<Key>*> sinks_;
};

}} // namespace screamer::dag
#endif
```

- [ ] **Step 4: Add the two binding entries**

In `bindings/bindings_dag.cpp`, add `#include "screamer/dag/broadcast.h"` and `#include "screamer/dag/functor_node.h"` (if not already), then factor a small `merge_drive` lambda-or-helper that runs the merge loop pushing width-1 frames to a `CombineLatestNode`'s ports (reuse the loop from Task 1). Add:

```cpp
// CombineLatestNode(2) -> FunctorNode(Sub) -> collector.
static py::tuple run_combine_then_sub_batch(py::list key_arrays,
                                            py::list value_arrays,
                                            bool when_all) {
    // ... build merge over the inputs exactly as run_combine_latest_batch ...
    // Gather into out_k / out_spread (width 1):
    std::vector<std::int64_t> out_k; std::vector<double> out_v;
    struct Gather1 : dag::Sink<std::int64_t> {
        std::vector<std::int64_t>& k; std::vector<double>& v;
        Gather1(std::vector<std::int64_t>& kk, std::vector<double>& vv) : k(kk), v(vv) {}
        void push(const dag::Frame<std::int64_t>& f) override {
            k.push_back(f.key); v.push_back(f.values[0]);
        }
    } gather(out_k, out_v);

    screamer::Sub sub;                                   // 2->1 EvalOp
    dag::FunctorNode<std::int64_t> sub_node(sub, gather);
    dag::CombineLatestNode<std::int64_t> node(key_arrays.size(), when_all, sub_node);
    // drive the merge loop pushing width-1 frames to node.port(source) ...
    // then marshal out_k (M,) and out_v (M,1) like Task 1.
}
```

Provide the full merge-drive + marshalling by reusing the Task-1 pattern (extract the `MergeSource`-building and drive-loop into a static helper `drive_ports(key_arrays, value_arrays, CombineLatestNode&)` to avoid duplicating the loop across the three entries — DRY). The `run_combine_latest_fanout` entry wires `CombineLatestNode → Broadcast → [Gather A, Gather B]` and returns both gathers' `(keys, aligned)` as a tuple of two tuples.

- [ ] **Step 5: Build and run**

Run: `make build && make install-dev && poetry run pytest tests/test_dag_combine.py -v`
Expected: all PASS — the align→compute idiom equals eager `Sub()(combine_latest)`, and fan-out delivers identical frames to both consumers.

- [ ] **Step 6: Full-suite guard + commit**

Run: `poetry run pytest -q` (green), then:

```bash
git add include/screamer/dag/broadcast.h bindings/bindings_dag.cpp tests/test_dag_combine.py
git commit -m "feat(dag): Broadcast fan-out; CombineLatestNode->Sub == eager"
```

---

## Self-Review

**1. Spec coverage (DAG-2b-3):**
- `CombineLatestNode` push fan-in, reuses the `CombineLatest` operator, emits width-N aligned frame → Task 1, byte-identical to batch `combine_latest` (when_all/on_any/3-input). ✓
- Ports (single-value inputs, routed by index) → nested `Port`. ✓
- Fan-out `Broadcast` → Task 2. ✓
- Align→compute end-to-end (`CombineLatestNode → FunctorNode(Sub)` == eager) → Task 2. ✓
- Zero per-event allocation: emitted frame points at the operator's `latest()` buffer; growable output vectors are per-call (reserve to `total`), not per-event. ✓
- Reuse (no re-derivation): `streams::CombineLatest` + `streams::MergeSource`. ✓
- Deferred (correctly absent): graph builder/compiler (2b-4), drivers/streaming/cutover (2b-5).

**2. Placeholder scan:** Task 2 Step 4 sketches the two entries and instructs extracting a shared `drive_ports` helper to keep the merge-drive loop DRY across the three entries — the pattern is fully shown in Task 1; the implementer factors it once. No TBDs.

**3. Type consistency:** `CombineLatestNode<Key>(n, when_all, Sink&)`/`.port(i)`, `Broadcast<Key>::add/push/flush`, `Frame`/`Sink` (2b-2), `streams::CombineLatest`/`MergeSource`/`VectorSource`, and `Sub` (EvalOp) are consistent. `_run_combine_latest_batch`/`_run_combine_then_sub_batch`/`_run_combine_latest_fanout` return shapes match the tests.

---

## Follow-on (DAG-2b, remaining)

- **DAG-2b-4** — C++ graph representation + builder + compiler (graph → wired push-graph of these nodes).
- **DAG-2b-5** — batch-replay + live-streaming drivers + `align_outputs` + thin `dag.py` cutover + DAG-1 `_run` → test oracle.
