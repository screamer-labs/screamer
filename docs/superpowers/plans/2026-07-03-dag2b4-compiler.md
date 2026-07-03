# DAG-2b-4 — graph representation + builder + compiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A C++ graph representation (`GraphSpec`) + builder + compiler that wires a graph of functor/combine nodes into a runnable push-graph, executed in batch, producing output byte-identical to the equivalent eager expression.

**Architecture:** One compile rule: process nodes in **reverse-topological order** (consumers before producers). Each node exposes its **input-sink(s)** — a `FunctorNode` is the sink for its one wide edge; a `CombineLatestNode` exposes N **ports**. To wire a producer: resolve its consumers' input-sinks, wrap in a `Broadcast` if >1 (fan-out), add a `Collector` if it's an output, and construct the producer with that downstream. Inputs are width-1 sources routed by `MergeSource`. That rule handles chains, fan-out, multi-input, and multi-output uniformly.

**Tech Stack:** C++17, pybind11, numpy, pytest.

## Global Constraints

- **One compile rule, no special cases:** reverse-topo build + input-sink resolution + `Broadcast` for fan-out + `Collector` for outputs.
- **Clear responsibilities:** `GraphSpec` (pure data), `GraphBuilder` (accumulate spec), `CompiledGraph` (owns wired nodes + runs batch), `compile()` (spec → CompiledGraph). No type does two jobs.
- **`CompiledGraph` is pure C++** (no pybind types in its interface) — WASM-ready; the binding marshals arrays.
- **Edge-width contract:** an Input is width-1; a `FunctorNode`'s single input edge has width == its `n_in` (a 2-input functor is fed by a `CombineLatestNode`, not two inputs directly); a `CombineLatestNode` port is width-1.
- **Byte-identity:** a compiled graph's batch output == the eager expression it encodes (`np.testing.assert_array_equal`).
- Zero per-event allocation on the push path (reuse 2b-2/2b-3 nodes). Compute math never modified. Never hand-edit `screamer/__init__.py` or version files.
- Build: `make build` **then `make install-dev`**. Tests: `poetry run pytest tests/test_dag_compile.py -v`.

---

## File Structure

- `include/screamer/dag/graph.h` (create) — `NodeKind`, `NodeSpec`, `GraphSpec`, `GraphBuilder`.
- `include/screamer/dag/compiled_graph.h` (create) — `CompiledGraph` + `compile(const GraphSpec&)`.
- `bindings/bindings_dag.cpp` (modify) — bind `GraphBuilder`/`CompiledGraph` (`_GraphBuilder`) + `run_batch`.
- `tests/test_dag_compile.py` (create) — chains (T1), fan-out + multi-output (T2), combine + multi-input (T3).

---

### Task 1: `GraphSpec` + compiler for chains (the wiring skeleton)

**Files:**
- Create: `include/screamer/dag/graph.h`, `include/screamer/dag/compiled_graph.h`
- Modify: `bindings/bindings_dag.cpp`
- Test: `tests/test_dag_compile.py`

**Interfaces:**
- Consumes: `dag::FunctorNode`, `dag::Frame`/`Sink` (2b-2); `streams::VectorSource`/`MergeSource`; `EvalOp`.
- Produces:
  - C++: `dag::GraphSpec` (data); `dag::GraphBuilder` (`add_input()->id`, `add_functor(EvalOp*, inputs)->id`, `set_outputs(ids)`); `dag::CompiledGraph` with `run_batch(const std::vector<const std::int64_t*>& in_keys, const std::vector<const double*>& in_vals, const std::vector<std::size_t>& in_lens)` returning, per output, `(std::vector<std::int64_t> keys, std::vector<double> values, std::size_t width)`; `dag::compile(const GraphSpec&) -> CompiledGraph`.
  - Python: `_GraphBuilder` with `add_input()`, `add_functor(op, [input_ids])`, `set_outputs([ids])`, `run_batch([(keys,values), ...]) -> [(keys, values2d), ...]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dag_compile.py`:

```python
import numpy as np
from screamer import RollingMean, Diff
from screamer import screamer_bindings as _b


def _row(v):
    v = np.ascontiguousarray(v, dtype=np.float64)
    return np.arange(v.size, dtype=np.int64), v


def test_compile_chain_equals_eager():
    x = np.random.default_rng(0).standard_normal(200)
    g = _b._GraphBuilder()
    xi = g.add_input()
    a = g.add_functor(RollingMean(5), [xi])
    b = g.add_functor(Diff(1), [a])
    g.set_outputs([b])
    (out_k, out_v), = g.run_batch([_row(x)])
    exp = Diff(1)(RollingMean(5)(x))
    np.testing.assert_array_equal(out_v.reshape(-1), exp)
    np.testing.assert_array_equal(out_k, np.arange(x.size, dtype=np.int64))


def test_compile_single_functor():
    x = np.random.default_rng(1).standard_normal(50)
    g = _b._GraphBuilder()
    xi = g.add_input()
    g.set_outputs([g.add_functor(RollingMean(3), [xi])])
    (out_k, out_v), = g.run_batch([_row(x)])
    np.testing.assert_array_equal(out_v.reshape(-1), RollingMean(3)(x))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_dag_compile.py -v`
Expected: FAIL — `_GraphBuilder` undefined.

- [ ] **Step 3: Create the graph spec + builder**

`include/screamer/dag/graph.h`:

```cpp
#ifndef SCREAMER_DAG_GRAPH_H
#define SCREAMER_DAG_GRAPH_H

#include <cstddef>
#include <vector>
#include "screamer/common/eval_op.h"

namespace screamer { namespace dag {

enum class NodeKind { Input, Functor, CombineLatest };

// Pure data: one node of a graph definition.
struct NodeSpec {
    NodeKind kind;
    EvalOp* op = nullptr;                 // Functor only
    bool when_all = true;                 // CombineLatest only
    std::vector<std::size_t> inputs;      // producer node ids (edges into this node)
};

struct GraphSpec {
    std::vector<NodeSpec> nodes;
    std::vector<std::size_t> input_ids;   // Input nodes, in signature order
    std::vector<std::size_t> output_ids;  // output nodes, in order
};

// Accumulates a GraphSpec; returns node ids.
class GraphBuilder {
public:
    std::size_t add_input() {
        spec_.nodes.push_back(NodeSpec{NodeKind::Input, nullptr, true, {}});
        std::size_t id = spec_.nodes.size() - 1;
        spec_.input_ids.push_back(id);
        return id;
    }
    std::size_t add_functor(EvalOp* op, std::vector<std::size_t> inputs) {
        spec_.nodes.push_back(NodeSpec{NodeKind::Functor, op, true, std::move(inputs)});
        return spec_.nodes.size() - 1;
    }
    std::size_t add_combine_latest(std::vector<std::size_t> inputs, bool when_all) {
        spec_.nodes.push_back(NodeSpec{NodeKind::CombineLatest, nullptr, when_all,
                                       std::move(inputs)});
        return spec_.nodes.size() - 1;
    }
    void set_outputs(std::vector<std::size_t> outs) { spec_.output_ids = std::move(outs); }
    const GraphSpec& spec() const { return spec_; }

private:
    GraphSpec spec_;
};

}} // namespace screamer::dag
#endif
```

- [ ] **Step 4: Create the compiler + compiled graph**

`include/screamer/dag/compiled_graph.h` (Task 1 handles `Input` and `Functor` node kinds; `CombineLatest` is added in Task 3):

```cpp
#ifndef SCREAMER_DAG_COMPILED_GRAPH_H
#define SCREAMER_DAG_COMPILED_GRAPH_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>
#include "screamer/dag/frame.h"
#include "screamer/dag/functor_node.h"
#include "screamer/dag/broadcast.h"
#include "screamer/dag/collector.h"
#include "screamer/streams/vector_source.h"
#include "screamer/streams/merge_source.h"

namespace screamer { namespace dag {

// One output stream gathered during a batch run.
struct OutputBuffer {
    std::vector<std::int64_t> keys;
    std::vector<double> values;   // row-major, width columns per row
    std::size_t width = 1;
};

// A gathering terminal sink writing into an OutputBuffer.
class GatherSink : public Sink<std::int64_t> {
public:
    explicit GatherSink(OutputBuffer& buf) : buf_(buf) {}
    void push(const Frame<std::int64_t>& f) override {
        buf_.width = f.width;
        buf_.keys.push_back(f.key);
        buf_.values.insert(buf_.values.end(), f.values, f.values + f.width);
    }
private:
    OutputBuffer& buf_;
};

class CompiledGraph {
public:
    // in_* are per-input arrays (row-number keys or explicit keys). Returns one
    // OutputBuffer per output, in output order.
    std::vector<OutputBuffer> run_batch(
            const std::vector<const std::int64_t*>& in_keys,
            const std::vector<const double*>& in_vals,
            const std::vector<std::size_t>& in_lens) {
        // reset stateful nodes so a graph can be run more than once
        for (auto& op : reset_ops_) op->reset();

        outputs_.assign(output_widths_.size(), OutputBuffer{});
        for (std::size_t o = 0; o < outputs_.size(); ++o)
            outputs_[o].width = output_widths_[o];

        // wire the gather sinks now (they reference outputs_)
        gathers_.clear();
        for (std::size_t o = 0; o < outputs_.size(); ++o)
            gathers_.push_back(std::make_unique<GatherSink>(outputs_[o]));
        // (re)point output collectors at fresh gathers
        rewire_output_gathers();

        // drive: merge all inputs by key, route each event to its input's sink
        std::vector<std::unique_ptr<streams::VectorSource<std::int64_t>>> srcs;
        std::vector<streams::Source<std::int64_t>*> child;
        for (std::size_t i = 0; i < in_keys.size(); ++i) {
            srcs.push_back(std::make_unique<streams::VectorSource<std::int64_t>>(
                in_keys[i], in_vals[i], in_lens[i]));
            child.push_back(srcs.back().get());
        }
        streams::MergeSource<std::int64_t> merge(child);
        double one;
        while (auto e = merge.next()) {
            one = e->value;
            Frame<std::int64_t> f{e->key, &one, 1};
            input_sinks_[e->source]->push(f);
        }
        for (auto* s : input_sinks_) s->flush();
        return std::move(outputs_);
    }

    friend CompiledGraph compile(const GraphSpec& spec);

private:
    void rewire_output_gathers();   // defined below compile()

    // Ownership of everything the wired graph needs, alive for run_batch.
    std::vector<std::shared_ptr<void>> owned_;    // functor nodes, broadcasts, ...
    std::vector<EvalOp*> reset_ops_;              // stateful functor ops to reset
    std::vector<Sink<std::int64_t>*> input_sinks_; // per input id (signature order)
    std::vector<std::size_t> output_widths_;
    // output collectors are Broadcast-or-direct sinks whose terminal gather is
    // swapped each run; we record the per-output "attach point".
    std::vector<class OutputAttach*> output_attach_;
    std::vector<std::unique_ptr<GatherSink>> gathers_;
    std::vector<OutputBuffer> outputs_;
};

}} // namespace screamer::dag
#endif
```

**NOTE for the implementer:** the sketch above shows the intended data model. Implement `compile()` as follows (place it in the same header, after the class):

1. Compute, for each node id, its list of **consumer input-sinks**. A consumer edge `(j, slot)` means `nodes[j].inputs[slot] == i`. The input-sink for that edge is: if `nodes[j]` is a `Functor`, node j's `FunctorNode` itself; (CombineLatest ports come in Task 3). If node i is in `output_ids`, add an output attach point.
2. Reverse-topological order (Kahn on reversed edges, or DFS post-order reversed) so every consumer is built before node i.
3. For each node i in that order, gather its downstream sinks (consumer input-sinks resolved in step 1 + its output attach if any). If exactly one → downstream = it; if more → make a `Broadcast` (own it), add all, downstream = the broadcast.
   - `Functor`: `auto n = std::make_shared<FunctorNode<int64_t>>(*op, downstream); owned_.push_back(n);` record node i's input-sink = `n.get()`; push `op` into `reset_ops_`.
   - `Input`: no node object; record `input_sinks_[signature_index(i)] = downstream`.
   - `output width` of a node = functor `op->n_out()` (CombineLatest: its input count — Task 3).
4. For each output id, its width → `output_widths_`, and record its output attach point so `rewire_output_gathers()` can point it at a fresh `GatherSink` each run. (Simplest attach: give each output its own dedicated `Broadcast` created at compile time whose sink list is cleared/re-added each run — or, cleaner, make the output attach a small indirection sink that forwards to the current gather. Pick the simplest that lets `run_batch` be called repeatedly; a single-run implementation that rebuilds on each call is also acceptable for Task 1 — see the simplification note.)

**Simplification permitted for Task 1:** if repeated `run_batch` calls add complexity, `compile()` may store the `GraphSpec` and (re)build the wired push-graph fresh inside each `run_batch` call (the graph is small; building is cheap relative to the event loop). This keeps the wiring logic in one place and avoids the gather-reattachment machinery. Prefer this if it is clearer — clarity over micro-optimization here. Document the choice.

- [ ] **Step 5: Bind the builder**

In `bindings/bindings_dag.cpp`, add `#include "screamer/dag/graph.h"` and `#include "screamer/dag/compiled_graph.h"`, then bind a Python-facing builder that accepts functor objects (as `EvalOp*`) and numpy arrays:

```cpp
py::class_<dag::GraphBuilder>(m, "_GraphBuilder")
    .def(py::init<>())
    .def("add_input", &dag::GraphBuilder::add_input)
    .def("add_functor", [](dag::GraphBuilder& b, EvalOp* op,
                           std::vector<std::size_t> inputs) {
        return b.add_functor(op, std::move(inputs));
    })
    .def("add_combine_latest", [](dag::GraphBuilder& b,
                                  std::vector<std::size_t> inputs, bool when_all) {
        return b.add_combine_latest(std::move(inputs), when_all);
    })
    .def("set_outputs", &dag::GraphBuilder::set_outputs)
    .def("run_batch", [](dag::GraphBuilder& b, py::list feeds) {
        // marshal feeds -> raw spans; compile; run; marshal outputs back
        std::vector<py::array_t<std::int64_t>> ks;
        std::vector<py::array_t<double>> vs;
        std::vector<const std::int64_t*> kp; std::vector<const double*> vp;
        std::vector<std::size_t> lens;
        for (auto item : feeds) {
            auto t = py::cast<py::tuple>(item);
            ks.push_back(py::cast<py::array_t<std::int64_t,
                         py::array::c_style | py::array::forcecast>>(t[0]));
            vs.push_back(py::cast<py::array_t<double,
                         py::array::c_style | py::array::forcecast>>(t[1]));
            kp.push_back(static_cast<const std::int64_t*>(ks.back().request().ptr));
            vp.push_back(static_cast<const double*>(vs.back().request().ptr));
            lens.push_back(static_cast<std::size_t>(vs.back().request().shape[0]));
        }
        dag::CompiledGraph g = dag::compile(b.spec());
        std::vector<dag::OutputBuffer> outs = g.run_batch(kp, vp, lens);
        py::list result;
        for (auto& o : outs) {
            std::size_t m = o.keys.size();
            py::array_t<std::int64_t> ok(static_cast<py::ssize_t>(m));
            if (m) std::memcpy(ok.request().ptr, o.keys.data(), m * sizeof(std::int64_t));
            py::array_t<double> ov({static_cast<py::ssize_t>(m),
                                    static_cast<py::ssize_t>(o.width)});
            if (m) std::memcpy(ov.request().ptr, o.values.data(),
                               o.values.size() * sizeof(double));
            result.append(py::make_tuple(ok, ov));
        }
        return result;
    });
```

(Because every functor is registered under `EvalOp` (2b-1), `EvalOp* op` casts from any functor object.)

- [ ] **Step 6: Build and run**

Run: `make build && make install-dev && poetry run pytest tests/test_dag_compile.py -v`
Expected: both chain tests PASS.

- [ ] **Step 7: Full-suite guard + commit**

Run: `poetry run pytest -q` (green), then:

```bash
git add include/screamer/dag/graph.h include/screamer/dag/compiled_graph.h bindings/bindings_dag.cpp tests/test_dag_compile.py
git commit -m "feat(dag): GraphSpec/builder + compiler (chains) == eager"
```

---

### Task 2: fan-out (`Broadcast` wiring) + multi-output

**Files:**
- Modify: `include/screamer/dag/compiled_graph.h` (only if Task 1 didn't already handle >1 consumer / >1 output)
- Test: `tests/test_dag_compile.py`

**Interfaces:**
- Consumes: Task 1's compiler + `dag::Broadcast`.
- Produces: a shared intermediate feeding multiple consumers is evaluated once and broadcast; a graph with multiple outputs returns all, each its own stream.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dag_compile.py`:

```python
def test_compile_fanout_and_multi_output():
    x = np.random.default_rng(2).standard_normal(200)
    g = _b._GraphBuilder()
    xi = g.add_input()
    shared = g.add_functor(RollingMean(5), [xi])   # one node...
    d = g.add_functor(Diff(1), [shared])           # ...two consumers
    m = g.add_functor(RollingMean(3), [shared])
    g.set_outputs([d, m])
    (dk, dv), (mk, mv) = g.run_batch([_row(x)])
    sm = RollingMean(5)(x)
    np.testing.assert_array_equal(dv.reshape(-1), Diff(1)(sm))
    np.testing.assert_array_equal(mv.reshape(-1), RollingMean(3)(sm))


def test_compile_output_is_also_intermediate():
    x = np.random.default_rng(3).standard_normal(100)
    g = _b._GraphBuilder()
    xi = g.add_input()
    a = g.add_functor(RollingMean(4), [xi])        # both an output AND feeds b
    b = g.add_functor(Diff(1), [a])
    g.set_outputs([a, b])
    (ak, av), (bk, bv) = g.run_batch([_row(x)])
    np.testing.assert_array_equal(av.reshape(-1), RollingMean(4)(x))
    np.testing.assert_array_equal(bv.reshape(-1), Diff(1)(RollingMean(4)(x)))
```

- [ ] **Step 2: Run test to verify it fails (or passes)**

Run: `poetry run pytest tests/test_dag_compile.py -k "fanout or also_intermediate" -v`
Expected: If Task 1's compiler already builds a `Broadcast` when a node has >1 downstream sink (consumers + output attach), these PASS immediately — then this task is verification-only, and you add the tests as regression guards. If they FAIL (e.g. only the first consumer got wired, or a node that is both an output and an intermediate dropped one), fix the downstream-collection in `compile()` so it gathers ALL consumer sinks plus the output attach and wraps them in a `Broadcast` when the count exceeds 1.

- [ ] **Step 3: Fix if needed**

Ensure `compile()` computes each node's downstream as *the full set* of {consumer input-sinks} ∪ {output attach if the node is an output}, and uses a `Broadcast` iff that set has more than one element. A node that is simultaneously an output and an intermediate therefore fans to both its consumer(s) and its gather.

- [ ] **Step 4: Build/run + commit**

Run: `make build && make install-dev && poetry run pytest tests/test_dag_compile.py -v` (all pass), then:

```bash
git add include/screamer/dag/compiled_graph.h tests/test_dag_compile.py
git commit -m "feat(dag): compiler fan-out + multi-output"
```

---

### Task 3: `CombineLatestNode` in the graph + multi-input routing

**Files:**
- Modify: `include/screamer/dag/compiled_graph.h`
- Test: `tests/test_dag_compile.py`

**Interfaces:**
- Consumes: Task 1/2 compiler + `dag::CombineLatestNode`.
- Produces: `add_combine_latest` nodes compile into `CombineLatestNode`s; their N ports are the input-sinks for their N input edges; a compiled `Sub(combine_latest(a, b))` graph equals eager.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dag_compile.py`:

```python
from screamer import Sub, combine_latest, RollingMean as RM


def test_compile_combine_then_functor_equals_eager():
    rng = np.random.default_rng(4)
    a_k = np.sort(rng.integers(0, 500, size=150)).astype(np.int64)
    a_v = rng.standard_normal(150)
    b_k = np.sort(rng.integers(0, 500, size=150)).astype(np.int64)
    b_v = rng.standard_normal(150)

    g = _b._GraphBuilder()
    ai, bi = g.add_input(), g.add_input()
    c = g.add_combine_latest([ai, bi], True)       # width-2 aligned
    spread = g.add_functor(Sub(), [c])             # 2-input functor over the width-2 edge
    z = g.add_functor(RM(10), [spread])            # smooth the spread
    g.set_outputs([z])
    (zk, zv), = g.run_batch([(a_k, a_v), (b_k, b_v)])

    keys, aligned = combine_latest((a_k, a_v), (b_k, b_v))   # when_all
    exp = RM(10)(aligned[:, 0] - aligned[:, 1])
    np.testing.assert_array_equal(zk, keys)
    np.testing.assert_array_equal(zv.reshape(-1), exp)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_dag_compile.py::test_compile_combine_then_functor_equals_eager -v`
Expected: FAIL — `compile()` does not yet handle `NodeKind::CombineLatest`.

- [ ] **Step 3: Handle `CombineLatest` in `compile()`**

Add the `CombineLatest` case:
- Construct `auto n = std::make_shared<CombineLatestNode<int64_t>>(node.inputs.size(), node.when_all, downstream); owned_.push_back(n);`.
- Its **input-sink for edge slot k** is `&n->port(k)` (not the node itself). So when resolving a consumer edge `(this_combine_node, slot k)`, the producer wires to `port(k)`.
- Its **output width** is `node.inputs.size()` (the aligned N).
- `CombineLatestNode` has no `EvalOp` to reset; its operator state is fresh per compiled instance (and the Task-1 simplification of rebuilding per `run_batch` keeps it clean).

Implementation detail: the per-edge input-sink resolution must be **slot-aware** — a `Functor` consumer resolves any incoming edge to the `FunctorNode` itself; a `CombineLatest` consumer resolves edge slot k to `port(k)`. Store, per node, a small `input_sink(slot)` accessor during compile.

- [ ] **Step 4: Build/run + full suite + commit**

Run: `make build && make install-dev && poetry run pytest tests/test_dag_compile.py -v` (all pass), then `poetry run pytest -q` (green), then:

```bash
git add include/screamer/dag/compiled_graph.h tests/test_dag_compile.py
git commit -m "feat(dag): compiler wires CombineLatestNode (multi-input) == eager"
```

---

## Self-Review

**1. Spec coverage (DAG-2b-4):**
- `GraphSpec` (data) + `GraphBuilder` → `graph.h`, Task 1. ✓
- `compile()` reverse-topo wiring, input-sink resolution, `Collector`/gather for outputs, `MergeSource` input routing → `compiled_graph.h`, Task 1. ✓
- Fan-out via `Broadcast` (downstream set >1) + multi-output → Task 2. ✓
- `CombineLatestNode` in the graph (ports as slot input-sinks) + multi-input → Task 3, `Sub(combine_latest(a,b))` graph == eager. ✓
- `CompiledGraph` pure C++ (binding marshals) → run_batch signature uses raw spans. ✓
- Byte-identity to eager across chain/fan-out/multi-output/combine. ✓
- Deferred (correctly absent): live-streaming driver, `align_outputs`, `dag.py` cutover, DAG-1 oracle matrix (all 2b-5).

**2. Placeholder scan:** the `compile()` body is described as a precise algorithm (reverse-topo + input-sink resolution + Broadcast/Collector) with the data model shown in full; Task 1 permits the "rebuild per run_batch" simplification for clarity. The steps that could be verification-only (Task 2) say so explicitly. No TBDs.

**3. Type consistency:** `NodeKind`/`NodeSpec`/`GraphSpec`/`GraphBuilder`(`add_input`/`add_functor`/`add_combine_latest`/`set_outputs`), `CompiledGraph::run_batch` / `OutputBuffer` / `GatherSink`, `compile()`, and `_GraphBuilder.run_batch([(keys,values),...]) -> [(keys,values2d),...]` are consistent across tasks. Input-sink resolution (functor=self, combine=port(k)) is stated in both Task 1 and Task 3.

---

## Follow-on

- **DAG-2b-5** — batch-replay (already have batch) + **live-streaming** driver over the same compiled graph + `align_outputs` + thin `dag.py` cutover (Python builds the `GraphSpec` via `_GraphBuilder`; `Node`/`Dag` become thin handles) + DAG-1 `_run` relocated to tests as the reference oracle, with a batch==stream==oracle identity matrix.
