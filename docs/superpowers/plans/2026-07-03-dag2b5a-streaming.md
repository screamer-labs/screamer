# DAG-2b-5a — persistent wired graph + live-streaming driver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the compiled graph persistent (wired once) with a `reset()`, then add a **live-streaming** driver — `push_event` one event at a time — and prove streaming produces byte-identical output to batch over the same events. This delivers the whole point of the C++ engine: define once, run live.

**Architecture:** Refactor `CompiledGraph` so `compile()` builds the wired push-graph **once** (nodes/broadcasts/gathers owned as members); `run_batch` becomes reset + drive-from-`MergeSource` + read the gathers; and a new `push_event(input_idx, key, value)` feeds one live event into the same wired graph, with `drain()` returning the outputs emitted since the last drain. Batch and streaming are two drivers over one persistent graph — identity by construction.

**Tech Stack:** C++17, pybind11, numpy, pytest.

## Global Constraints

- **One persistent wired graph, two drivers:** `run_batch` (drive from arrays via `MergeSource`) and streaming (`push_event` externally); the node code is identical → batch==stream by construction.
- **`reset()` for repeatability:** resets every stateful op (`FunctorNode`'s `EvalOp`) and every `CombineLatestNode`. `run_batch` calls `reset()` first; streaming does not reset between events (state persists).
- **Byte-identity:** streaming the same events in the same key order as batch yields identical output (`np.testing.assert_array_equal`, NaN-aware).
- Zero per-event allocation on the push path (persistent nodes reuse buffers). Compute functors' math never modified. Never hand-edit `screamer/__init__.py` or version files.
- Build: `make build` **then `make install-dev`**. Tests: `poetry run pytest tests/test_dag_compile.py tests/test_dag_stream.py -v`.

---

## File Structure

- `include/screamer/streams/combine_latest.h` (modify) — add `reset()` to the `CombineLatest` operator.
- `include/screamer/dag/combine_latest_node.h` (modify) — add `reset()` forwarding to the operator.
- `include/screamer/dag/compiled_graph.h` (modify) — persistent wiring in `compile()`; `reset()`; `run_batch` reset+drive+read; `push_event`/`drain`.
- `bindings/bindings_dag.cpp` (modify) — bind a persistent `_CompiledGraph` (`push_event`, `drain`, `run_batch`) and `_GraphBuilder.compile()`.
- `tests/test_dag_stream.py` (create) — streaming==batch identity.

---

### Task 1: persistent wired graph + `reset()` (refactor; batch unchanged)

**Files:**
- Modify: `include/screamer/streams/combine_latest.h`, `include/screamer/dag/combine_latest_node.h`, `include/screamer/dag/compiled_graph.h`
- Test: `tests/test_dag_compile.py` (existing — must stay green)

**Interfaces:**
- Produces: `CompiledGraph` wires once in `compile()` (owns nodes/broadcasts/gathers as members); `CompiledGraph::reset()` resets all stateful ops + combine nodes; `run_batch(...)` now does `reset()` → drive from `MergeSource` → return the gathers' `OutputBuffer`s (unchanged observable behavior). `streams::CombineLatest::reset()`; `dag::CombineLatestNode::reset()`.

- [ ] **Step 1: Add `reset()` to the operator + node**

In `include/screamer/streams/combine_latest.h`, add to the `CombineLatest` operator:

```cpp
    void reset() {
        std::fill(latest_.begin(), latest_.end(),
                  std::numeric_limits<double>::quiet_NaN());
        std::fill(seen_.begin(), seen_.end(), static_cast<char>(0));
        seen_count_ = 0;
    }
```

(Ensure `<algorithm>` is included.) In `include/screamer/dag/combine_latest_node.h`, add to `CombineLatestNode`:

```cpp
    void reset() { cl_.reset(); }
```

- [ ] **Step 2: Refactor `CompiledGraph` to persistent wiring**

Move the wiring currently done inside `run_batch` into `compile()`, storing as `CompiledGraph` members: the `owned` node objects, `input_sinks_` (per input signature index), `output_gathers_`/`outputs_` (per output), `reset_ops_` (functor `EvalOp*`s), `reset_combines_` (`CombineLatestNode*`s), and `output_widths_`/`num_in_`. Add:

```cpp
    void reset() {
        for (auto* op : reset_ops_) op->reset();
        for (auto* c : reset_combines_) c->reset();
        for (auto& b : outputs_) { b.keys.clear(); b.values.clear(); }
    }
```

`run_batch(in_keys, in_vals, in_lens)` becomes: validate counts → `reset()` → build a `MergeSource` over the inputs → drive each event to `input_sinks_[source]` (width-1 frame) → `flush()` → return `outputs_` (copied). Keep the input-count/cycle/unresolved-sink guards. The GatherSinks now write into the persistent `outputs_` members (allocated in `compile()`), so they must set `outputs_[o].width` = the output node's width at wire time.

- [ ] **Step 3: Run the existing compile tests**

Run: `make build && make install-dev && poetry run pytest tests/test_dag_compile.py -v`
Expected: all 6 PASS unchanged (this is a pure refactor — persistent wiring + reset produce identical batch output).

- [ ] **Step 4: Full-suite guard + commit**

Run: `poetry run pytest -q` (green), then:

```bash
git add include/screamer/streams/combine_latest.h include/screamer/dag/combine_latest_node.h include/screamer/dag/compiled_graph.h
git commit -m "refactor(dag): persistent wired graph + reset() (batch unchanged)"
```

---

### Task 2: `push_event` streaming driver + streaming==batch identity

**Files:**
- Modify: `include/screamer/dag/compiled_graph.h`
- Modify: `bindings/bindings_dag.cpp`
- Test: `tests/test_dag_stream.py`

**Interfaces:**
- Consumes: the persistent `CompiledGraph` (Task 1).
- Produces:
  - C++: `CompiledGraph::push_event(std::size_t input_idx, std::int64_t key, double value)` (routes a width-1 frame to `input_sinks_[input_idx]`); `std::vector<OutputBuffer> drain()` (returns the outputs emitted since the last drain/reset, then clears them).
  - Python: `_CompiledGraph` bound object — `compile()` on `_GraphBuilder` returns it; methods `reset()`, `push_event(input_idx, key, value)`, `drain() -> [(keys, values2d), ...]`, and `run_batch([(keys,values),...]) -> [(keys,values2d),...]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dag_stream.py`:

```python
import numpy as np
from screamer import RollingMean, Diff, Sub, combine_latest, merge
from screamer import screamer_bindings as _b


def test_stream_equals_batch_chain():
    x = np.random.default_rng(0).standard_normal(200)
    xk = np.arange(x.size, dtype=np.int64)

    b = _b._GraphBuilder()
    xi = b.add_input()
    y = b.add_functor(Diff(1), [b.add_functor(RollingMean(5), [xi])])
    b.set_outputs([y])

    cg = b.compile()
    (bk, bv), = cg.run_batch([(xk, x)])          # batch

    cg.reset()
    for k, v in zip(xk, x):                        # streaming: one event at a time
        cg.push_event(0, int(k), float(v))
    (sk, sv), = cg.drain()
    np.testing.assert_array_equal(sk, bk)
    np.testing.assert_array_equal(sv, bv)


def test_stream_equals_batch_combine():
    rng = np.random.default_rng(1)
    a_k = np.sort(rng.integers(0, 500, size=120)).astype(np.int64)
    a_v = rng.standard_normal(120)
    b_k = np.sort(rng.integers(0, 500, size=120)).astype(np.int64)
    b_v = rng.standard_normal(120)

    gb = _b._GraphBuilder()
    ai, bi = gb.add_input(), gb.add_input()
    spread = gb.add_functor(Sub(), [gb.add_combine_latest([ai, bi], True)])
    gb.set_outputs([spread])
    cg = gb.compile()

    (bk, bv), = cg.run_batch([(a_k, a_v), (b_k, b_v)])   # batch

    cg.reset()
    for k, v, src in merge((a_k, a_v), (b_k, b_v)):       # key-ordered tagged events
        cg.push_event(int(src), int(k), float(v))
    (sk, sv), = cg.drain()
    np.testing.assert_array_equal(sk, bk)
    np.testing.assert_array_equal(sv, bv)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_dag_stream.py -v`
Expected: FAIL — `_GraphBuilder` has no `compile`; `_CompiledGraph` undefined.

- [ ] **Step 3: Add `push_event`/`drain` and bind `_CompiledGraph`**

In `compiled_graph.h`:

```cpp
    void push_event(std::size_t input_idx, std::int64_t key, double value) {
        if (input_idx >= num_in_) throw std::runtime_error("push_event: input index out of range");
        double v = value;
        input_sinks_[input_idx]->push(Frame<std::int64_t>{key, &v, 1});
    }

    std::vector<OutputBuffer> drain() {
        std::vector<OutputBuffer> out = std::move(outputs_);
        outputs_.assign(output_widths_.size(), OutputBuffer{});
        for (std::size_t o = 0; o < outputs_.size(); ++o) outputs_[o].width = output_widths_[o];
        rebind_gathers();   // point the GatherSinks at the fresh outputs_
        return out;
    }
```

(Implement `rebind_gathers()` to re-seat the gather sinks on the new `outputs_` — or, simpler, have `GatherSink` hold a `CompiledGraph*` + output index and always append to `outputs_[idx]`, so `drain()` just swaps the vector. Pick the clearer option.)

In `bindings/bindings_dag.cpp`, bind `dag::CompiledGraph` as `_CompiledGraph` with `reset`, `push_event`, `drain` (marshal `OutputBuffer`s to `[(keys, values2d), ...]`), and `run_batch` (move the existing `_GraphBuilder.run_batch` marshalling into a shared helper used by both `_CompiledGraph.run_batch` and `drain`). Add `_GraphBuilder.compile()` returning a `_CompiledGraph` (holding the `CompiledGraph` plus the builder's `op_refs` so functors stay alive).

- [ ] **Step 4: Build and run**

Run: `make build && make install-dev && poetry run pytest tests/test_dag_stream.py tests/test_dag_compile.py -v`
Expected: all PASS — streaming (event-at-a-time) is byte-identical to batch for a chain and a combine graph.

- [ ] **Step 5: Full-suite guard + commit**

Run: `poetry run pytest -q` (green), then:

```bash
git add include/screamer/dag/compiled_graph.h bindings/bindings_dag.cpp tests/test_dag_stream.py
git commit -m "feat(dag): live-streaming push_event driver; stream == batch"
```

---

## Self-Review

**1. Spec coverage (DAG-2b-5a):**
- Persistent wired graph (wired once in `compile()`) + `reset()` → Task 1; batch unchanged (existing tests green). ✓
- `CombineLatest`/`CombineLatestNode` `reset()` → Task 1. ✓
- Live-streaming `push_event` + `drain` → Task 2. ✓
- Streaming == batch by construction (one persistent graph, two drivers) → Task 2 identity tests (chain + combine, feeding key-ordered events via `merge`). ✓
- Zero per-event alloc (persistent nodes reuse buffers) → preserved. ✓
- Deferred (correctly absent): `align_outputs` in the engine, `dag.py` cutover, DAG-1 oracle matrix (all 2b-5b).

**2. Placeholder scan:** the `rebind_gathers()`/GatherSink-holds-CompiledGraph* detail offers two concrete implementations and says pick the clearer — not a TBD; both are fully described. Task 1's wiring move is a refactor of existing code (already in `compiled_graph.h`), reset() shown in full.

**3. Type consistency:** `CompiledGraph::reset/run_batch/push_event/drain`, `OutputBuffer`, `CombineLatest::reset`, `CombineLatestNode::reset`, and the `_CompiledGraph`/`_GraphBuilder.compile()` Python surface are consistent across tasks. `push_event(input_idx, key, value)` and `merge(...) -> (key, value, source)` line up in the identity test.

---

## Follow-on

- **DAG-2b-5b (final)** — `align_outputs` in the engine (a terminal `combine_latest` over outputs); thin `dag.py` cutover (`Node`/`Input` build the C++ `GraphSpec` via `_GraphBuilder`; `Dag.__call__` → `_CompiledGraph.run_batch`; `Dag.stream` → `push_event`); remove the Python `_run` executor and relocate it to tests as the DAG-1 reference oracle; batch==stream==oracle identity matrix. Graph `combine_latest` is alignment-only (no `func=`); reduction via C++ functors.
