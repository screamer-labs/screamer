# Node core, step 2: port operators onto the node contract, retire `streams::`

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Remove the remaining places where operator COMPUTE runs in Python instead of the C++ core, so every operator is reachable from a non-Python binding. Add the comparison/logic operator family (a prerequisite for the later `Filter` mask gate). All behavior-preserving or purely additive; the suite is the oracle.

**Architecture (verified against the source, corrected from the earlier coarse map):** `streams::` is NOT a "node world to retire" - it is the batch-driver + operator + source infrastructure that the target "one operator, three drivers (batch/lazy/graph)" model wants. `streams::CombineLatest` (`combine_latest.h`) is the alignment OPERATOR reused inside `dag::CombineLatestNode`; `streams::MergeSource` (`merge_source.h`) is the k-way merge operator reused by the eager `merge` and by `replay`; `combine_latest`'s batch binding and `_run_chain` are legitimate batch drivers over those C++ operators. Those stay.

The genuine debt is where a driver reimplements an operator in PYTHON:
- `_merge_lazy` (streams.py) is a Python k-way merge duplicating the C++ `MergeSource`. `merge`'s eager path already uses the C++ operator (`_merge_f64/i64`) and `replay` uses the `_MergePuller`; only the lazy path forks into Python. (Task 2)
- `dropna`/`select` compute in numpy (eager) and Python generators (lazy), duplicating the existing C++ `DropNaNode`/`SelectNode` graph nodes - two/three implementations of one operator that can diverge and are unreachable from a non-Python binding. (Tasks 4, 5)

`merge` is INPUT ROUTING, not a graph node - the code already enforces this (`merge` raises on `Node` inputs: "it is input routing"). There is no `MergeNode` and this step does not add one. The `dag` engine's `push_event(input_idx, index, value)` carries a scalar; multi-column (row) inputs need a widened push before `dropna`/`select`'s multi-column lazy paths can route through the C++ node (Task 4).

**Tech Stack:** C++17 (headers under `include/screamer/`), pybind11, CMake GLOB (no CMake edits to add files), pytest. `make install-dev` after every C++ change.

## Global Constraints

- **Behavior-preserving / additive.** No numeric result changes for existing operators; new operators are additive. Suite green after each task (main baseline: 3962 passed, 2 pre-existing `TestBOP`, 2 skipped). For every rerouted path, a test asserts byte-identical output to the pre-change path.
- The Python API shape is unchanged in this step (still `op(data, ...)` functions). CamelCase collapse, `Filter` mask-gate, and `Stream` removal are step 3 (breaking).
- No version-file edits. No em-dashes / no ` -- `. Zero-allocation hot path (nodes reuse buffers).
- `make install-dev` (not `make build`) after C++ edits.

---

### Task 1: comparison and logic operator family (additive C++ functors)

**Files:** new headers under `include/screamer/` (one per functor or a shared `logic.h`), binding lines in the appropriate `bindings/bindings_*.cpp`. Test: a new `tests/test_logic_ops.py`.

Add these as `ScreamerBase` (1-in) or `FunctorBase` (2-in) functors, producing `1.0`/`0.0` masks (NaN in -> NaN out, so NaN propagates and later drops in `Filter`):

- 2-in: `GreaterThan`, `LessThan`, `GreaterEqual`, `LessEqual`, `Equal`, `NotEqual` (a>b etc. -> 1.0/0.0), `And`, `Or` (nonzero test), `Where` (3-in: `Where(mask, a, b)` -> a if mask nonzero else b).
- 1-in: `Not` (nonzero -> 0.0, zero -> 1.0), `IsNan` (-> 1.0/0.0), `IsFinite`.

- [ ] **Step 1:** write `tests/test_logic_ops.py` asserting each op's truth table on arrays incl. NaN (e.g. `GreaterThan()(np.array([1,2,3]), np.array([2,2,2]))` -> `[0,0,1]`; NaN inputs -> NaN).
- [ ] **Step 2:** run to confirm failure (ops absent).
- [ ] **Step 3:** implement the functors (follow the existing `Add`/`Sub`/`Sign` patterns: `FunctorBase<D,2,1>::call` for binary, `ScreamerBase::process_scalar` for unary). NaN handling: any NaN input -> NaN output.
- [ ] **Step 4:** add binding lines (pattern: `py::class_<screamer::GreaterThan, screamer::ScreamerBase>(m,"GreaterThan").def(py::init<>()).def("__call__",...).def("reset",...);` for unary; the 2-in ones bind `operator()` variadically like `Add`).
- [ ] **Step 5:** `make install-dev`; `pytest tests/test_logic_ops.py -q`; full suite. Green.
- [ ] **Step 6: commit.**

---

### Task 2: route `merge`'s lazy path through the C++ `MergeSource`; delete the Python `_merge_lazy`

**Files:** `screamer/streams.py` (`_merge_lazy`, `_merge_lazy_dispatch`, `merge`); reuse the existing `_MergePuller_i64/f64` binding (the same C++ `MergeSource` puller `replay` uses). Possibly a small binding helper if the puller cannot drive lazy Python iterators directly.

`merge`'s eager path already uses the C++ `MergeSource` (`_merge_f64/i64`) and `replay` drives it through `_MergePuller`. Only the lazy path forks into a Python k-way merge (`_merge_lazy`). Route lazy through the same C++ operator so merge's compute lives in one place (C++), reachable from any binding.

- [ ] **Step 1:** oracle - capture current `merge(generators)` lazy output `(value, index_or_None, source)` for several cases: indexed sources, positional (unequal lengths), ties across sources (equal index -> lower source wins), single source, empty source. These become the byte-identical assertion.
- [ ] **Step 2:** study `_MergePuller` (`bindings/bindings_streams.cpp`, used by `replay` at streams.py:522) - it wraps `streams::MergeSource` over child sources and pulls events one at a time. Determine whether it can be fed lazy Python iterators (a `PySource` adapter that calls `next()` on a Python iterator) or whether `replay` materializes first. Reuse or minimally extend it so a lazy iterator merge pulls through the C++ `MergeSource`.
- [ ] **Step 3:** rewrite `_merge_lazy_dispatch` to drive the C++ `MergeSource` puller (classify sources on first `__next__` as today, then pull from the C++ merge), yielding `(value, index_or_None, source)` byte-identical to Step 1. Delete `_merge_lazy` (the Python k-way loop).
- [ ] **Step 4:** assert byte-identical to the Step 1 oracle across all cases; `make install-dev`; full suite green (existing merge lazy tests must stay green unchanged).
- [ ] **Step 5: commit.**

Note: `streams::MergeSource`, `streams::CombineLatest`, `event.h` (Source/Event), the batch `combine_latest`/`merge` bindings, and `_run_chain` are the C++ operators + batch drivers of the three-driver model and STAY. This task removes only the Python reimplementation of the merge operator.

---

### Tasks 3-4: DEFERRED to step 3 (see note below)

**Deferred (2026-07-11):** Investigation during step-2 execution found that routing standalone `dropna`/`select` through the C++ `DropNaNode`/`SelectNode` requires the engine to support **multi-column INPUTS**. Today the graph's `add_input()` creates width-1 input nodes and `push_event` carries a scalar; multi-column data exists only INTERNALLY (from `combine_latest`/`resample`). That is exactly why standalone eager `dropna(2d)`/`select(2d)` use numpy directly (bypassing the graph) - the duplicate the user flagged. Adding multi-column-input support (width-N inputs + wide push + `_LazyDag` row push) is the same data-model generalization step 3 performs wholesale (index-as-data, `Stream` removal, rows flowing through the engine). Building it in isolation here would be redone in step 3. So Task 3 (wide push) and Task 4 (dropna/select port) MOVE into step 3, done once alongside the index-as-data rework. Step 2 closes at Tasks 1-2.

The original Task 3/4 text is retained below for the step-3 plan to pull from.

---

### Task 3 (deferred): widen the engine push to carry multi-column rows

**Files:** `include/screamer/dag/compiled_graph.h` (`push_event`), `bindings/bindings_dag.cpp` (`_CompiledGraph.push_event`), `screamer/dag.py` (`_LazyDag._pull`/`push`).

The engine's `push_event(input_idx, index, value)` takes a scalar. `Frame` already carries `const double* values` + `width`, so nodes handle width > 1 internally; only the injection boundary is scalar. Widen it so a lazy multi-column input row can be pushed.

- [ ] **Step 1:** add an overload `push_event_row(input_idx, index, const double* values, width)` (or make `push_event` accept a small array) on `CompiledGraph`, building a `Frame{index, values, width}`. Keep the scalar `push_event` as a width-1 convenience.
- [ ] **Step 2:** expose it in the binding (accept a 1-D numpy row or a float).
- [ ] **Step 3:** `_LazyDag` pushes a row when an input event value is a vector (a multi-column `(row, index)` event), scalar otherwise.
- [ ] **Step 4:** test: a lazy multi-column feed through a passthrough/`select` node equals the batch result. Behavior-preserving for existing scalar paths (suite green).
- [ ] **Step 5: commit.**

---

### Task 4: route `dropna`/`select` eager + lazy through their C++ nodes; delete the numpy/Python duplicates

**Files:** `screamer/streams.py` (`dropna`, `select`, `_dropna_lazy`, `_select_lazy`). Test: existing dropna/select tests (must stay green byte-for-byte) + a multi-column lazy case.

- [ ] **Step 1:** oracle - the existing dropna/select tests already assert the numpy results; capture any not covered (multi-column dropna how=any/all, select column lists, lazy multi-column).
- [ ] **Step 2:** route `dropna` eager and lazy through `DropNaNode` (build a one-node `Dag`, like `_resample_via_cpp`); the lazy multi-column path uses Task 3's row push. Delete the numpy mask path and `_dropna_lazy`.
- [ ] **Step 3:** same for `select` through `SelectNode`; delete the numpy pick and `_select_lazy`.
- [ ] **Step 4:** assert byte-identical to the oracle across eager/lazy/scalar/multi-column; `make install-dev`; full suite green.
- [ ] **Step 5: commit.**

---

## Self-review notes

- **Oracle everywhere:** every rerouted operator asserts byte-identical output to its pre-change path; new operators assert their truth tables. `batch == lazy == graph` holds.
- **`streams::` is NOT retired:** it is the reused operator + batch-driver + source infrastructure of the three-driver model. This step removes only the PYTHON reimplementations of operators (`_merge_lazy`, `dropna`/`select` numpy+generator paths).
- **After this step:** no operator computes in Python; the comparison/logic family exists. Remaining: the API-shape collapse to `Op(config)(data)` + CamelCase, the `Filter` mask-gate (uses Task 1's comparison family), `Resample` fold-in, and `Stream` removal (step 3, breaking).
- **Do NOT** change the Python API shape, rename to CamelCase, touch `Filter`'s predicate, add a `MergeNode`, delete any `streams::` operator/driver, or remove `Stream` here.
