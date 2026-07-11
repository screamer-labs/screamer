# Node core, step 1: unify the C++ node contract

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Establish the single C++ node contract the rest of the migration depends on: give the existing graph-node base (`dag::Sink`) the full `reset()` + `n_in()`/`n_out()` interface, collapse `CompiledGraph`'s five typed reset lists into one polymorphic traversal, and retire the older `streams::` single-value node world so there is exactly one push-node hierarchy. Behavior-preserving; the suite is the oracle.

**Architecture (from the C++ map):** The push/flush node model already exists as `dag::Sink<Index>` (`include/screamer/dag/frame.h`): `push(Frame) + flush()`. Graph nodes (`FunctorNode`, `ResampleNode`, `CombineLatestNode`, `DropNaNode`, `SelectNode`, ...) implement it. Math functors reach it through the `dag::FunctorNode` adapter (`include/screamer/dag/functor_node.h`). `resample` already routes batch, lazy, and live through one `ResampleNode` via `CompiledGraph`. The gaps to the target single-contract node are narrow and structural: `dag::Sink` carries no `reset()` and no arity, `CompiledGraph::reset()` walks five separate typed lists (`compiled_graph.h:277-288`), and a parallel `streams::Sink`/`Event` hierarchy (`include/screamer/streams/event.h`) still backs `_run_chain_*` and the standalone `combine_latest`/`merge` batch paths in `bindings_streams.cpp`.

**Tech Stack:** C++17 (header-heavy), pybind11, CMake/scikit-build-core (GLOB_RECURSE, no CMake edits to add files), pytest. `make install-dev` after every C++ change.

## Global Constraints

- **Behavior-preserving.** No numeric result changes; `batch == lazy == live` is unchanged. The full suite (3959 passed on `main`, 2 pre-existing `TestBOP` unrelated, 2 skipped) stays green after each task. This is a refactor, not a feature.
- All new C++ compiles under `-O3 -flto`; keep zero-allocation on the hot path (nodes reuse buffers).
- No version-file edits. No em-dashes and no ` -- ` (double-hyphen) in code/comments.
- Do not change operator semantics, the Python API surface, or `Stream` (later steps). This step is entirely under the hood.
- After any header/source change run `make install-dev` (not `make build` alone), then the suite.

## Interfaces (target)

A single graph-node base, `dag::Node<Index>` (rename or extend `dag::Sink`), with:

```cpp
template <class Index>
struct Node {
    virtual ~Node() = default;
    virtual std::size_t n_in()  const = 0;   // number of input ports
    virtual std::size_t n_out() const = 0;   // output frame width
    virtual void push(const Frame<Index>& f) = 0;
    virtual void flush() {}
    virtual void reset() {}
};
```

Every graph node implements it; `CompiledGraph` holds one `std::vector<Node<Index>*> reset_nodes_` and resets by iterating it.

---

### Task 1: add `reset()` to the graph-node base and every node

**Files:** `include/screamer/dag/frame.h` (the `Sink`/`Node` base); every node header in `include/screamer/dag/` (`functor_node.h`, `resample_node.h`, `resample_generic_node.h`, `combine_latest_node.h`, `multi_resample_node.h`, `dropna_node.h`, `select_node.h`, `broadcast.h`, and the port structs). Test: the existing suite.

**Interfaces:** Produces `virtual void reset() {}` on the node base; each stateful node overrides it with its existing reset logic; stateless nodes inherit the no-op.

- [ ] **Step 1: add `virtual void reset() {}`** to the `Sink<Index>` base in `frame.h` (keep the name `Sink` for now to minimize churn; the rename is Task 4).
- [ ] **Step 2: give each node a `reset()` override** that performs exactly what its current external reset does. `ResampleNode`, `GenericResampleNode`, `MultiResampleNode`, `CombineLatestNode` already have `reset()` methods (per the map) - make them `override`. `FunctorNode` gains `void reset() override { op_.reset(); }` (it currently has none; the embedded `EvalOp` is reset from `CompiledGraph::reset_ops_`). `DropNaNode`/`SelectNode`/`Broadcast` are stateless: inherit the no-op. Multi-port nodes forward reset to their internal state (they already do).
- [ ] **Step 3: build + suite.** `make install-dev`; run `python -m pytest -q`. Green (no behavior change; reset() is added but `CompiledGraph` still uses its old lists until Task 2).
- [ ] **Step 4: commit.**

---

### Task 2: collapse `CompiledGraph`'s reset into one polymorphic traversal

**Files:** `include/screamer/dag/compiled_graph.h` (constructor wiring at 197-273, `reset()` at 277-288). Test: suite.

- [ ] **Step 1:** replace the five typed lists (`reset_ops_`, `reset_combines_`, `reset_resamples_`, `reset_generic_resamples_`, `reset_multi_resamples_`) with one `std::vector<Sink<std::int64_t>*> reset_nodes_`. During construction, push every owned node that carries state into it (a `FunctorNode` wrapping a stateful `EvalOp`, and every resample/combine node). Since `FunctorNode::reset()` now forwards to `op_.reset()` (Task 1), the separate `reset_ops_` EvalOp list is no longer needed.
- [ ] **Step 2:** `reset()` becomes `for (auto* n : reset_nodes_) n->reset();` plus the output-buffer clear it already does.
- [ ] **Step 3: build + suite.** Green. Pay attention to any node whose reset was previously called via `reset_ops_` (the embedded EvalOp) - it must now be reset via its `FunctorNode`. Add a focused test if any resample-of-functor or repeated-run case is not already covered (the existing `batch == lazy` and reset tests should cover it; confirm).
- [ ] **Step 4: commit.**

---

### Task 3: retire the `streams::` node world; route its two callers through `dag::`

**Files:** `include/screamer/streams/event.h` (`streams::Sink`, `Event`), `include/screamer/streams/*` (`streams::FunctorNode`, `streams::CombineLatest` if it is the node-layer one), `bindings/bindings_streams.cpp` (`_run_chain_*`, `_combine_latest_*` batch, `_merge_*` batch, the pullers). Test: `tests/` covering `resample` chains, `combine_latest`, `merge`, `replay`.

**Context:** `_run_chain_*` (a utility that chains `ScreamerBase` functors) and the standalone eager `combine_latest`/`merge` batch use the `streams::` single-value `Event` hierarchy, which duplicates `dag::`. This is where `combine_latest` gets its second C++ implementation.

- [ ] **Step 1: identify every `streams::` caller** - grep `streams::` under `bindings/` and `src/`, list each entry point (`_run_chain_i64/f64`, `_combine_latest_i64/f64`, `_merge_i64/f64`, `_CombineLatestPuller`, `_MergePuller`) and what Python calls it (`grep -rn "_run_chain\|_combine_latest_\|_merge_\|_CombineLatestPuller\|_MergePuller" screamer/`).
- [ ] **Step 2: route each through `dag::`.** Replace the `streams::`-based batch with a one-node (or small) `CompiledGraph` built from the corresponding `dag` node (`CombineLatestNode`, and a `dag::MergeNode` if none exists - if merge has no `dag` node, that port belongs to a later step; in that case leave `merge`'s `streams::` path and scope this task to `combine_latest` + `_run_chain`, noting merge explicitly). `_run_chain_*` becomes a chain of `FunctorNode`s in a `CompiledGraph` (or is deleted if its only caller can use the normal functor batch path - check its callers).
- [ ] **Step 3: assert identical results.** For every rerouted entry point, a test asserts the new `dag::` path is byte-identical to the pre-change `streams::` output (capture expected values before deleting the old path). Do not weaken.
- [ ] **Step 4: delete** the now-unused `streams::` node headers and bindings once no caller remains. If `merge` still needs its `streams::` puller (no `dag::MergeNode` yet), keep only that and note it in the ledger for the merge-port step.
- [ ] **Step 5: build + suite.** Green.
- [ ] **Step 6: commit.**

---

### Task 4: name the single contract (`dag::Node`) and document it

**Files:** `include/screamer/dag/frame.h` (rename `Sink` -> `Node`, or add `using Node = Sink`), all `dag/` node headers, `include/screamer/dag/README` or a header doc comment.

- [ ] **Step 1:** rename `dag::Sink<Index>` to `dag::Node<Index>` (the base now carries `n_in/n_out/push/flush/reset` - it is a node, not just a sink). Keep `Sink` as a deprecated alias for one step if the churn is large, else rename outright. Update all references.
- [ ] **Step 2:** add a header doc comment stating the contract precisely (the target interface above) - this is the reference every future operator implements and every driver drives.
- [ ] **Step 3: build + suite.** Green.
- [ ] **Step 4: commit.**

---

## Self-review notes

- **Oracle is the suite + byte-identical reroutes.** Every task is behavior-preserving; Task 3 explicitly captures pre-change outputs and asserts equality after rerouting.
- **`n_in`/`n_out` on the node:** Task 1 adds `reset()`; arity (`n_in`/`n_out`) is currently in `NodeSpec` not on the node. Adding the arity accessors to the node base is part of the target interface - fold it into Task 1 or Task 4 (each node returns its known arity; `FunctorNode` returns `op_.n_in()/op_.n_out()`). Keep it behavior-preserving (arity is read-only metadata).
- **What this does NOT do:** no operator porting (dropna/select/merge/filter eager+lazy stay as-is), no API collapse, no `Stream` removal, no new operators. Those are steps 2+ with their own plans. This step only makes the one C++ node contract real, so those steps have a single interface to target.
- **Merge caveat:** if there is no `dag::MergeNode`, Task 3 leaves merge on its `streams::` path and flags it; the merge port (a later step) adds a `dag::MergeNode` and finishes the `streams::` retirement.
