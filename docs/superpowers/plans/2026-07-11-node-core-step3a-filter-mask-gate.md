# Node core step 3A: Filter as a 2-input mask gate

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the host-predicate `filter(values, predicate)` with `Filter()(data, mask)` - a pure C++ 2-input mask gate reachable from any binding. It emits each `data` value whose aligned `mask` value is nonzero (float zero-test; NaN drops). The mask is built upstream from the step-2 comparison/logic family (e.g. `Filter()(x, GreaterThan()(x, 0.0))`). No Python callback crosses the language boundary.

**Architecture:** A new `dag::FilterNode<Index>` modeled on `dag::CombineLatestNode` (`include/screamer/dag/combine_latest_node.h`): 2 single-value ports (0=data, 1=mask), reusing `streams::CombineLatest(2, when_all=true)` for as-of alignment. On each aligned distinct index it emits the width-1 data value IFF the aligned mask is nonzero-and-not-NaN; otherwise nothing. Coalesce per index and flush exactly like `CombineLatestNode`. Wired into the graph as `NodeKind::Filter`; the Python `Filter` class drives batch/lazy/graph through the existing `Dag` machinery (two width-1 inputs -> FilterNode), identical to how `combine_latest` already builds a 2-input graph.

**Tech Stack:** C++17 (header-only nodes under `include/screamer/dag/`), pybind11, CMake GLOB (no CMake edits to add headers), pytest. `make install-dev` after every C++ change.

## Global Constraints

- **`batch == lazy == graph`, bit-for-bit.** Every regime asserts byte-identical output. Causal, no lookahead.
- Suite green after each task (main baseline: 4105 passed, 2 skipped, 0 failed).
- Mask gate semantics EXACTLY: keep the data row iff `mask != 0.0 && !isnan(mask)`. Zero drops, NaN drops, any other value (incl. negative, incl. 1.0) keeps. Data value passes through unchanged (including a NaN data value - only the MASK gates).
- Breaking: `filter(values, predicate)` and `_filter_lazy` are DELETED (no lowercase alias, no predicate path). This is approved ("legacy removed, breaking is fine").
- No version-file edits. No em-dashes, no ` -- ` in code/comments/docs. Zero per-event heap allocation on the hot path (reuse buffers, like CombineLatestNode).
- `make install-dev` (not `make build`) after C++ edits.

## Interfaces (target)

- C++: `dag::FilterNode<Index>` with a `Sink<Index>& port(std::size_t)` returning port 0 (data) or 1 (mask), constructed `FilterNode(Sink<Index>& downstream)`; derives `Resettable`, `reset()` override. Emits width-1 frames.
- Graph: `NodeKind::Filter`; `GraphBuilder::add_filter(std::vector<std::size_t> inputs)` where `inputs = {data_id, mask_id}`; `node_width[Filter] = 1`.
- Binding: `_GraphBuilder.add_filter(inputs)` in `bindings/bindings_dag.cpp`.
- Python: `class Filter` with `__call__(self, data, mask)` dispatching on the pair by Rule A (raw arrays -> filtered `(values, index)`; lazy iterators -> lazy iterator of survivors; `Node` args -> a graph `Node`).

---

### Task 1: `FilterNode` (C++) + graph wiring + binding

**Files:** create `include/screamer/dag/filter_node.h`; edit `include/screamer/dag/graph.h` (`NodeKind`, `add_filter`, `NodeSpec` already has `inputs`), `include/screamer/dag/compiled_graph.h` (width case + wiring case + Resettable registration), `bindings/bindings_dag.cpp` (`add_filter`). Test: a C++/Python graph test that builds a 2-input Filter Dag and checks gating.

**Model to copy:** `include/screamer/dag/combine_latest_node.h` - same port/coalesce/flush structure. The ONLY differences: (1) `n = 2` fixed; (2) on emitting a buffered row, emit ONLY the data column (width 1) and ONLY if the buffered mask is nonzero-and-not-NaN.

- [ ] **Step 1:** write `filter_node.h`. Structure mirrors `CombineLatestNode`:
  - members: `streams::CombineLatest cl_{2, true};` `Sink<Index>& downstream_;` two `Port` structs (idx 0,1) forwarding to `on_port(i, f)`; a coalescing buffer holding the latest `(buffered_index_, buffered_data_, buffered_mask_)`; `has_buffered_`; the per-port `flushed_`/`flushed_count_` end-of-input coalescing (copy verbatim from CombineLatestNode, n=2).
  - `on_port(i, f)`: `assert(f.width == 1); if (cl_.on_event(i, f.values[0])) { const auto& row = cl_.latest(); if (has_buffered_ && f.index != buffered_index_) emit_buffered(); buffered_index_ = f.index; buffered_data_ = row[0]; buffered_mask_ = row[1]; has_buffered_ = true; }`
  - `emit_buffered()`: `if (buffered_mask_ != 0.0 && !screamer::isnan2(buffered_mask_)) { downstream_.push(Frame<Index>{buffered_index_, &buffered_data_, 1}); } has_buffered_ = false;` (include `screamer/common/float_info.h` for `isnan2`; store `buffered_data_` in a member so `&buffered_data_` is valid for the push).
  - flush: mirror CombineLatestNode's `flush_downstream(i)` - wait until all ports flushed, then `emit_buffered()` (gated) once, then `downstream_.flush()`, then re-arm.
  - `reset() override { cl_.reset(); has_buffered_ = false; fill(flushed_,false); flushed_count_=0; }`. Non-copyable/movable (ports hold back-references), like CombineLatestNode.
- [ ] **Step 2:** `graph.h`: add `Filter` to `NodeKind`; add `add_filter(std::vector<std::size_t> inputs)` (pattern of `add_combine_latest` but no `when_all`); NodeSpec reuses `inputs`.
- [ ] **Step 3:** `compiled_graph.h`: width case `case NodeKind::Filter: node_width[id] = 1; break;`; wiring case (mirror `CombineLatest` at ~210: `make_shared<FilterNode<int64>>(downstream)`, then `return &n->port(slot)` for each producer so data->port0, mask->port1 by input order), register the node in `reset_nodes_` (it is `Resettable`), keep it owned in `owned_`. Add `#include "screamer/dag/filter_node.h"`.
- [ ] **Step 4:** `bindings_dag.cpp`: bind `_GraphBuilder.add_filter(inputs)`.
- [ ] **Step 5:** build (`make install-dev`) + a focused test: construct a Dag with two Inputs -> Filter, run batch on data=`[1,2,3,4]` mask=`[1,0,1,0]` (aligned index) -> expect data `[1,3]` at the kept indices. Also mask with a NaN -> that row drops. Full suite green.
- [ ] **Step 6: commit.**

---

### Task 2: Python `Filter` class + delete `filter`/`_filter_lazy` + oracle tests

**Files:** `screamer/streams.py` (add `Filter` class, its node-builder registration in the Dag compile map, delete `filter` and `_filter_lazy`), `screamer/__init__.py` (regenerate exports via the devtools generator), any doc page for `filter`. Test: new `tests/test_filter_gate.py`; update/remove existing `filter` tests.

**Interfaces consumed:** Task 1's `add_filter`. Follow how `combine_latest` implements its three regimes (node via `make_operator_node`, lazy via a lazy Dag driver, batch via a small `CompiledGraph`/`Dag` call) - `Filter` is the same 2-input shape.

- [ ] **Step 1:** capture the oracle: for data/mask arrays, the expected survivors are `data[i]` where `mask[i] != 0 and not isnan(mask[i])`, with the aligned index. Write `tests/test_filter_gate.py` asserting this for: all-keep, all-drop, alternating, NaN-in-mask drops, negative mask keeps, NaN data value kept when its mask is nonzero. Assert `batch == lazy == graph` byte-identical (build all three regimes).
- [ ] **Step 2:** run - fails (`Filter` absent).
- [ ] **Step 3:** implement `class Filter` reusing the existing `Dag` machinery (do NOT hand-write separate batch/lazy drivers):
  - Register in the `dag.py` compile map (`_compile_cpp`, ~line 395): `elif name == "Filter": nid = gb.add_filter(inp)`. Identity is `fn.__name__`, so passing the `Filter` class as the operator gives `name == "Filter"`.
  - `__init__(self)` (no config in v1). `__call__(self, data, mask)`:
    - if `is_node(data) or is_node(mask)` -> `return make_operator_node(Filter, (data, mask), {})` (a graph `Node`).
    - else -> build a 2-input Dag and call it: `d, m = Input(), Input(); dag = Dag(inputs=[d, m], outputs=[Filter()(d, m)]); return dag(data, mask)`. The `dag(...)` call dispatches batch vs lazy via Rule A automatically (exactly as `_resample_via_cpp` does for 1 input), so a raw pair returns `(survivor_values, survivor_index_or_None)` and a lazy pair returns a lazy iterator - no bespoke driver. Confirm `Dag`/`Input` are importable in `streams.py` and the multi-input call convention `dag(feed_d, feed_m)` (read `_resample_via_cpp` + `Dag.__call__`).
  - Rule A container/rank preserved by the Dag call; positional (no index) -> index None.
- [ ] **Step 4:** DELETE `filter` and `_filter_lazy`. Grep for their callers in `screamer/` and tests; update or remove. Regenerate `screamer/__init__.py` (`PYTHONPATH=. python devtools/generate_screamer__init__.py`) and `help.json` if a doc page is added/removed (`PYTHONPATH=. python devtools/build_help_registry.py`).
- [ ] **Step 5:** `make install-dev`; `pytest tests/test_filter_gate.py -q`; full suite green.
- [ ] **Step 6: commit.**

---

## Self-review notes

- **Gate semantics single source of truth:** `mask != 0.0 && !isnan(mask)` keeps; everything else drops. Tested for zero, NaN, negative, and 1.0 masks, and for a NaN data value passing through under a keeping mask.
- **`batch == lazy == graph`** asserted byte-identical in Task 2 Step 1.
- **Reuse, do not fork:** `FilterNode` embeds `streams::CombineLatest` (the alignment operator) and copies `CombineLatestNode`'s coalesce/flush structure; it does not reimplement alignment.
- **Scope:** v1 data is a single-column stream (2 width-1 inputs). Multi-column-data filtering (a mask over rows of a packed stream) is DEFERRED with the multi-column-input work; note it, do not build it here.
- **Do NOT** rename other operators, touch `Stream`, or add lowercase aliases here - that is Plan 3D.
