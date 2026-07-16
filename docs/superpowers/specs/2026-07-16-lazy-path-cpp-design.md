# screamer: Move the lazy-path logic into the C++ core - Design

**Status:** draft design, pending review
**Date:** 2026-07-16
**Scope:** the lazy (iterator) streaming path in `screamer/streams.py` and `screamer/dag.py`

## Context

This is the last part of the C++-core remediation (operator and data-path logic
lives in C++; Python is thin bindings/orchestration; one implementation per
behavior). Two Part B items are already settled:

- `split` now runs in a C++ `split_batch` (merged).
- `_resample_ohlcv` stays in Python: it is the composition pattern that
  deliberately replaced `MultiResampleNode` (a prior user-directed removal:
  "superseded by composition"). It only splits columns, runs C++ sub-resamples,
  and `column_stack`s pre-aligned results. Not a violation.

What remains is the **lazy (iterator) streaming path**. The library has three
execution modes that must produce identical results (the crown-jewel invariant,
`batch == lazy == graph`): a batch array call, a lazy pull over Python-iterator
feeds, and the compiled-graph engine. The batch and graph modes run in C++. The
**lazy mode still carries real algorithm logic in Python**, and in two places a
second implementation of behavior the C++ engine already has.

The C++ side already provides the lazy building blocks, which this design builds
on rather than replaces:

- `MergeLazyPuller<Index>` (bindings_streams.cpp) - a k-way as-of merge of
  Python-iterator sources through a C++ heap, deferred-refill, yielding
  `(value, index, source)`.
- `CombineLatestPuller<Index>` - a lazy combine-latest over Python-iterator
  sources in C++.
- `CompiledGraph` (dag/compiled_graph.h) - `push_event` / `advance` / `flush` /
  `drain` / `reset`, the streaming API the batch driver already uses.

## Goal

Move the genuine lazy-path algorithm logic from Python into the C++ core, and
delete the duplicate implementations, so every mode has one C++ implementation
and `batch == lazy == graph` is enforced by a single code path rather than by
two parallel ones agreeing.

## The three items

### 1. Unify the lazy `combine_latest` (delete the dual implementation)

`streams.py::_combine_latest_zip_lazy` (lines ~289-302) reimplements the
positional (aligned-clock) combine-latest join in Python with
`itertools.zip_longest`. The eager positional path calls the C++
`combine_latest_batch`; the as-of lazy path already routes through the C++
graph; only the **positional lazy** path stayed a separate Python join. This is
a dual implementation of one behavior - a maintenance and parity hazard.

**Design.** Route positional lazy combine-latest through the existing C++
`CombineLatestPuller<Index>` (already bound as `_CombineLatestPuller_i64/f64`),
the same primitive the indexed lazy path uses, driven positionally. Delete
`_combine_latest_zip_lazy`. `combine_latest`'s lazy dispatch then has one C++
path for both positional and indexed.

**Risk:** low. The C++ puller already exists and is tested; this removes Python,
does not add C++.

### 2. Make the eager `combine_latest` coalesce in C++ (like the graph node already does)

This is not a new cadence - it aligns an inconsistency. The graph
`CombineLatestNode` (dag/combine_latest_node.h) **already coalesces in C++**: it
emits one frame per DISTINCT index (same-index events update a buffered row; the
row is pushed when the index advances, with end-of-input flush coalescing). So a
pure-C++ user driving combine_latest through the graph already gets one row per
index - that is the established output contract, and it does not change.

The inconsistency is the *separate* eager `combine_latest_batch`
(bindings_streams.cpp), which emits one row per merged **event**;
`streams.py::_collapse_last_per_index` (lines ~178-187) then dedups it in numpy
to match the node. That is two C++ implementations of combine_latest with
different emit cadence, plus a Python patch to reconcile them.

**Design.** Give the eager path the node's coalescing so there is ONE cadence.
Preferred: route `combine_latest`'s eager path through the graph
`CombineLatestNode` + a `Collector`, so a single coalescing implementation
serves eager, lazy, and graph. Simpler alternative: port the node's
buffer-and-emit-on-index-advance logic into `combine_latest_batch`. Either way,
delete `_collapse_last_per_index` and its call sites.

**Right for pure-C++ users:** yes - they already receive coalesced
(one-row-per-index) output from the node; this makes the eager helper match it,
rather than imposing anything new. It also removes a dual implementation.

**Risk:** low-moderate. Guarded by the combine-latest and `batch == lazy ==
graph` tests, which already assert the coalesced output.

### 3. Move `_LazyDag` into a C++ lazy driver

`dag.py::_LazyDag` (lines ~201-325, ~120 lines) is the lazy driver for a
compiled `Pipeline` over iterator feeds. It does three things, two of which are
real algorithm logic:

- **As-of event scheduling** - merges the input feeds by index (min across
  per-input heads, same-index coalescing) and pushes each event into the C++
  compiled graph. This duplicates what `MergeLazyPuller` already does in C++.
- **Multi-output watermark join** (`_drain_rows` / `_settle`, the
  `_buf`/`_wm`/`_latest` machinery) - the M outputs of a multi-output pipeline
  drain at independent rates, so their merged index stream is not globally
  sorted; the join buffers drained events and finalizes an index only once every
  output has drained past it (an as-of `when_all` coalesce on a watermark). This
  is genuine algorithm logic, currently only in Python.

**Design.** A C++ lazy driver (extend `dag/driver.h`, currently only the batch
`replay_batch`) that:
- consumes the input feeds via `MergeLazyPuller` (reuse - the as-of merge is
  already C++),
- pushes each merged event into the `CompiledGraph` (`push_event`), and
- runs the multi-output watermark join as a C++ collector (port `_settle`'s
  buffer-and-settle-per-watermark logic to a C++ node/collector that emits
  coalesced rows).

`_LazyDag` becomes a thin Python shell that constructs the C++ driver and yields
its rows, mirroring how `merge`'s lazy path became a thin wrapper over
`MergeLazyPuller`.

**Risk:** moderate-high. This is the delicate piece - the watermark join
semantics (buffer unsettled events, settle every index strictly below
`min(watermark)`, suppress until every output has a value) must be ported
faithfully, and the single-output fast path kept. It is the one item that
warrants implementing on its own, after 1 and 2.

## Correctness and testing

- The invariant is `batch == lazy == graph`, byte-identical. The existing suites
  are the guard and must all stay green: `test_dag_lazy`, `test_dag_stream`,
  `test_stream_vs_batch`, `test_stream_vs_generator`, `test_combine_latest_flush`,
  `test_dag_exec`, `test_streams_*`.
- Each item is validated by re-running these before/after; no test is weakened.
- The watermark-join port (item 3) needs the multi-output, independent-drain-rate
  cases exercised explicitly (a pipeline whose two outputs fire at different
  rates over the same index axis), which the lazy dag tests already cover; add a
  hand-constructed case if coverage is thin.

## Scope and staging

- **In:** items 1, 2, 3 above.
- **Order:** 1 then 2 (both small, self-contained, delete Python) first; 3 (the
  C++ lazy driver + watermark join) as its own focused effort.
- **Out:** `_resample_ohlcv` (sanctioned composition); the datetime offset
  parsing in resample (binding-specific input adaptation, stays Python); the two
  capability-gap operators (`stack_lags`, bars->bars resample) which are new C++
  ops, not lazy-path fixes.

## Open decisions

1. **Unify combine_latest through the node (items 1 + 2 converge):** the cleanest
   design routes BOTH the eager and the lazy combine_latest paths through the
   single graph `CombineLatestNode` (which already coalesces per index),
   retiring `combine_latest_batch` and `_combine_latest_zip_lazy` together, so
   there is exactly one combine_latest implementation. Decide whether to do that
   full unification, or the narrower per-item fixes (route lazy through
   `CombineLatestPuller`; give `combine_latest_batch` the node's coalescing).
   Either removes the Python; the unification also removes a C++ dual-impl.
2. **Item 3 collector shape:** whether the watermark join is a new
   `dag/collector` variant or folded into the existing `Collector`/`Sink`. To be
   decided during implementation against `dag/collector.h`.
