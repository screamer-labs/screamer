# Computational DAG — the C++ push-graph engine (DAG-2)

**Status:** design approved, ready for planning
**Date:** 2026-07-03
**Scope:** The all-C++ execution engine for the DAG defined in DAG-1: live
streaming and batch over one compiled push-graph, with the graph representation,
compiler, and executor centralized in C++ so every binding (Python now,
JS/WASM later) is thin and pure-C++ use needs no binding at all. Replaces
DAG-1's Python executor (which is retained only as a test-time reference oracle).

## Motivation

DAG-1 gives correct batch DAGs, but its executor is Python and batch-only. The
real prize is **live streaming at C++ speed** — pushing events one at a time
through the whole graph with no Python per event — which makes the
backtest/live-parity story real: define a DAG, backtest it, run the *same* DAG
live, byte-identical.

Two further requirements shape the design:

- **Centralization / no bloat.** Logic must live in C++ exactly once. Moving
  execution or graph-building into a binding language means re-implementing it
  for every future binding (JS/WASM), which is duplication and an alignment
  risk. Bindings are thin adapters; a pure-C++ user calls the core directly.
- **Identity as a structural guarantee.** Batch and streaming must always agree.
  We make that true *by construction*, not just by test: both are thin drivers
  over the *same* compiled push-graph.

## Principles

- **The C++ core owns the graph representation, the compile step, and both
  execution modes.** Bindings only build the graph through a C++ builder and
  marshal arrays/events across the boundary.
- **Identity by construction.** Streaming feeds live events; batch *replays the
  input arrays as events* through the identical graph. The per-node code runs
  once, the same way, in both modes.
- **Graph node ops are C++-only.** A node op is a C++ functor or a C++
  combinator — never an arbitrary Python callable (a Python lambda cannot run in
  the C++ engine and would not be WASM-portable). This reinforces the
  foundation's *alignment ≠ computation*: in a graph, `combine_latest` is
  alignment-only and arithmetic/reduction is done by C++ functor nodes.
- **One implementation, reused everywhere.** The stateful cores already in C++
  (`CombineLatest`, `MergeSource`, `FunctorNode`, `Sink`/`Source`) are reused
  directly; nothing computational is added in Python.

## Non-goals (deferred)

- **The WASM/JS binding itself.** DAG-2 only ensures logic sits where a thin WASM
  binding *could* reuse it. Building that binding is a separate, later effort.
- **DAG-2c: `dropna`/`filter`/`split` as C++ push-nodes** (cardinality ops in the
  graph). A later increment; the engine ships first with functors + `combine_latest`.
- **Python-lambda reducers in graphs.** Removed from graph-mode by the C++-only-ops
  principle. (Eager, non-graph `combine_latest(..., func=lambda)` may keep it —
  that path is unaffected.)

## Component 1 (DAG-2a) — uniform op interface + arithmetic functors

The graph must hold *any* functor behind one pointer. Today `ScreamerBase`
(1→1) and `FunctorBase<_,N,M>` (N→M) share no common base — the recurring wall.
The centralized fix is a single small virtual interface both implement:

```cpp
struct EvalOp {
    virtual ~EvalOp() = default;
    virtual std::size_t n_in()  const = 0;
    virtual std::size_t n_out() const = 0;
    virtual void eval(const double* in, double* out) = 0;  // one event
    virtual void reset() = 0;
};
```

- `ScreamerBase::eval` writes `out[0] = process_scalar(in[0])`; `n_in=n_out=1`.
- `FunctorBase<_,N,M>::eval` calls `call({in[0..N-1]})` and writes the `M`
  results; `n_in=N`, `n_out=M`.

Then a graph node holds one `EvalOp*` for any functor, and **one** push-node type
drives them all (no per-arity node families — the anti-bloat move).

Also add the **binary arithmetic functors** `Add`/`Sub`/`Mul`/`Div` (2→1),
following the existing 2-input math pattern (`Atan2`, `Hypot`). These are the
graph's reduction vocabulary (`Sub()(combine_latest(a, b))` replaces the DAG-1
`func=lambda x,y: x-y`) and are independently useful eagerly.

## Component 2 (DAG-2b) — C++ graph representation + builder

A C++ `Graph` holds nodes and edges. A node is one of:
- **Input** — a named/indexed source slot.
- **Functor** — an `EvalOp*` (the configured functor instance) + its input node ids.
- **Combinator** — a kind (`combine_latest`, `merge`) + params + input node ids.

A C++ `GraphBuilder` exposes `add_input`, `add_functor(EvalOp*, inputs)`,
`add_combinator(kind, params, inputs)`, returning node ids. Bindings call this;
the Python `Node`/`Input`/`Dag` become **thin handles** over C++ node ids (the
DAG-1 dispatch hook now builds a C++ graph node instead of a Python one).

## Component 3 (DAG-2b) — the push-node library

Reuses the foundation's `Sink`/`Source`/`Event` and existing operators:
- **FunctorNode** — generalized to drive an `EvalOp` (any arity), emitting the
  result; keys pass through (shape-preserving).
- **CombineLatestNode** — a push fan-in `Sink` wrapping the existing
  `CombineLatest` operator (`on_event`/`latest`); N input ports, emits the
  aligned row when warm.
- **Broadcast/fan-out** — a node with multiple downstream sinks (the current
  single-downstream `Sink` wiring is generalized to a sink list) so a shared
  intermediate feeds all consumers.
- **CollectorSink** — terminal, gathers an output stream (batch) or forwards it
  (streaming).

**Open design point (resolve in planning):** how an aligned `combine_latest`
output (N values/event) reaches an N-input functor node — either a *wide edge*
(an event carrying a small fixed-width value vector) or *fusion* of
`combine_latest` with its immediately-downstream N-input functor into one node.
Fusion keeps events single-valued but can't apply when the aligned stream fans
out or is itself an output; the wide edge is more general. The plan picks one
(leaning wide edge for generality) with the tradeoff documented.

## Component 4 (DAG-2b) — the compiler

`GraphBuilder.compile()` walks the graph and produces a wired push-graph:
instantiate one push-node per graph node, wire each node's sink(s) to its
consumers (fan-out via the sink list), route each Input to its consumers, and
drive all Inputs through one `MergeSource` (so multi-clock inputs interleave in
key order). Output nodes get `CollectorSink`s; `align_outputs` wires the output
nodes into a final `combine_latest` (matching DAG-1's boundary semantics).

## Component 5 (DAG-2b) — the two drivers (structural identity)

One compiled graph, two thin drivers:
- **Batch** — build a replay `Source` over each input's `(keys, values)` arrays,
  merge them, pump every event through the graph, collect outputs into arrays.
- **Streaming** — feed live `(key, value)` events into the same graph as they
  arrive; emit output events live.

Because the node-processing code is identical, batch == streaming **by
construction**. The Python-facing `Dag` is polymorphic exactly as in DAG-1
(`dag(batch)` → arrays; `dag.stream(feeds)` → live events).

## Component 6 — thin Python binding + DAG-1 as oracle

`screamer/dag.py` shrinks to a thin builder + I/O marshaller over the C++ graph.
DAG-1's Python `_run` executor is **removed from production** and relocated into
the test suite as a **reference oracle**: every DAG-2 result (batch and stream)
is asserted byte-identical to the oracle over a matrix of graphs and inputs.

## Testing

- **Oracle equivalence:** DAG-2 batch output == DAG-1 reference-oracle output,
  bit-for-bit (`assert_array_equal`), across a matrix of graph shapes (chains,
  fan-out, multi-input functors over `combine_latest`, multi-output, `align`
  on/off), key dtypes, and randomized inputs.
- **Structural batch==stream:** the same compiled graph driven batch vs.
  streaming yields identical event sequences.
- **Arithmetic functors:** `Add`/`Sub`/`Mul`/`Div` eager correctness and the
  `Sub()(combine_latest(a,b))` spread idiom equals the DAG-1 `func=` result.
- **`EvalOp`:** `n_in`/`n_out`/`eval` agree with `process_scalar`/`call` across
  the library; `eval` per event equals the batch array path (foundation identity).
- **Fan-out once, validation, composition:** carried from DAG-1's contract,
  re-checked against the C++ engine.
- **Hot path:** the `EvalOp` addition must not add per-call cost to eager
  scalar/array functor calls.

## Implementation-plan decomposition

1. **DAG-2a** — `EvalOp` interface on `ScreamerBase`/`FunctorBase` + `Add`/`Sub`/
   `Mul`/`Div` functors (C++ + eager tests). Independently useful; unblocks the engine.
2. **DAG-2b** — the engine: C++ graph repr + builder, push-node library
   (generalized `FunctorNode`, `CombineLatestNode`, fan-out, collectors, the
   aligned-edge mechanism), compiler, batch + streaming drivers, `align_outputs`;
   thin `dag.py` cutover; DAG-1 `_run` → test oracle. (Its own multi-task plan.)
3. **DAG-2c** (later) — `dropna`/`filter`/`split` as C++ push-nodes (cardinality
   ops in the graph) for full parity and eventual WASM.

## Open decisions carried into planning

- The aligned-edge mechanism (wide edge vs `combine_latest`+functor fusion).
- `EvalOp` naming and exact placement (a mixin base vs added directly to
  `ScreamerBase`/`FunctorBase`); how bindings hand an `EvalOp*` to the builder
  (the functor is a bound object → recover its `EvalOp*`).
- Streaming input API shape (push events per input vs one merged event iterator).
- Whether `merge` is a graph combinator in DAG-2b or only the internal input
  driver (it may not need a user-facing graph node initially).
