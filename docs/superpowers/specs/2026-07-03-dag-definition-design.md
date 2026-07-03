# Computational DAG — definition + naive batch executor (DAG-1)

**Status:** design approved, ready for planning
**Date:** 2026-07-03
**Scope:** DAG-1 only — the Python definition layer and a batch executor that
evaluates the graph using the existing functions. The compiled all-C++
push-graph executor (DAG-2) and live streaming get their own spec afterward.

## Motivation

The multi-stream foundation (see
`2026-07-02-multistream-foundation-design.md`) gives us operators that combine,
split, filter, and align streams, all sharing one polymorphic call contract.
The natural next step is to **define a computation once as a graph** — one or
more input streams, intermediate streams, and one or more named output streams —
and run that definition, first as a backtest and later live.

This spec covers the *definition* of such a graph in Python and a straightforward
executor that runs it in batch. It deliberately reuses the existing functions
rather than inventing a new execution engine: correctness first, speed later.

## Design principles (carried from the foundation)

- **A `Node` is one more input kind.** The library already dispatches on input
  type (scalar → compute, array → compute, iterator → `LazyIterator`). A
  symbolic `Node` joins that table: a functor or combinator called on a `Node`
  returns a `Node` instead of computing. No separate "graph mode."
- **The DAG mirrors the library's polymorphism.** A compiled `Dag` is itself a
  callable that behaves like a bigger functor: batch in → batch out.
- **Definition is Python; execution reuses tested functions.** DAG-1 introduces
  essentially no new C++ (one dispatch hook + one small functor convention).
- **Cross-mode identity is preserved for later.** DAG-2 will compile the *same*
  node graph to a C++ push-graph with byte-identical results; DAG-1's executor
  is the reference.

## Non-goals (deferred)

- **DAG-2: the compiled C++ push-graph executor** and **live streaming**. DAG-1
  is batch-only. Streaming's natural, efficient home is the push-graph, so a
  throwaway Python event loop is not built here.
- **Function tracing (the `@dag` decorator form).** A thin wrapper over the
  symbolic-handle layer; can be added later.
- **Building DAGs from C++.** Python first; the C++ builder is a later effort.

## Component 1 — the `(T, N)` input convention for N-input functors

This is a precondition for the `combine_latest → functor` idiom and is useful on
its own, independent of DAGs.

**New input form:** an `N`-input functor accepts a **single 2-D array of shape
`(T, N)`**; its `N` columns are treated as the `N` inputs, column `j` → input
`j`. It is accepted **iff the second dimension equals `N`** (unambiguous match).
Any other single-array shape passed to an `N`-input functor remains an error,
now with a clear message (e.g. "RollingCorr expects 2 inputs; got a single array
with 3 columns").

```python
aligned = combine_latest((t_a, p_a), (t_b, p_b))[1]   # shape (T, 2)
RollingCorr(20)(aligned)                # == RollingCorr(20)(aligned[:, 0], aligned[:, 1])
```

- Fills a currently-erroring case (a single array to an N-input functor), so it
  conflicts with no existing behavior.
- Lives in the central N-input dispatch (`FunctorBase::handle_input_Ni_1o` and
  `handle_input_Ni_Mo`), so every N-input functor gains it uniformly.
- Documented in `docs/polymorphic_api.md` under the multi-input contract.
- Because `combine_latest` emits exactly `(T, N)`, the DAG executor needs no
  special column-splitting logic — it threads `values` through and the functor's
  own dispatch does the split.

**Arity exposure:** add a read-only property giving a functor's input count
(working name `num_inputs`; `ScreamerBase` reports 1, `FunctorBase<_,N,M>`
reports `N`). The convention needs it, and the DAG uses it for validation.

## Component 2 — the definition layer

### `Input(name)`
Creates a source `Node` — a named placeholder for a timed stream `(keys,
values)`. Duplicate names within one graph refer to the same input.

### `Node`
An immutable handle recording `(op, input_nodes)`, where `op` is one of:
- an **`Input(name)`** marker (a source),
- a **functor instance** (a configured `ScreamerBase`/`FunctorBase`, e.g.
  `MovingAverage(30)`), or
- a **combinator spec** — the combinator function plus its kwargs, e.g.
  `(combine_latest, {"emit": "when_all", "func": None})`.

A `Node` represents a stream and can fan out to many consumers. It is distinct
from a `LazyIterator` (a runtime, single-consumer stream): a `Node` is a
build-time definition.

### `Dag(outputs={name: node}, inputs=None)`
- Walks back from the output nodes to discover all reachable `Input` leaves —
  **inputs are inferred**; `inputs=[...]` is optional, for validation/ordering.
- Validates at construction (see Validation below).
- Is itself callable (Component 4).

## Component 3 — graph building via the dispatch hook

- **Functors (C++):** add a central hook to `ScreamerBase::operator()` and
  `FunctorBase::handle_input` — if any argument is a `Node`, do not compute;
  call a registered Python builder `screamer._dag.make_functor_node(self, args)`
  which returns a `Node` whose op is `self` (the configured functor instance)
  and whose inputs are the argument nodes. One change; every functor — current
  and future — becomes graph-capable.
- **Combinators (Python):** `merge`/`combine_latest`/`dropna`/`filter`/`split`
  check whether any argument is a `Node`; if so they return a `Node` (or, for
  `split`, a list of `Node`s) recording `(function, kwargs)` instead of running.
- **Stateful-safety rule:** a functor instance may back **at most one** node.
  Reusing one instance across nodes (`f = MovingAverage(30); f(x); f(y)`) raises
  a clear error, because functor state cannot be shared between graph positions.
  The inline idiom `MovingAverage(30)(x)` satisfies this naturally.

## Component 4 — the batch executor

`dag(feeds)` where `feeds` is `{input_name: (keys, values)}` returns
`{output_name: (keys, values)}` (an output backed by `combine_latest` returns
`(keys, aligned)`).

- Evaluate nodes in dependency order, **memoizing each node's result** so a
  fan-out intermediate is computed exactly once and shared by all consumers.
- Per node:
  - **Input:** result = `feeds[name]`.
  - **Functor node:** `k, v = eval(input)`; apply the functor to `v` (via the
    `(T, N)` convention when the input is an aligned width-N stream feeding an
    N-input functor); keys pass through → `(k, out)`.
  - **Combinator node:** call the combinator on the inputs' `(keys, values)`
    results with the recorded kwargs → its `(keys, values|aligned)` result.
- Collect the outputs by name.

Cycles are impossible by construction (nodes are built bottom-up, so a node's
inputs always already exist).

## Validation

At `Dag(...)` construction:
- Every reachable `Input` is discovered and, if `inputs=` was given, matches it.
- Output nodes are `Node`s and reachable.
- The stateful-safety rule holds (no functor instance backs two nodes).
- Arity/width mismatches (e.g. a 2-input functor fed a width-3 aligned stream)
  surface from the underlying functions' own checks at execution, with the
  clearer messages from Component 1.

At `dag(feeds)` call:
- Every inferred input has a feed; missing/extra feeds raise a clear error.

## Testing

- **Equivalence:** a built DAG's batch output equals the hand-written expression
  it encodes (e.g. `dag(...)["z"]` equals `RollingZscore(100)(MovingAverage(30)
  (spread))` computed directly).
- **Fan-out computes once:** an intermediate feeding two outputs is evaluated a
  single time (assert via a counting/spy op) and both consumers agree.
- **Multi-output:** a DAG with several named outputs returns all of them.
- **`(T, N)` convention:** `RollingCorr(20)(aligned)` equals
  `RollingCorr(20)(aligned[:,0], aligned[:,1])`; a wrong-width single array
  raises a clear error; `num_inputs` reports the right arity across the library.
- **Stateful-safety:** reusing a functor instance across nodes raises.
- **Validation:** missing feed, unknown output, instance-reuse all raise clearly.

## Implementation-plan decomposition

Likely a single plan with these tasks: (1) the `(T, N)` input convention +
`num_inputs` arity (C++ + docs); (2) `Node`/`Input` and the combinator
Node-awareness; (3) the functor dispatch hook + `make_functor_node`; (4) `Dag`
construction, input inference, and validation; (5) the memoized batch executor +
equivalence/fan-out/multi-output tests.

## Open decisions carried into planning

- Final public names: `Node`, `Input`, `Dag`, `num_inputs` (working names).
- Exact `Node` marker used by the C++ hook to recognize a graph handle (e.g. an
  `isinstance` check against the bound `Node` type vs a duck-typed attribute).
- Whether `dag(feeds)` accepts bare value arrays (row-number keys) as a
  convenience in addition to `(keys, values)` pairs.
