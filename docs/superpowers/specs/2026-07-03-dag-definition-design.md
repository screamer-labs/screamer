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
- **A `Dag` is a plain function.** A compiled `Dag` presents the same interface
  as any other screamer callable: positional `N` inputs → `M` outputs (single
  return for `M == 1`, tuple otherwise), the same per-slot input polymorphism,
  and — because a `Dag` is itself graph-capable — it nests inside other DAGs. By
  default it aligns its outputs (via `combine_latest`) so multiple returns are
  co-indexed and equal-length, exactly like a multi-output functor; it is a
  strict *superset* of a plain function, additionally accepting multi-clock
  inputs the ordinary contract can't express.
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

### `Dag(inputs=[...], outputs=[...], align_outputs=True)`
A compiled `Dag` is a **positional N-input / M-output callable**, indistinguishable
from a built-in `FunctorBase<_, N, M>` — that is the whole point (see "Component 4
— the boundary"). Ordered lists define the signature (like Keras
`Model(inputs=..., outputs=...)`):

```python
dag = Dag(inputs=[price_a, price_b], outputs=[z, spread])
```

- `inputs` is an ordered list of `Input` nodes → the positional parameter order.
  (Their `name`s also enable keyword calls, a Python convenience.)
- `outputs` is an ordered list of nodes → the positional return order.
- Validates at construction (see Validation below).
- Is itself callable (Component 4) — and, because a `Dag` is graph-capable like
  any op, it **composes**: pass `Node`s to a `Dag` and it builds a node, so a
  `Dag` nests inside another `Dag`.

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

## Component 4 — the boundary (a `Dag` behaves like a plain function)

A compiled `Dag` presents the **same interface as any other screamer function**:
positional `N` inputs in, `M` outputs out, with the same input polymorphism per
slot.

```python
out         = dag(sa)                       # 1 in, 1 out  -> single return
z_s, spr_s  = dag(sa, sb)                   # N in, M out  -> tuple, unpackable
z_s, spr_s  = dag(price_a=sa, price_b=sb)   # kwargs by Input name (Python bonus)
```

**Per-slot input.** Each input slot accepts what the combinators accept: a bare
value array (row-number keys), a `(keys, values)` pair, a pandas Series
(index = keys), or a `Node` (→ the `Dag` composes into a bigger graph). Feeding
co-indexed inputs is exactly the ordinary N-input functor case; differently-
clocked inputs are handled by explicit `combine_latest` **inside** the graph
(alignment stays its own layer). A `Dag` is thus a strict **superset** of a plain
function — it accepts everything a normal N-input functor does, plus multi-clock
inputs the ordinary contract can't express.

**Return shape — aligned by default.**
- **`align_outputs=True` (default):** the `Dag` joins its `M` outputs with
  `combine_latest` onto their common key axis (union of output keys,
  `emit="when_all"` — the combinator's own default). The outputs become
  **co-indexed and equal-length**, returned as one shared `keys` plus a stackable
  `(T, M)` value block (single stream for `M == 1`). When the outputs are already
  co-indexed (no internal `dropna`/`resample`), this join is a no-op. This makes
  a `Dag` a drop-in `N→M` functor — the multiple returns are the same length,
  matching every other multi-output function in the library.
- **`align_outputs=False` (opt-out):** outputs are returned as a tuple of `M`
  independent `(keys, values)` streams, possibly of different clocks/lengths —
  for the advanced case where you deliberately want unaligned outputs.

**Execution.** Evaluate nodes in dependency order, **memoizing each node's
result** so a fan-out intermediate is computed exactly once and shared by all
consumers. Per node:
- **Input:** result = the fed stream for that positional/keyword slot.
- **Functor node:** `k, v = eval(input)`; apply the functor to `v` (via the
  `(T, N)` convention when an aligned width-`N` stream feeds an `N`-input
  functor); keys pass through → `(k, out)`.
- **Combinator node:** call the combinator on the inputs' `(keys, values)`
  results with the recorded kwargs → its `(keys, values|aligned)` result.

Then the output boundary applies the alignment above and returns single-or-tuple.
Cycles are impossible by construction (nodes are built bottom-up, so a node's
inputs always already exist).

## Validation

At `Dag(...)` construction:
- Every `Input` reachable from the outputs appears in `inputs=` (and vice
  versa) — a graph referencing an undeclared input, or declaring an unused one,
  raises. `inputs`/`outputs` order defines the positional signature.
- Output nodes are `Node`s and reachable from the declared inputs.
- The stateful-safety rule holds (no functor instance backs two nodes).
- Arity/width mismatches (e.g. a 2-input functor fed a width-3 aligned stream)
  surface from the underlying functions' own checks at execution, with the
  clearer messages from Component 1.

At `dag(...)` call:
- The number of positional args equals `N` (or all inputs supplied by keyword
  name); a wrong count / unknown keyword / missing input raises a clear error,
  as a normal function would.

## Testing

- **Equivalence:** a built DAG's batch output equals the hand-written expression
  it encodes (e.g. the `z` output equals `RollingZscore(100)(MovingAverage(30)
  (spread))` computed directly).
- **Function-like boundary:** `dag(sa)` returns a single stream for `M == 1`;
  `dag(sa, sb)` returns an unpackable tuple for `M > 1`; keyword calls
  (`dag(price_a=sa, price_b=sb)`) match positional; wrong arg count / unknown
  keyword raises like a normal function.
- **Aligned outputs (default):** with `align_outputs=True`, two outputs where one
  branch was `dropna`'d come back **co-indexed and equal-length** on the union
  key axis (== `combine_latest` of the two outputs); already-co-indexed outputs
  are unchanged. With `align_outputs=False`, the same DAG returns independent
  streams of their natural lengths.
- **Composition:** feeding a `Dag` into another `Dag` (Node args) builds a nested
  node and evaluates identically to the inlined graph.
- **Fan-out computes once:** an intermediate feeding two outputs is evaluated a
  single time (assert via a counting/spy op) and both consumers agree.
- **`(T, N)` convention:** `RollingCorr(20)(aligned)` equals
  `RollingCorr(20)(aligned[:,0], aligned[:,1])`; a wrong-width single array
  raises a clear error; `num_inputs` reports the right arity across the library.
- **Stateful-safety:** reusing a functor instance across nodes raises.
- **Validation:** undeclared/unused input, unreachable output, instance-reuse,
  and wrong call arity all raise clearly.

## Implementation-plan decomposition

Likely a single plan with these tasks: (1) the `(T, N)` input convention +
`num_inputs` arity (C++ + docs); (2) `Node`/`Input` and the combinator
Node-awareness; (3) the functor dispatch hook + `make_functor_node`; (4) `Dag`
construction (ordered `inputs`/`outputs`), validation, and the positional/keyword
call boundary incl. `align_outputs` default; (5) the memoized batch executor +
equivalence / fan-out / multi-output / composition tests.

## Open decisions carried into planning

- Final public names: `Node`, `Input`, `Dag`, `num_inputs`, `align_outputs`
  (working names).
- Exact `Node` marker used by the C++ hook to recognize a graph handle (e.g. an
  `isinstance` check against the bound `Node` type vs a duck-typed attribute).
- Per-slot input forms accepted at call time — bare value array (row-number
  keys), `(keys, values)` pair, pandas Series, iterator, `Node`; confirm the
  full set and their detection order (mirrors the combinators' carrier
  detection).
- Output-alignment key axis + firing rule default (proposed: union of output
  keys, `emit="when_all"`).
