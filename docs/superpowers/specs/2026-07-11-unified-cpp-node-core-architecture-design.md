# Unified C++ node core: one operator interface, one API shape

**Status:** architecture design, approved in discussion; awaiting written-spec review before planning.

## Goal

1. All operator functionality lives behind one C++ interface, so every language binding (Python now, JS/WASM later) is a thin driver over the same C++ core. No operator logic in a binding layer.
2. Collapse the two API shapes (functor classes `Op(config)(data)` and operator functions `op(data, config)`) into one: `Op(config)(data)` for every operator.

## Problem

- `dropna`, `select`, `filter`, and `merge`'s lazy path run compute in Python. `dropna`/`select` have two implementations of the same logic (numpy eager plus a C++ graph node) that can diverge (`batch != graph`) and are unreachable from a non-Python binding.
- Two unjustified API shapes: math ops are functor classes, stream ops are functions. The split is an artifact of C++ classes versus Python glue, not a design choice.
- The `Stream` type persists although index-as-data supersedes it.

`resample` and `combine_latest` already do it right (C++ compute in every regime, thin Python) and are the existence proof.

## The single node contract

Every operator is a C++ node with:

```
n_in    : number of input ports
n_out   : number of output columns per emitted row
reset()
push(input_index, value, index)  -> emits zero or more output rows (each: n_out values + an index)
flush()                          -> emits any trailing rows held past the last input
```

- **Variable output cardinality.** A push emits 0 (dropna of a NaN, resample mid-bar), 1 (a math functor, resample at a boundary), or many.
- **Multi-input.** `input_index` selects the port; a node with `n_in > 1` aligns internally (as-of / combine_latest semantics live inside the node, fed by per-port pushes).
- **Stateful.** State is per-instance; `reset()` clears it. One instance backs one graph position (the existing rule).
- **index is ordinary data** carried in push/emit. No separate index channel, no `Stream`.

The current `EvalOp` (synchronous N-in/M-out, one row in to one row out) is the special case: fixed `n_in`, emits exactly one row per push, holds no state past a row. The ~178 math functors are this case.

## Fast path (performance)

The push interface must not regress tight-loop batch math. The batch driver keeps a specialization: a 1-in / 1-out node that emits exactly one row per push and never flushes is driven by the existing synchronous eval loop over the contiguous array, with no per-event dispatch. The node declares this shape (a capability flag or a distinct synchronous method) so the batch driver picks the tight loop; general nodes use the push loop. High-speed positioning is preserved.

## The three drivers

All in C++ or a thin shell, generic over the node contract:

- **batch**: given input array(s), loop push over rows, collect emitted rows into an output array, then flush. Uses the 1-1 fast path when the node declares it.
- **lazy**: given input iterator(s), pull one event, push, yield each emitted row; on exhaustion, flush. This is the current `LazyEvalIterator` / `_LazyDag` generalized.
- **graph**: the Dag `CompiledGraph`, which already is push / drain / flush.

A binding exposes exactly the node classes plus the three drivers. Nothing operator-specific lives in the binding.

## API shape

`Op(config)(data)`, uniformly:

- config (window, freq, agg, how, columns, emit, predicate) in the constructor; data in the call.
- Rule A dispatch on the applied data: scalar to scalar, ndarray to ndarray, list to list, lazy iterator to lazy iterator, Node to Node. Multi-input ops apply variadically: `CombineLatest()(a, b, c)`.
- Operators are CamelCase classes. Stream operators are renamed: `resample` to `Resample`, `dropna` to `Dropna`, `select` to `Select`, `filter` to `Filter`, `combine_latest` to `CombineLatest`, `merge` to `Merge`. (Trades pandas/SQL lowercase familiarity for one coherent shape; see Open decisions.)
- agg composition: `Resample(freq=..., agg=ExpandingMean())` feeds each bar's rows to the agg functor and resets it per bar (the settled resample-agg design). agg accepts a functor or a short string synonym.

## index-as-data, Stream removal

index travels in push/emit. A stream is a sequence of `(value, index)` events, or bare values (positional). The `Stream` type is deleted; its roles (carrying an index, column labels) become `(value, index)` events and positional output columns. This subsumes the earlier index-as-data stage.

## filter: the one host callback

`Filter`'s predicate is host-language code. The C++ `Filter` node holds a callback and invokes it per pushed row to decide emit or drop. Python passes a Python callable; a JS/WASM binding passes a JS function. Everything else in `Filter` (the streaming scaffold) is C++. This is the single deliberate cross-boundary point; a pure-C++ use of `Filter` with no host function is therefore not possible.

## Operator inventory on the model

- **Math functors (~178):** 1-in-1-out (or N-in-1-out like `BOP`); emit one row per push; no flush; no cardinality change. Already conform; get the fast-eval path.
- **Dropna(how):** 1-in; emits 0 or 1 per push. No flush.
- **Select(columns):** 1-in (multi-column); emits 1 per push (projected columns).
- **Filter(predicate):** 1-in; emits 0 or 1 per push via the host callback.
- **CombineLatest(emit):** N-in; as-of align; emits a coalesced row when the join fires; last-value state per port.
- **Merge():** N-in; emits one row per push in index order, with `source` as an extra output column.
- **Resample(freq | count, agg, origin, label, fill):** 1-in; accumulates into the owned agg functor; emits a bar row at each boundary and on flush. Contextual `freq` (index/time window) and `count` (arrival) per the resample design; the two are mutually exclusive.

## Migration / sequencing

1. Define the C++ node contract and the three generic drivers; adapt the existing `EvalOp` math nodes to declare the fast-eval shape (no behavior change).
2. Port `Dropna`, `Select`, `Merge` to C++ nodes on the contract; delete the numpy-eager and Python-lazy duplicates. `Filter`: C++ scaffold plus host callback.
3. Collapse the API to `Op(config)(data)`; rename the stream operators to CamelCase; retire the function forms.
4. Fold `Resample` onto the contract, carrying the freq/agg/composition semantics already built (Tasks 1-4 on `feat/resample-redesign`); drop the resample-branch Task 5 (it migrates to the old shape).
5. Delete the `Stream` type; index-as-data throughout.
6. Rewrite docs and notebooks to the one shape, with the Dag-as-reusable-function framing.
7. Bindings: the thin Python shell is the reference; a JS/WASM binding is a later, separate project this enables.

Each step keeps the suite green and the `batch == lazy == graph` invariant.

## Preserved (do not rework)

- The C++ `EvalOp` math implementations and the Dag `CompiledGraph` engine (the push model). This is consolidation onto them, not a greenfield rewrite.
- Causality (no lookahead) and `batch == lazy == graph` equality as the oracle.
- Rule A container/rank dispatch (fixed this session).

## Open decisions

1. **Naming.** CamelCase for all operators (`Dropna`, `Select`, `Filter`, `CombineLatest`, `Merge`) is coherent but breaks pandas/SQL lowercase familiarity. Confirm, or keep lowercase aliases.
2. **Fast-eval signal.** A node capability flag versus a separate synchronous method on the contract. Implementation-level; resolve at build.
3. **Joins.** Confirm as-of (`combine_latest`) is the only built-in multi-input join in v1 (no strict join), consistent with the earlier streaming spec.
4. **filter callback under WASM.** Accept that `Filter` requires a host function; everything else is host-free.
5. **Scope.** This supersedes the standalone resample redesign and the index-as-data / Stream-retirement stages; they fold in here. Confirm the resample branch is parked (not merged) and its semantics migrate into `Resample`.

## Testing / invariants

- `batch == lazy == graph`, bit-for-bit, for every operator (the three drivers over one node).
- Causality re-verified (no lookahead).
- Rule A dispatch across scalar / array / list / iterator / node for every operator.
- Per-operator numeric oracles unchanged from today: the compute logic moves language, not results. The `Dropna`/`Select`/`Merge` port to C++ must match the current numpy/Python results exactly.
