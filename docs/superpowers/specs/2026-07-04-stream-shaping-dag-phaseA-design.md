# Stream shaping in the DAG — Phase A: `dropna` + `select`

**Status:** design approved 2026-07-04 (pending written-spec review)
**Scope:** Make two shape/cardinality stream operations usable **inside a
compiled `Dag`** as pure-C++ push-nodes: `dropna` (drop NaN rows) and a new
`select` (pick columns from a wide stream). Both also work **eagerly** and as
`_iter` streaming generators, consistent with the existing combinators.

This is **Phase A** of the stream-shaping effort. **Phase B** (`resample` —
windowed downsampling by key-interval or event-count, with reducers and an
end-of-input flush) is a separate, larger design with its own spec. `filter`
with a Python predicate is explicitly **dropped** from the graph (the engine
forbids Python callbacks — "no lambda"); the old `split` (partition a merged
tagged stream by source) is **replaced** by the more fundamental column
`select`.

## Background

The DAG engine (`screamer::dag`, C++) compiles a `Node` graph and runs it in
batch and live-streaming modes with byte-identical results. Today a graph node
is one of: **Input**, **Functor** (`EvalOp`, shape-preserving compute), or
**combine_latest** (alignment-only fan-in). The cardinality-changing stream
combinators `dropna`/`filter`/`split` exist only as **Python** functions in
`screamer/streams.py` (eager + `_iter` twins); there are **no C++ operators**
for them, and they cannot appear in a `Dag`.

The push-graph substrate already supports cardinality change: a node is a
`dag::Sink<Key>` whose `push(Frame)` may choose to emit downstream or not, and
`Frame{key, const double* values, width}` carries a variable-width value span
(`frame.h`). `Sink` already declares a `flush()` hook (default no-op). Nodes
follow the `FunctorNode` pattern (`functor_node.h`): hold a downstream `Sink&`,
transform on `push`, forward `flush`.

## The two operations

### `dropna` — drop NaN rows (cardinality-reducing)

**Eager (exists today, unchanged):** `dropna(keys, values, how="any")` — returns
`(keys', values')` with rows removed. `values` may be 1-D `(M,)` or 2-D `(M, N)`.
`how="any"` drops a row if any element is NaN; `how="all"` drops only all-NaN
rows. For 1-D, `any` and `all` coincide.

**Graph (new):** `dropna(stream, how="any")` where `stream` is a `Node`. Builds a
graph node. Semantics identical to eager, event by event: on each frame, apply
the NaN test to the `width` values; emit the frame unchanged if it survives,
drop it otherwise.

### `select` — pick columns from a wide stream (width-reducing, new)

**Eager (new):** `select(keys, values, columns)` where `columns` is an `int` or a
sequence of `int`. Returns `(keys, values[:, columns])`; a scalar `int` yields a
1-D result, a list yields a 2-D result with those columns in the given order.
Row count and keys are unchanged (shape op, not cardinality). Indices are
validated against the input width; out-of-range raises `ValueError`. Negative
indices are **not** supported (explicit, to avoid silent surprises) — raise.

**`_iter` (new):** `select_iter(events, columns)` — the streaming-generator twin,
matching the existing `*_iter` combinators.

**Graph (new):** `select(stream, columns)` where `stream` is a `Node`. Builds a
graph node emitting a width-`len(columns)` (or width-1) frame per input frame.

## Architecture

### C++ push-nodes (`include/screamer/dag/`)

**`DropNaNode<Key>`** (new `dropna_node.h`) — a `Sink<Key>` holding a downstream
`Sink&` and a `how` policy (an enum `NanDrop{Any, All}`). On `push`: scan
`f.values[0..width)`; under `Any` emit iff none are NaN, under `All` emit iff not
all are NaN. Forward `flush`. Zero per-event allocation (no buffers; it forwards
the incoming frame pointer unchanged when surviving).

**`SelectNode<Key>`** (new `select_node.h`) — a `Sink<Key>` holding a downstream
`Sink&`, a `std::vector<std::size_t> columns`, and a reused `std::vector<double>
out_(columns.size())`. On `push`: validate `f.width` covers the max index
(throw once on first violation), gather `out_[j] = f.values[columns[j]]`, emit
`Frame{f.key, out_.data(), out_.size()}`. Forward `flush`. One reused buffer,
zero per-event allocation.

Both mirror `FunctorNode`'s structure (single responsibility, reused buffer,
flush forwarding) — the anti-bloat, one-pattern approach.

### Graph builder + compiler

`GraphSpec`/`GraphBuilder` (`graph.h`) gain:
- `std::size_t add_dropna(std::vector<std::size_t> inputs, bool how_all)` — one
  input; `how_all` selects `All` vs `Any`.
- `std::size_t add_select(std::vector<std::size_t> inputs, std::vector<std::size_t> columns)`
  — one input; the column index list.

`CompiledGraph::compile()` (`compiled_graph.h`) wires these like the existing
single-upstream nodes: resolve the input sink, construct the node with its
downstream, register it. Both are single-input, single-output in node-degree
(one upstream, fan-out via the existing `Broadcast` if multiple consumers).

### Bindings (`bindings/bindings_dag.cpp`)

`PyGraphBuilder` gains `add_dropna(inputs, how_all)` and
`add_select(inputs, columns)` thin forwarders, exposed on the `_GraphBuilder`
class alongside `add_combine_latest`/`add_functor`.

### Python dispatch (`screamer/streams.py` + `screamer/dag.py`)

- `dropna` and `select` detect a `Node` first argument via `is_node` and return
  `make_combinator_node(fn, (stream,), kwargs)` — the same pattern
  `combine_latest`/`merge` already use. Eager array inputs take the existing
  path.
- `dag.py`'s `build()` dispatch maps the combinator function name to the builder
  call: `"dropna"` → `gb.add_dropna(inp, kwargs["how"] == "all")`; `"select"` →
  `gb.add_select(inp, _normalize_columns(kwargs["columns"]))`. Any other
  combinator remains rejected (`combine_latest` stays the only alignment op;
  `merge` is input-routing only).

The eager/graph signature reconciliation: eager single-stream ops take
`(keys, values, ...)`; the graph form takes a single `Node` standing for the
whole `(keys, values)` stream (a `Node` bundles both), detected by `is_node` on
the first argument. `select`'s `columns` is passed positionally after the node
(`select(stream, [0, 2])`) or as a keyword.

## Testing

Reuse the existing identity harness (`tests/_dag_oracle.py`,
`tests/test_dag_identity.py`) and combinator tests.

- **Eager `select`:** column int and list; order preservation; 1-D vs 2-D result;
  out-of-range and negative-index errors; keys unchanged.
- **Eager `dropna`:** unchanged behavior still passes (regression).
- **Graph `dropna`:** a `Dag` with `dropna(input)` produces the same
  `(keys, values)` as the eager `dropna` on the same data — **batch == stream ==
  eager-oracle**, including `how="any"` vs `how="all"` on a wide stream, an
  all-surviving stream, an all-dropped stream, and interior drops.
- **Graph `select`:** a `Dag` with `select(input, cols)` equals eager `select`;
  batch == stream; composed with a downstream functor (e.g.
  `RollingMean(select(input, 1))`) and with `combine_latest`
  (`select(combine_latest(a, b), 0)`).
- **Composition/cardinality:** `dropna` before a functor changes downstream row
  count consistently in batch and stream; `select` narrows width feeding a
  1-input functor.
- **Errors:** `select` with an out-of-range column in a graph fails at build or
  first event with a clear message; `filter` with a Python predicate on a `Node`
  raises "not supported as a DAG graph node" (the no-lambda guard), mirroring the
  existing `combine_latest(func=...)` rejection.

## Non-goals (Phase B / dropped)

- **`resample`** (by key-interval and by event-count; reducers
  `first/last/min/max/sum/count/mean/ohlc`; left/left boundaries with
  `label="right"` allowed; end-of-input flush so the trailing partial bucket
  emits and batch == streaming holds). Separate spec — it needs stateful
  accumulators and an explicit engine flush/end-of-stream signal (the `Sink`
  `flush()` seam already exists for it).
- **`filter` with a Python predicate in the graph** — forbidden by the
  C++-only-ops principle. Stays eager-only.
- **`split` by source tag in the graph** — superseded by `select`.
