# Node core step 3: API collapse, Filter mask-gate, Resample fold, Stream removal

**Status:** design (execution-ordering + concrete decisions). Direction pre-approved in the v2 architecture spec (`2026-07-11-unified-cpp-node-core-architecture-design.md`); this doc pins the mechanics and the order so the plans are unambiguous. Breaking API change (approved: "legacy removed, breaking is fine").

## Goal (the two user goals, remaining half)

1. All operator COMPUTE in the C++ core (thin bindings). Remaining: `dropna`/`select` port off numpy/generators; `filter` off the Python predicate; `resample` semantics onto the node.
2. One API shape: `Op(config)(data)`, CamelCase, for every operator. Remaining: rename the stream operators from functions `op(data, cfg)` to config-classes, delete `Stream`.

## Settled naming (from the v2 spec)

Stream operators become CamelCase config-classes, matching the ~190 existing functors (`Asin()(x)`, `RollingMean(5)(x)`):
`resample`->`Resample`, `dropna`->`Dropna`, `select`->`Select`, `filter`->`Filter`, `combine_latest`->`CombineLatest`, `merge`->`Merge`. `replay`/`split` stay functions (they are drivers/utilities, not stream operators). No lowercase aliases (legacy removed).

## Data model after `Stream` removal (index-as-data)

The `Stream` class is deleted. A stream value at the Python boundary is:
- **positional**: bare `values` (ndarray / list / scalar), index is implicit row-number.
- **indexed**: a `(values, index)` pair (index is an ndarray). This is already how `merge`/`combine_latest` return today.
- **lazy**: an iterator of `(value, index)` events (indexed) or bare `value` (positional) - unchanged from today's lazy contract.
- **graph**: a `Node` - unchanged.

So Stream removal = replace the `Stream` CLASS with the `(values, index)` tuple convention everywhere it was used, and route dispatch (Rule A) on the tuple/array/iterator/Node shape instead of on `isinstance(x, Stream)`. `_adapt`/`_to_streams`/`_regime`/`_normalize_streams` are rewritten to the tuple convention; the "stream" regime disappears (folds into indexed-tuple).

Rule A dispatch (unchanged in spirit): scalar->scalar, ndarray->ndarray (with the length-1/0-d rule already fixed), list->list, `(values, index)`->`(values, index)`, lazy iterator->lazy iterator, `Node`->`Node`. Multi-input ops apply variadically.

## Per-operator target signatures

- `CombineLatest(emit="when_all")(a, b, ...)` -> aligned `(rows, index)` / rows / iterator / Node. (Rename of `combine_latest`; drop the `func=` arg - compose a functor on the output instead, as the graph path already requires.)
- `Merge()(a, b, ...)` -> `(values, index_or_None, sources)`. Input routing; raises on Node (unchanged). (Rename of `merge`.)
- `Dropna(how="any")(data)` -> filtered stream. Node-backed: 1-D through `Input->DropNaNode`; 2-D through `CombineLatest(*columns)->DropNaNode` (combine_latest positional IS the column pack - no new engine feature). Lazy through the lazy driver over `DropNaNode`. Deletes the numpy mask and `_dropna_lazy`.
- `Select(columns)(data)` -> projected stream. Node-backed via `SelectNode`, same routing as Dropna. Deletes the numpy pick and `_select_lazy`.
- `Filter()(data, mask)` -> a 2-input MASK GATE: emit each `data` row whose aligned `mask` value is nonzero (a float zero-test; NaN drops the row). NO Python predicate. New `FilterNode` (2-in, combine_latest-style alignment of data+mask, emits the data row when the mask port's latest value is nonzero). The mask is built upstream from the comparison/logic family shipped in step 2 (e.g. `Filter()(x, GreaterThan()(x, 0.0))`). The old `filter(values, predicate)` and `_filter_lazy` are deleted.
- `Resample(freq=None, count=None, agg="last", origin="epoch", label="right", fill="skip")(data)` -> resampled stream. Carries the parked `feat/resample-redesign` semantics (Tasks 1-4): `freq` = index/time window, `count` = arrival bars, mutually exclusive (Option B); `agg` is a functor or a short string synonym, applied per bar and reset per bar. Folded onto the current `Resample` node path.

## Execution order (minimize test/doc rework)

Tests and docs are the bulk of the churn; each cross-cutting pass rewrites them. To write them once against the FINAL surface, do the surface/data-model change per operator-cluster, node-backing included, not as three separate sweeps:

- **Plan 3A - `Filter` mask-gate (self-contained, highest signal).** Add `FilterNode` (C++), the `Filter` class, delete `filter`/`_filter_lazy`. Uses the step-2 comparison family. Smallest blast radius (only `filter`'s callers), validates the mask-gate + 2-in-node design before the big surface change. Does NOT touch `Stream` yet (Filter's inputs/outputs use whatever the current convention is; it is re-touched only trivially by 3D).
- **Plan 3B - `Resample` fold.** Bring the parked `feat/resample-redesign` freq/count/agg semantics onto `Resample`. Self-contained to resample + its tests.
- **Plan 3C - `Dropna`/`Select` node-backing.** Route eager (1-D `Input`, 2-D `CombineLatest`-pack) + lazy through the C++ nodes; delete numpy/generator duplicates. Byte-identical oracle.
- **Plan 3D - the coordinated breaking surface pass: CamelCase + `Op(config)(data)` + `Stream` removal.** Rename every stream operator to its config-class, delete `Stream` and the lowercase functions, rewrite `_adapt`/`_to_streams`/`_regime`/`_normalize_streams` to the tuple convention, and update ALL tests + docs to the final surface in one pass. This is the big mechanical-but-wide change; doing it last means it rewrites each test file exactly once, over operators whose compute is already final.
- **Plan 3E - docs/notebooks rewrite** to the one shape + the "Dag as a reusable user-defined function" framing (the deferred `project_docs_resample_dag_rewrite` task).

Rationale for last-not-first on 3D: 3A/3B/3C change operator INTERNALS + add nodes; their per-operator tests get written against the current surface, then 3D rewrites the surface once. The alternative (rename first) would force 3A/3B/3C to also carry Stream-removal semantics before the data-model helpers exist. Filter/Resample signatures are NOT mechanical renames (predicate->mask, every/count->freq/count), so they cannot ride a pure-rename sweep anyway - they need their own plans (3A/3B) regardless of order.

## Invariants (unchanged)

- `batch == lazy == graph`, bit-for-bit, per operator (the three drivers over one node).
- Causality (no lookahead).
- Suite green after each plan; each rerouted path asserts byte-identical to its pre-change output.
- Preserve the C++ `EvalOp` math impls + the Dag engine + the reused `streams::` operators/drivers. No em-dashes / no ` -- `. No version-file edits.

## Deferred / open

- **Multi-column INPUT** (width-N `add_input` + wide `push_event` + `_LazyDag` row push): NOT needed if `Dropna`/`Select` 2-D routes through `CombineLatest`-pack (Plan 3C). Add it later only if a first-class width-N input proves necessary; the pack routing covers the current cases.
- **Fast-eval signal** (node capability flag vs synchronous method): unchanged open item from the v2 spec; resolve when batch math perf is measured, not blocking here.
