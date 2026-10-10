# Shape contract: time, assets, and fields

Status: approved design (2026-10-11), phase 1 to be planned.

## Problem

A second axis means different things in different layers of screamer, and
nothing states which:

| Layer | What axis 1 (or a frame's width) means today |
|---|---|
| Batch operators on `(T, K)` | K independent series (assets), one state each |
| Multi-output results `(T, K, M)` | M fields of one record |
| A single `(T, N)` array passed to an N-input operator | N fields (the inputs) |
| Graph frames inside a `Pipeline` | fields of one event (a width-4 frame is one OHLC bar) |
| Row operators (`Cross*`, portfolio engines) | K assets of one time step |
| JS reducers (`wrapReducerOp`) | "groups", a third name for assets |

Where the meanings meet, the library fails or is inconsistent:

- Multi-input operators do not broadcast. `BacktestPriceTarget()(targets (T, K),
  price (T,))`, `RollingCorr()(X (T, K), y (T,))` and `Add()(X, y[:, None])`
  raise "dimensions mismatch". `position_test`, `strategy_test` and
  `walk_forward` broadcast by hand in Python (`screamer/validation.py`), and
  `js/src/validation.ts` repeats it.
- An iterator of `(K,)` rows raises `std::bad_cast`, although the batch
  `(T, K)` array works.
- `Resample` rejects 2-D input, so a panel of K prices cannot be turned into
  K sets of OHLC bars in one call.
- `CombineLatest` accepts only one-column streams (since the 2026-10-10 fix it
  raises instead of silently keeping the first column), so an OHLC bar stream
  cannot be joined with a signal without splitting it with `Select`.
- A `Pipeline` input of shape `(T, K)` fails inside `RollingMean` ("frame width
  does not match op n_in"), and row operators refuse to run in a `Pipeline`.

## The contract

Every array has shape `(T, *A, F?)`:

- `T`, **time**, is always axis 0.
- `A`, **assets**: zero or more axes of independent series. Each asset has its
  own operator state. Only row operators read across assets.
- `F`, **fields**: an optional last axis holding the parts of one record: the
  inputs packed into one array, the outputs of a multi-output operator, or a
  record such as an OHLC bar.

An **event** is one time step, of shape `(*A, F?)`. A batch array is events
stacked along `T`.

### Rules

1. **Container independence.** A batch array, an iterator of events, and a live
   session give the same results. An iterator of `(K,)` events equals the
   `(T, K)` batch. A direct call with a 1-D array remains a time series, as
   today.
2. **Broadcasting.** A multi-input operator takes N inputs. Each input is one
   field of shape `(T, *A_i)`; a single packed array is first split along its
   last axis into the N inputs. The asset shapes `A_i` broadcast with NumPy
   rules and `T` must match. The output is `(T, *A)` for one output and
   `(T, *A, M)` for M outputs. Fields never enter the broadcast, so a field
   axis can never be aligned with an asset axis.
3. **A size-1 field axis is consumed.** An operator reads a trailing field axis
   of size 1 as its single input field: `(T, K, 1)` behaves exactly like
   `(T, K)` (`RollingMean` returns `(T, K)`, `BollingerBands` `(T, K, 3)`).
   This generalizes rule 2's packed-array split to N = 1.
4. **Time operations work per column and append fields.** `Resample`, `Delay`,
   `Dropna` and `Filter` treat every column alike. A multi-field aggregation
   appends a field axis: `Resample(agg="ohlc")` turns `(T, X)` into
   `(T', X, 4)`. `agg="ohlc_bars"` remains the separate aggregation for
   re-aggregating existing bars.
5. **Combining streams joins fields.** `CombineLatest` concatenates its inputs
   along `F`: a `(T, 4)` bar stream and a `(T,)` signal give `(T, 5)`. Inputs
   with asset axes broadcast those axes (rule 2) and concatenate their fields.
6. **Only row operators cross assets.**

### Defaults for unlabeled 2-D data

A bare `(T, X)` array does not say whether X counts assets or fields. The
reading matters in only two places:

| Where | Fields reading | Assets reading |
|---|---|---|
| `CombineLatest` of a 2-D and a 1-D stream | concatenate: `(T, X + 1)` | broadcast: `(T, X, 2)` |
| `Pipeline` inputs (phase 2) | one event of X fields | X assets, X operator states |

Everywhere else both readings give the same numbers: one-input operators treat
columns independently, broadcasting sees only separate arguments, packed
arrays are fields by necessity, multi-output results append fields, row
operators read assets by definition, and `Resample` aggregates per column.

The defaults keep today's behavior: compute operators treat `(T, X)` as
independent columns, and stream operators treat `(T, X)` as fields. The user
states another reading with `layout`.

### `layout`: naming the axes

A layout string names the axes of the data in their current order with the
letters `T` (time), `A` (asset), and `F` (field). An optional `->` gives the
wanted result:

```python
reshape(x, "TA")         # (T, K)    -> (T, K, 1)   canonical (T, *A, F)
reshape(x, "TAF->TA")    # (T, K, 1) -> (T, K)      drop the size-1 field axis
reshape(x, "TF->TAF")    # (T, 4)    -> (T, 1, 4)   insert a size-1 asset axis
reshape(x, "AT->TA")     # (K, T)    -> (T, K)      reorder
reshape(x, "TFA")        # (T, 4, K) -> (T, K, 4)   fields before assets, reordered
```

- The left side names every axis of the input: `len(left) == x.ndim`, exactly
  one `T`, at most one `F`, any number of `A` (for example assets by venue,
  `"TAAF"`).
- A letter missing on the left is inserted with size 1. A letter missing on
  the right is dropped and must have size 1. Shared letters are reordered.
- Without `->`, the result is the canonical `(T, *A, F)`.
- Violations raise `ValueError` naming the layout and the input shape.

The same argument, with the same meaning, appears wherever the reading
matters:

```python
CombineLatest(layout=("TA", "T"))(panel, market)  # market broadcast to every asset
CombineLatest()(bars, signal)                     # default fields reading: (T, 5)
from_pandas(df_of_tickers, layout="TA")           # (values (T, K, 1), index)
Input("prices", layout="TA")                      # phase 2
```

In JavaScript: `reshape(x, "TA")`, `CombineLatest({ layout: ["TA", "T"] })`,
`fromColumns(..., { layout })` where a converter exists. JS error messages and
docs use "assets" instead of "groups".

### Containers

`reshape` and `layout` accept every container screamer accepts and return the
same kind:

| Input | Result |
|---|---|
| NumPy array | a view; a copy only when axes are reordered |
| `(values, index)` tuple | `(reshaped values, index)` |
| pandas DataFrame | flat columns with `"TA"` become MultiIndex columns `(column, "value")`; `from_pandas` reads `(asset, field)` MultiIndex columns as `(T, A, F)` |
| list of events, iterator of events | the same, each event reshaped (the `T` letter refers to the sequence) |
| JS `NdArray` | the same buffer with a new shape (a copy when reordering) |
| `Pipeline` `Node` (phase 2) | a node with the declared output shape |

## Phase 1 components

Phase 1 makes rules 1 to 5 hold for batch calls, iterators of events, and the
eager stream functions. `Pipeline` internals keep their single-asset frames.

1. **Shape module (C++).** One header, `include/screamer/common/shape.h`, holds
   layout parsing and validation, the broadcast of asset shapes, the
   size-1-field consumption, and the strided iteration used to feed broadcast
   columns without copying. Python (`include/screamer/common/dispatch.h`) and
   the WASM runtime (`wasm/embind_runtime.h`) both call it, so the rules exist
   once.
2. **Multi-input broadcasting.** `read_n_arrays` and the `handle_input_Ni_*`
   paths in `dispatch.h` replace the equal-shape check with the broadcast:
   each output column reads its inputs through broadcast strides, so a `(T,)`
   price feeding K targets is never materialized K times. The JS
   multi-input path (`js/src/runtime.ts`, which today accepts only 1-D inputs
   for multi-input operators) gains the same N-D and broadcast support through
   the C++ helper.
3. **Iterators of events.** The lazy path (`LazyEvalIterator`) accepts events
   of any fixed shape: the first event fixes the event shape, every later
   event must match, and each asset keeps its own state, as in the batch path.
4. **Size-1 field consumption** in the single-input and multi-output batch
   paths (rule 3). This changes the result for explicit `(T, K, 1)` inputs
   from `(T, K, 1)` and `(T, K, 1, M)` to `(T, K)` and `(T, K, M)`; recorded
   as a breaking change in the changelog.
5. **`Resample`** accepts `(T, *A)` and `(T, *A, F)` values and aggregates each
   column; multi-field aggregations append the field axis (rule 4). Eager and
   lazy paths; the graph node keeps its current single-asset behavior until
   phase 2.
6. **`CombineLatest`** accepts multi-column inputs, concatenates fields, and
   takes `layout` (rule 5). The eager path in `screamer/streams.py` and its
   C++ core (`include/screamer/streams/combine_latest.h`); the graph node
   (`include/screamer/dag/combine_latest_node.h`) concatenates frame widths
   instead of requiring width 1, which also restores `CombineLatest` of an
   OHLC node and a signal inside a `Pipeline`.
7. **`reshape`** in Python and JS for every container in the table, and
   `layout=` on `from_pandas`.
8. **Remove the duplicated broadcasting** in `screamer/validation.py` and
   `js/src/validation.ts`; the engines broadcast themselves.
9. **Contract tests.** One parametrized test module runs every operator family
   through `(T,)`, `(T, K)`, `(T, K, 1)` and, where meaningful, `(T, K, F)`;
   through iterators of events; through the broadcasting cases (`(T,)` with
   `(T, K)`, `(T, 1)` with `(T, K)`, incompatible shapes raising); and through
   the `reshape` and `layout` cases for every container. New operators are
   covered automatically, like the existing contract-compliance meta-tests.
   The JS suite runs the same cases against Python golden values.
10. **Docs.** `docs/polymorphic_api.md` states the contract and rules;
    `docs/multistream.md` the stream defaults and `layout`; a `reshape`
    reference page; the recipes and notebooks drop their manual broadcasting
    and `Select` splitting where the new behavior covers it.

## Phase 2 (designed now, built later)

Graph frames carry an event shape `(assets, fields)` instead of a bare width.
`Input(name, layout=...)` declares it, and every node derives its output shape
from its inputs by the rules above:

- a compute node on a K-asset stream keeps K operator states;
- `CombineLatest` broadcasts assets and concatenates fields;
- `Resample` aggregates per asset;
- row operators run inside a `Pipeline`, reading across assets;
- live sessions push events of the declared shape.

Phase 1 must not preclude this: the shape module is the single place both
phases use, and the graph node changes in phase 1 (the `CombineLatest` field
concatenation) already follow the phase 2 rules for single-asset frames.

## Compatibility

- Unchanged: every call that works today on `(T,)` and `(T, K)` data, the
  defaults for unlabeled 2-D data, and 1-D direct calls as time series.
- Newly accepted: mixed-shape multi-input calls, iterators of row events,
  multi-column `Resample` and `CombineLatest`, `layout`, `reshape`.
- Changed: explicit `(T, K, 1)` inputs lose the trailing size-1 axis in the
  output (rule 3).

## Out of scope

- Labeled array types (no screamer-specific container; labels come from
  `layout` and, in phase 2, from the graph).
- Cross-asset computation in ordinary operators (only row operators).
- Phase 2 implementation.
