# Release 3.0.0: open issues

Goal: zero open issues before `make major`. Each item is closed by a fix with a test, or by an explicit user decision recorded here. Status: `open`, `decided: ...`, or `done (commit)`.

## A. Behavior consistency (needs a decision)

| # | Issue | Status |
|---|---|---|
| A1 | A list of rows into a row operator (`CrossRank()([r1, r2])`) returns an ndarray; rule 3 says list in, list out (`RollingMean(2)([r1, r2])` returns a list). 2.8.0 behavior. | decided: a list into a row operator returns a list of rows (rule 3); "Changed" entry; to implement |
| A2 | Overlapping async `next()` on one stream: Python raises `RuntimeError`, JS queues the calls (2.8.0 JS behavior). | decided: keep native convention per language (Python raises, JS queues); document as intentional; docs to update |
| A3 | An error during a stream: JS closes the sources and ends the stream; Python raises and keeps the stream open (sources freed on garbage collection). | decided: keep native convention per language (Python map-like, JS generator-like); document, advise closing sources (contextlib.closing); docs to update |
| A4 | Shape errors: Python raises `ValueError`; JS raises `TypeError` for a length mismatch and `RangeError` for an asset mismatch. | decided: JS raises RangeError for every shape mismatch (counterpart of Python ValueError), TypeError for wrong kinds; to implement |
| A5 | Reported: `Return` and `Diff` "lose state across chunked batch calls". Needs verification of what is meant and whether it violates a rule. | done: verified, not a bug; every batch call is an independent run (D8) for all operators; chunked continuation moved to Future features |
| A6 | A `str`, `dict` or pandas `DataFrame` passed to a multi-input operator is treated as a lazy iterator and fails only at the first `next()`, with an unclear error. | decided: pandas Series/DataFrame read as their values (result is a NumPy array); str, bytes, dict, set raise TypeError at the call; to implement |

## B. Python and JavaScript parity

| # | Issue | Status |
|---|---|---|
| B1 | Python `Pipeline.live().push` takes single values only; JS can push wide events (`pushWide`). | decided: one method in both languages; `push(input, index, value)` takes a number or a sequence of numbers (one wide event); JS `pushWide` stays as an alias; to implement |
| B2 | The five microstructure classes (`SignedVolume`, `RollingKyleLambda`, `EwKyleLambda`, `RollingOrderImbalance`, `QueueImbalance`) are missing in JS. | decided: move them into C++ as named operators reusing the existing kernels (input swap in C++); both bindings generated; Python names and 2.8.0 keyword inputs kept; compat gate checks results; to implement |
| B3 | JS lacks `backtestReport`, `forecastPairs`, `split`, and options-object constructors. | decided: (1) BacktestReport summary + column names move into C++, Python/JS label wrappers over it; (2) forecast_pairs graph built once in C++, both bindings call it; (3) expose the C++ split partition to JS; (4) generated JS constructors also accept one options object (`EwMean({span: 20})`), positional unchanged, unknown keys raise TypeError; to implement |
| B4 | JS `combineLatest` has no `layout` option (it is graph-only; assets in a `Pipeline` are phase 2). JS `combineLatest` takes Pipeline nodes only; Python's layout join (`_join_spec`, `_assemble`, `_combine_latest_zip_lazy`) is Python logic. | decided: move the layout join (asset broadcast, field concat, assembly, positional zip) into C++; Python `combine_latest` calls it with unchanged results (compat gate); JS `combineLatest` gains the direct forms (arrays/NdArray, (values, index), iterators, async) with `layout`, graph form kept; layout inside a Pipeline stays E1; to implement |

## C. Correctness bugs that predate this branch

| # | Issue | Status |
|---|---|---|
| C1 | `PortfolioReport` turnover sums units across assets, which is meaningless for weighted books. | open |
| C2 | `strategy_test` rejects `(T, K)` input. | open |
| C3 | Backtest engines declare `nan_policy` "ignore" but hold the position on a NaN target. | open |
| C4 | `Add()([v, w])` raises an error that reports shape `(1, 30)`, which is misleading. | open |
| C5 | A module-level async iterator whose source references it forms a reference cycle the garbage collector cannot see (no `tp_traverse` on the lazy iterators); nanobind reports "leaked instances" at exit. | open |

## D. Performance

| # | Issue | Status |
|---|---|---|
| D1 | A scalar call to a multi-input operator (`RollingCorr()(x, y)`) costs about 1.5 µs, against 0.07 µs for a one-input operator. It matters for live loops. | open |
| D2 | JS `evalShaped` makes one virtual `eval` per element and plans the broadcast twice; the Python batch path uses the faster column kernels. | open |
| D3 | The Python async path copies each event's items twice. | open |
| D4 | No build flag pins floating-point contraction; GCC defaults to `-ffp-contract=fast`, which could reintroduce last-bit batch/stream differences on a future platform (the contract tests would catch it). | open |

## E. Phase 2 of the shape contract (spec scope)

| # | Issue | Status |
|---|---|---|
| E1 | Inside a `Pipeline`, frames carry no asset shape: `Input(layout=...)`, compute nodes on K-asset streams, row operators inside a `Pipeline`. | open |
| E2 | `Resample(agg="ohlc")` on X columns inside a `Pipeline` gives a flat `(T', 4 * X)`, while the eager call gives `(T', X, 4)`. | open |
| E3 | `Resample` with a functor `agg` takes one column. | open |
| E4 | `CombineLatest(emit="on_any")` with a wide `Input` inside a `Pipeline` can raise when a row is due before the wide input arrives. | open |

## F. Verification and documentation

| # | Issue | Status |
|---|---|---|
| F1 | Citations to verify: Mulloy 1994, Hutson 1983, Williams 1985, Lambert 1980, Kaufman 1995, and the RoofingFilter magazine issue. | open |
| F2 | The CHANGELOG states WilliamsR packed input is "about 3.5x faster"; the figure was not re-measured. | open |
| F3 | Ehlers cycle and spectral operators still want a domain review of their numerics. | open |
| F4 | Verify the JS build in a real browser before release (standing rule). | open |

## G. Robustness details

| # | Issue | Status |
|---|---|---|
| G1 | JS `exceptionMessage` still dereferences a user-thrown number that happens to be an aligned in-heap address. | open |
| G2 | JS row-operator and reducer paths take an `NdArray`'s `shape` without checking it against `data.length`. | open |
| G3 | JS factories capture vector arguments by reference, so mutating the array later changes copies made afterwards. | open |
| G4 | JS `zipSources` calls `return()` on a source whose own `next()` threw (a `for...of` loop does not). | open |

## Future features (not in 3.0.0)

- Chunked batch processing that carries operator state from one batch call to the next (e.g. `op(chunk, continue_=True)`), for data larger than memory. Today: one stream, or `op(x)` per value. (From A5.)
- pandas in, pandas out: operators return a Series or DataFrame with the input's index and column names. Today pandas reads as its values and the result is a NumPy array; use `to_pandas`. (From A6.)
