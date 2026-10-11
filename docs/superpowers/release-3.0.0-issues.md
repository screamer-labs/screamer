# Release 3.0.0: open issues

Goal: zero open issues before `make major`. Each item is closed by a fix with a test, or by an explicit user decision recorded here. Status: `open`, `decided: ...`, or `done (commit)`.

## A. Behavior consistency (needs a decision)

| # | Issue | Status |
|---|---|---|
| A1 | A list of rows into a row operator (`CrossRank()([r1, r2])`) returns an ndarray; rule 3 says list in, list out (`RollingMean(2)([r1, r2])` returns a list). 2.8.0 behavior. | decided: a list into a row operator returns a list of rows (rule 3); "Changed" entry; to implement |
| A2 | Overlapping async `next()` on one stream: Python raises `RuntimeError`, JS queues the calls (2.8.0 JS behavior). | decided: keep native convention per language (Python raises, JS queues); document as intentional; docs to update |
| A3 | An error during a stream: JS closes the sources and ends the stream; Python raises and keeps the stream open (sources freed on garbage collection). | decided: keep native convention per language (Python map-like, JS generator-like); document, advise closing sources (contextlib.closing); docs to update |
| A4 | Shape errors: Python raises `ValueError`; JS raises `TypeError` for a length mismatch and `RangeError` for an asset mismatch. | decided: JS raises RangeError for every shape mismatch (counterpart of Python ValueError), TypeError for wrong kinds; done (a02d55f) |
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
| C1 | `PortfolioReport` turnover sums units across assets, which is meaningless for weighted books. | decided: optional price input `PortfolioReport()(out, prices)` with prices (T, K) gives turnover as traded notional sum(abs(dpos) * price), in initial-capital units like drawdown/cum_cost; without prices the 2.8.0 result is unchanged; docs say to pass prices for mixed-unit books; to implement |
| C2 | `strategy_test` rejects `(T, K)` input. | decided: accept (T, K) inputs; "bars" and "returns" (every mode: log, diff, level) apply one time permutation per copy to all assets, so cross-asset correlation is preserved and only serial structure is destroyed; "shift" uses one offset for all columns; the C++ permutation kernels take (T, K); mixed (T, K) and (T,) inputs allowed; 1-D results unchanged (compat gate); to implement |
| C3 | Backtest engines declare `nan_policy` "ignore" but hold the position on a NaN target. | decided: labeling bug, behavior is correct; audit all 12 engines, relabel those that hold on a NaN target `nan-aware`, keep `ignore` where NaN gives a NaN row with unchanged state; update nan_and_warmup.md list; add a test checking each label against behavior on NaN target and NaN price; no result change; to implement |
| C4 | `Add()([v, w])` raises an error that reports shape `(1, 30)`, which is misleading; `Add()([1.0, 2.0])` raises `std::bad_cast` (2.8.0 raised bad_cast for both). | decided: both raise ValueError naming the list-as-events reading, the offending event's value count, and the fix (`Add()(v, w)`); JS RangeError (A4); message tests in Python and JS; error-only change; to implement |
| C5 | A module-level async iterator whose source references it forms a reference cycle the garbage collector cannot see (no `tp_traverse` on the lazy iterators); nanobind reports "leaked instances" at exit. Reproduced with the sync iterator too (`src.it = RollingMean(2)(src)`). | decided: add tp_traverse/tp_clear via nb::type_slots on LazyEvalIterator and LazyAsyncIterator (source + operator copies) and audit every other binding that holds Python objects; tests: weakref after gc.collect, subprocess with no "leaked" on stderr, plus a soak test (create/drain/drop streams, pipelines and live sessions many times; memory stays flat) since the library must run for months; to implement |
| C6 | `RollingSkew` and `RollingKurt` are numerically unstable: they use raw running sums of x^2, x^3, x^4, which cancel when the mean is large next to the spread. On a price series near 1000, rounding alone (FMA on/off) moves RollingKurt by 3.4e-3 of its scale (22% at one point). Found while measuring D4. | decided: rewrite the update in a stable form (centered/shifted moments, Welford-style); audit every other moment-based op (EW and expanding skew/kurt, variance family) for the same pattern; test at a large mean level against a high-precision reference; "Fixed" changelog entry (results change because they were wrong); to implement |

## D. Performance

| # | Issue | Status |
|---|---|---|
| D1 | A scalar call to a multi-input operator (`RollingCorr()(x, y)`) costs about 1.5 µs, against 0.07 µs for a one-input operator. It matters for live loops. Same in 2.8.0 (measured); the scalar case is reached last in `functor_Ni_1o`. | decided: fast path first in functor_Ni_1o and functor_Ni_Mo: exactly N Python float/int or NumPy scalar args call the kernel directly; target < 0.15 µs; benchmark test pins it against the 1-input cost; no result change; to implement |
| D2 | JS `evalShaped` makes one virtual `eval` per element and plans the broadcast twice; the Python batch path uses the faster column kernels. Measured (1M elems): RollingMean 2.0 vs 6.7 ns, RollingCorr 5.8 vs 15.9 ns, Add 10.3 ns in JS. | decided: one batch entry on the EvalOp base (run a whole strided column), implemented once in the CRTP FunctorBase (inlined kernel loop, one virtual call per column); Python batch and WASM evalShaped both use it; plan once; JS bench gate within ~1.5x Python on these ops, no Python regression, results unchanged; to implement |
| D3 | The Python async path copies each event's items twice. Measured: async op overhead about 0.25 µs/event, mostly the per-event Python coroutine; the copies are pointer copies (tens of ns). | decided: remove the list-to-vector copy (`_step` reads the list directly); keep the Python coroutine (a C++ awaitable saves at most ~0.2 µs at a stability cost); benchmark test pins async overhead per event; to implement |
| D4 | No build flag pins floating-point contraction; GCC defaults to `-ffp-contract=fast`, which could reintroduce last-bit batch/stream differences on a future platform (the contract tests would catch it). Measured on arm64 (218 ops, 200k batch): geomean +4%, median +0.4%; outliers CumSum/ExpandingSum/ExpandingMean 4x, CumProd 2.9x, ATR/PlusDM/NATR 1.4-1.6x (codegen side effect, CumSum has no multiply-add); 57 ops change last bits (~1e-9). | decided: pin `-ffp-contract=off` (GCC/Clang, Python and WASM) for bit-identical results across platforms; before merge fix the codegen outliers (D2 column entry / local accumulators) so no op is >10% slower, enforced by a benchmark gate; "Changed" entry for last-bit differences on Apple Silicon; to implement |

## E. Phase 2 of the shape contract (spec scope)

| # | Issue | Status |
|---|---|---|
| E1 | Inside a `Pipeline`, frames carry no asset shape: `Input(layout=...)`, compute nodes on K-asset streams, row operators inside a `Pipeline`. | decided: additive (raises today); moved to Future features (phase 2) |
| E2 | `Resample(agg="ohlc")` on X columns inside a `Pipeline` gives a flat `(T', 4 * X)`, while the eager call gives `(T', X, 4)`. | decided: for 3.0.0 a multi-column `Resample(agg="ohlc")` inside a Pipeline raises ValueError naming the alternatives (eager call, or one Resample per column); single-column unchanged; phase 2 returns (T', X, 4), an error-to-result change; to implement |
| E3 | `Resample` with a functor `agg` takes one column. | decided: additive (raises today); moved to Future features (phase 2) |
| E4 | `CombineLatest(emit="on_any")` with a wide `Input` inside a `Pipeline` can raise when a row is due before the wide input arrives. | decided: additive; keep the documented error with a test; full support moved to Future features (phase 2); to implement (test) |

## F. Verification and documentation

| # | Issue | Status |
|---|---|---|
| F1 | Citations to verify: Mulloy 1994, Hutson 1983, Williams 1985, Lambert 1980, Kaufman 1995, and the RoofingFilter magazine issue. | decided: verify each reference against its source (authors, title, venue, year, and that it supports the claim); correct the pages; to do |
| F2 | The CHANGELOG states WilliamsR packed input is "about 3.5x faster"; the figure was not re-measured. | decided: re-measure; state the measured figure in the CHANGELOG or drop the claim; to do |
| F3 | Ehlers cycle and spectral operators still want a domain review of their numerics. | decided: review pack per operator (Ehlers formula, our code, plots on synthetic signals with known period/phase: sine, chirp, trend + cycle) plus automated checks that period/phase estimates recover the known values within a tolerance; user reviews failures and odd plots; to do |
| F4 | Verify the JS build in a real browser before release (standing rule). | decided: headless-Chrome check on the final JS build as the last pre-release step; to do |

## G. Robustness details

| # | Issue | Status |
|---|---|---|
| G1 | JS `exceptionMessage` still dereferences a user-thrown number that happens to be an aligned in-heap address. | decided: native WebAssembly exceptions (-fwasm-exceptions) + Emscripten getExceptionMessage, delete the address heuristic; narrow the try blocks so normalizeError wraps only calls into WASM and user errors pass through unchanged; minimums Chrome 95, Firefox 100, Safari 15.2, Node 17 (document in README/javascript.md); confirm size/speed with the throughput bench; tests for user-thrown number/string/Error and C++ RangeError; done (bac33b1) |
| G2 | JS row-operator and reducer paths take an `NdArray`'s `shape` without checking it against `data.length`. | decided: one asNd() entry that runs checkNd for every NdArray on every path (nothing reads .shape before it); data/shape mismatch throws RangeError (A4), malformed shape TypeError; test feeds a malformed NdArray to every op kind (elementwise, multi-input, row op, reducer, row stream, live push) and asserts it raises before any C++ call; done (a02d55f) |
| G3 | JS factories capture vector arguments by reference, so mutating the array later changes copies made afterwards. | decided: the generator copies every array parameter once at construction (Array.from) and the factory closure captures the copy, matching Python; test mutates each kind of array parameter (taps, per-asset vectors) after construction and checks results are unchanged; done (2b1ea33) |
| G4 | JS `zipSources` calls `return()` on a source whose own `next()` threw (a `for...of` loop does not). | decided: record which source threw and skip it in the cleanup, as for an ended source (JS iterator-close semantics), in zipSources and zipAsyncSources; test (sync and async): error reaches caller unchanged, thrower's return() never called, the other source's return() called once; done (2f716f2) |

## Future features (not in 3.0.0)

- Phase 2 of the shape contract (E1, E3, E4, and E2's (T', X, 4) in a Pipeline): asset frames inside a `Pipeline`, `Input(layout=)`, functor `agg` over several columns, `on_any` with wide inputs. All raise today, so they can land in 3.x. Note (user, 2026-10-11): 3.0.0 may wait until phase 2 and phase 3 are done as well; if so these move back into scope.

- Chunked batch processing that carries operator state from one batch call to the next (e.g. `op(chunk, continue_=True)`), for data larger than memory. Today: one stream, or `op(x)` per value. (From A5.)
- pandas in, pandas out: operators return a Series or DataFrame with the input's index and column names. Today pandas reads as its values and the result is a NumPy array; use `to_pandas`. (From A6.)
