# Changelog

All notable changes to this project are documented in this file.

Unreleased
----------

### Changed

* **Build:** the C++ extension is now bound with
  [nanobind](https://github.com/wjakob/nanobind) instead of pybind11. nanobind
  is fetched by CMake at configure time, so it is no longer a pip build
  dependency. Every operator's observable behavior is unchanged; the full test
  suite is the oracle for that.

* **Packaging:** releases now ship a single stable-ABI (`cp312-abi3`) wheel per
  platform rather than one wheel per Python version. The wheel is built on
  CPython 3.12 and installs on 3.12 and every later 3.x.

* `EwMean`'s array path holds its accumulators in locals for the duration of
  the loop, 4.0 to 1.0 ns/sample on 1M samples. It was the one operator where
  an alternative was meaningfully ahead in the batch comparison, and it now
  leads its row: TA-Lib's `EMA` is 1.74 ns and pandas 4.20. Streaming and every
  result are unchanged.

  Same cause as `RollingMean`: the accumulators live behind `this`, and the
  compiler cannot prove the caller's output buffer does not alias the operator,
  so in a batch loop it reloads them every sample.

* `RollingMean`'s array path hoists the recurrence state into locals for the
  duration of the loop, 3.00 to 0.87 ns/sample on 1M samples. Streaming and
  every result are unchanged.

  The per-sample path keeps the running sum, buffer index and count behind
  `this`. In a batch loop the compiler cannot prove the caller's output buffer
  does not alias the operator, so it reloads all of it every iteration. Loading
  it once, running the loop, and writing it back is worth 3.4x here, and it
  takes the operator past an equivalent hand-written numba loop, which measures
  1.42 ns/sample.

  The same shape applies to every stateful operator's array path, so this is
  one instance of a general opportunity rather than a one-off.

* `WilliamsR` computes its array result by block decomposition, 19.9 to 3.4
  ns/sample on 1M samples, which puts it just ahead of TA-Lib's `WILLR` at 3.8.
  Streaming is unchanged, and the array result is asserted identical to the
  per-sample one.

  `FunctorBase` gains an optional `process_columns` hook for this, the
  multi-input counterpart of the `process_array_no_stride` override that
  `ScreamerBase` already had. It defaults to declining, so every other
  operator is untouched, and an override is expected to decline rather than
  change a result when it cannot apply: here, non-contiguous input or a `NaN`
  in the bars, which under `ignore` skips a bar and breaks the fixed block
  structure.

* `RollingMax`, `RollingMin`, `RollingRange`, `RollingArgmax` and
  `RollingArgmin` compute their array results by
  block decomposition instead of by replaying the streaming recurrence, which
  is 5 to 7 times faster and no longer depends on the shape of the data:
  `RollingMax(50)` on 1M random samples goes from 8.6 to 1.2 ns/sample, and
  from 10.6 to 1.2 on a random walk; `RollingArgmax` goes from 8.9 to 2.1.
  Streaming is untouched.

  The argmin/argmax form carries indices through the same decomposition and
  has to break ties the way the deque does, newest index first, or the two
  paths disagree on flat stretches. That is asserted against the scalar path
  on integer input, where ties are frequent.

  The deque's real cost on random input is not its comparisons but its pop
  loop, whose trip count depends on the data, so the loop-exit branch
  mispredicts on most samples. Splitting the array into blocks of `window` and
  taking a running extremum forward and backward through each gives the same
  answers with two comparisons per element and no data-dependent branch. It
  needs up to `window` samples of lookahead, so it can only be the array path;
  the event path keeps the deque. The array path falls back to the streaming
  recurrence when the operator already holds state or the input contains
  `NaN`, so results are identical either way.

* The sliding-window extremum used by `RollingMin`, `RollingMax`,
  `RollingRange`, `RollingArgmin`, `RollingArgmax`, `WilliamsR`, `Stoch`,
  `StochRSI` and `DonchianChannels` now stores its candidates in a fixed ring
  buffer instead of a `std::deque`. Nothing is allocated after construction,
  which matters more for streaming latency than for throughput: it is about 12%
  faster on the array path (`RollingMax(50)`, 9.9 to 8.7 ns/sample) and removes
  the allocator from the per-event path. Results are unchanged.

  This came out of a comparison against TA-Lib, where the family looked 6 to 7
  times slower. Most of that gap is not fixable and should not be fixed: the
  monotonic deque is O(1) per sample in the worst case, while the usual
  alternative tracks the extremum and rescans the window when it expires, which
  is quicker on typical data and degrades to O(window) on a monotone run. On a
  monotonically falling series, `RollingMax(50)` costs 3.0 ns/sample against
  TA-Lib's 16.2. A falling series is a sell-off, not a contrived input, so the
  bounded worst case is the right trade for a streaming library.
  `tests/test_extremum_worst_case.py` pins that property.

### Removed

* **Breaking:** dropped support for Python 3.11. `requires-python` is now
  `>=3.12`. The stable-ABI wheel needs nanobind's `STABLE_ABI`, which is
  available from CPython 3.12 onward, so 3.11 users must stay on an earlier
  screamer release or upgrade their interpreter.

### Added

* References for `RollingCorr`, `RollingCov`, `RollingBeta`, `RollingIqr`,
  `HullMA` and `RollingVWAP`, taking baseline coverage to 174 of 231. The pair
  statistics come from `pandas.rolling().corr()` / `.cov()` / `.var()`, which
  are external implementations rather than transcriptions.

### Fixed

* `ROC`, `ROCP` and `ROCR` defaulted to `window_size=1` while their pages
  documented `10`, TA-Lib's `timeperiod` default and the value the sibling
  `Momentum` uses. `ROC()` computed a 1-period rate of change and the page said
  otherwise. The bindings now default to 10. **Passing `window_size` explicitly
  is unaffected.** `tests/test_documented_defaults.py` now asserts every
  documented default against the pybind11 signature, so a page and its binding
  cannot drift apart again.

* `EwSkew` and `EwKurt` returned a value scaled down by roughly the effective
  sample size, so the result shrank as the window lengthened. Both fed
  *mean*-based moment ratios into bias corrections written for *sum*-based
  quantities. On exponential data (true skew 2.0) `EwSkew(span=100)` returned
  0.019; `EwKurt` returned about 0 for every distribution, which looked correct
  only on normal data, where the true excess kurtosis is 0. Both now use the
  adjusted Fisher-Pearson estimators, the same ones `scipy.stats` returns for
  `bias=False`:

      G1 = sqrt(n(n-1))/(n-2) * m3 / m2^(3/2)
      G2 = (n-1)/((n-2)(n-3)) * ((n+1) * m4/m2^2 - 3(n-1))

  **Output changes for both operators, by roughly a factor of `n_eff`.** Any
  stored value or threshold calibrated against the old output needs recomputing.

  Their reference implementations were wrong too, in a different way, which is
  why the comparison had been disabled with a `# todo baselines` note rather
  than investigated. Both baselines were rewritten from the estimators above,
  the comparison is enabled, and `tests/test_ew_moments.py` anchors them: the
  formulas reduce to `scipy.stats.skew/kurtosis(bias=False)` under equal
  weights, they recover the known moments of four distributions, and the
  estimate must not drift with window length on stationary input.

* A baseline whose optional dependency is not installed (TA-Lib needs a C
  library, so it sits in an opt-in extras group) raised out of
  `devtools/baselines/__init__.py` and took down collection for the entire test
  suite. Such a baseline is now skipped and recorded in
  `baselines.missing_dependencies()`. Only `ImportError` is swallowed; a
  baseline that fails for any other reason still raises.

* `FillNa` on a strided array tested the *output* buffer for `NaN` instead of the
  input, so it filled arbitrary elements with the fill value and passed real `NaN`
  inputs through unchanged. Reachable from the public API on any strided input;
  the contiguous path was correct. Found by giving the parity harness coverage of
  every operator (below).

* Twelve operators declared `nan_policy: ignore` but let a `NaN` input reach
  their state, so a single `NaN` cost more than the one output slot the policy
  promises. Two mechanisms: the `NaN` was stored into a carried previous-value
  slot that also served as the warmup sentinel (`TrueRange`, `ATR`, `TRIX`,
  `RollingYangZhangVar`/`Vol`), or it occupied a position in a fixed-size window
  and advanced warmup (`RollingIqr`, `RollingMaxDrawdown`, `RollingHurst`,
  `RollingTSF`, `RollingHitRate`, `Stoch`, `WilliamsR`). `RollingVWAP` and
  `KeltnerChannels` had a third form: they consumed one input into their own
  state while delegating the rest to a sub-operator, so a bar with one missing
  field left the two halves permanently out of step. `NATR` inherited the `ATR`
  defect and is fixed with it.

  **Output changes on input containing `NaN`.** A series with gaps now produces
  the documented result, which differs from what these operators returned
  before. Output on gap-free input is unchanged.

  Two compliance properties were added to `tests/test_nan_input_compliance.py`
  to catch this class: an ignored `NaN` must cost exactly one output slot, and
  for a multi-input operator one missing field must skip the whole row. The
  existing properties only checked that state recovered eventually and that the
  `NaN` index itself was `NaN`, which every one of these operators satisfied.

### Added

* 59 reference implementations, taking baseline coverage from 62 of 213
  operators to 121. Elementwise math (`Acos`, `Asin`, `Atan`, `Ceil`, `Cos`,
  `Cube`, `Floor`, `Identity`, `IsFinite`, `IsNan`, `Not`, `Round`, `Sin`,
  `Square`), cumulative and positional (`CumSum`, `CumProd`, `CumMax`,
  `CumMin`, `Diff2`, `Momentum`, `ROC`, `ROCP`, `ROCR`, `Detrend`, `Drawdown`,
  `MaxDrawdown`), and windowed (`RollingRange`, `RollingMad`,
  `RollingMedianAD`, `RollingHitRate`, `WMA`, `TRIMA`, `DEMA`, `TEMA`). Each is
  written from the documented definition rather than from screamer's
  implementation, and each is compared on every run. Writing them is what
  exposed the `ROC` default above.

* TA-Lib references for the technical-indicator family, taking baseline
  coverage to 158 of 213: `ATR`, `NATR`, `TrueRange`, `CCI`, `WilliamsR`,
  `ADX`, `Stoch`, `StochRSI`, `UltimateOscillator`, `BOP`, `AD`, `ADOSC`,
  `MFI`, `OBV`, `RollingRSI`, `KAMA`, `TRIX`, `MACD`, `RollingArgmax` and
  `RollingArgmin`. Several of these pages already cite TA-Lib as the definition
  they implement, so it is the right oracle rather than a second opinion.
  Fourteen of the twenty matched on the first run.

  `TA-Lib` is now a test dependency. Its wheels have bundled the C library
  since 0.6, so it installs in CI like any other package.

* `tests/test_baseline_defaults.py`: a reference's default arguments must match
  the operator's documented ones. Three of this batch's apparent failures were
  a reference defaulting to a different window than the operator (`KAMA` at 30
  against 10, `CCI` at 20 against 14, `TRIX` at span 15 against 14), which
  reads as an operator bug. Together with `tests/test_documented_defaults.py`
  this pins operator, page and reference to one value.

* References for the exponentially weighted volatility twins and the EW pair
  statistics, taking baseline coverage to 138 of 213. `EwParkinson*` and
  `EwGarmanKlass*` reuse the QuantLib per-bar kernel with EW weighting in place
  of rolling; `EwRogersSatchell*` is transcribed and anchored by the simulation
  test, which now covers the EW forms including the drift-robustness split.
  `EwCov`, `EwCorr` and `EwBeta` are checked against `pandas.ewm().cov()`,
  `.corr()` and `.var()`.

* References for the eight range-based volatility operators, taking baseline
  coverage to 129 of 213. `RollingParkinson*` and `RollingGarmanKlass*` are
  checked against **QuantLib** (`ParkinsonSigma` and `GarmanKlassSigma5`), an
  independent implementation of the formula rather than a transcription of it.
  `RollingRogersSatchell*` and `RollingYangZhang*` are not in QuantLib and are
  transcribed from their documented formulas.

  Because a transcription cannot catch a wrong formula,
  `tests/test_ohlc_volatility.py` anchors all four families against ground
  truth: a geometric Brownian motion is simulated at fine intra-bar resolution,
  aggregated into OHLC bars, and each estimator must recover the sigma that
  generated it, in the regime its page claims. The negative claims are asserted
  too. Under a drift of five times sigma, Rogers-Satchell stays on target while
  Parkinson reads about 3x and Garman-Klass about 2x, and with overnight gaps
  only Yang-Zhang reports the gap-inclusive total.

  `QuantLib` is now a test dependency. It is a plain wheel with no system
  library behind it, so unlike TA-Lib it installs in CI and these comparisons
  run on every push.

* The Garman-Klass pages now say *which* estimator from the paper is
  implemented (the simplified sigma_5 form), why the higher-order sigma_4 and
  the gap-aware sigma_1 / sigma_3 / sigma_6 variants are not shipped, and that
  Yang-Zhang supersedes the gap-aware ones. Identifying the variant previously
  meant testing all five against the formula.

* The baseline harness now drives multi-input operators, one independent array
  per input, which is what let the two- and three-input math operators be
  compared at all: `Add`, `Sub`, `Mul`, `Div`, `Hypot`, `Atan2`, `Linear2`,
  `Cart2Polar`, `Polar2Cart`, `Where`, and the comparison and logic family.
  Independence matters, since feeding one array to both sides of `Sub` compares
  all zeros. The `Butter*` family, `MovingAverage`, `SchmittTrigger`, `Hold` and
  `KalmanFilter` gained references in the same pass.

* `array_type: "discrete"`, small integers with repeats and real zeros, for the
  logic, equality and select operators. On continuous input two independent
  draws are never equal and are always both nonzero, so `Equal`, `And`, `Or`
  and `Where` were each compared on one branch only and asserted nothing.

* `array_type: "unit"` in the test harness, giving the inverse trigonometric
  operators input inside their domain. Driven with the previous `positive`
  input (uniform 0.1 to 10) most samples fell outside the domain and both sides
  returned `NaN`, so the comparison passed without asserting anything.


* `FracDiff(d, window_size, threshold, start_policy)` - fractional differentiation
  (Lopez de Prado, *Advances in Financial Machine Learning*, chapter 5). An FIR
  filter with taps `w_k = (-1)^k * binom(d, k)`, truncated at `window_size` taps or
  at the first weight below `threshold`. `d=0` is the identity, `d=1` reproduces
  `Diff(1)`, `d=2` reproduces `Diff2()`; fractional orders in between remove the
  trend while keeping memory an integer difference discards. Contributed by
  Mohammadjavad Vakili.

* `Resample(..., clock=True, agg='last', fill='carry', max_age=N)` bounds the
  age of a carried scalar value in a target-clock as-of join. Once its age
  exceeds `N`, the output is `NaN`; omitting `max_age` retains unbounded carry.
  Contributed by Édouard Tallent (@tagomatech).

1.1.1 - 2026-07-29
------------

### Fixed

* Value-initialized several latent uninitialized class members flagged by
  clang-tidy's member-init check (the resample plan entry, `BacktestReport`,
  `BacktestL1TradesOrders`, the L1 fill and PnL-account helpers, `TickRuleSign`,
  `LeeReadySign`, the stream `Event`, and the DAG node spec). These were
  uninitialized-read hazards of the class that can surface as platform-specific
  misbehavior; each fix is behavior-preserving (the values are set before use).

1.1.0 - 2026-07-29
------------

### Added

* `Hold(n, release=0.0)` - a time-based latch. On a nonzero input it holds that value
  for `n` steps, then returns `release`. Complements `SchmittTrigger` (level hysteresis)
  with time hysteresis.
* `Resample` information bars: `Resample(value, driver, threshold=T, agg=...)` closes a
  bar when a cumulative `driver` reaches `threshold`, giving volume bars, dollar bars,
  and other event-clock bars from one mechanism.
* `Resample` bar re-aggregation aggs `ohlc_bars` and `ohlcv_bars`: downsample
  already-built `[O,H,L,C]` / `[O,H,L,C,V,...]` bars into coarser bars (first/max/min/last
  on OHLC, sum on trailing columns) in a single C++ pass.
* `Resample` target-clock mode: `Resample(value, clock, clock=True, agg='last',
  fill='carry')` samples the last-known value as of each event of a separate clock stream
  (an as-of / `merge_asof` / reindex-forward-fill).

### Changed

* The `ohlcv` and `ohlcv2` resample aggregations now run in every regime (eager, graph,
  and lazy), not just eager. Batch results are unchanged; previously-rejected graph and
  lazy calls now succeed.
* `forecast_pairs` now runs in every regime (eager, graph, and lazy) for both `count=` and
  `duration=` modes, so a training set can be built by streaming a live feature engine.
  The public API and return values are unchanged.

1.0.1 - 2026-07-26
------------

### Changed

* screamer now installs with no runtime dependencies. `pybind11` is a build-time-only,
  header-only dependency and has been removed from the runtime requirements; it stays in
  the build-system requirements. Existing installs are unaffected.

1.0.0 - 2026-07-26
------------

First stable release. The public API is stable from this point and follows
semantic versioning.

### Changed (breaking)

* `forecast_pairs` now returns `(X_shifted, y)` instead of `(X_shifted, y, as_of)`.
  The third return (the row index) was redundant: the function delays `X` and passes
  `y` through, so the output rows sit on the caller's own timeline. Keep your own index
  alongside `X` and `y` if you need to map rows back to time.

### Added

* `Pipeline` accepts `max_pending` (default 1,000,000), the cap on the `CombineLatest`
  reorder buffer. A stream that stalls long enough to exceed the cap raises a
  `RuntimeError` instead of buffering without bound.

### Fixed

* A `Delay` feeding a `CombineLatest` merge inside a single `Pipeline` produced
  lookahead: the merge aligned each delayed frame against inputs that had not yet
  advanced, so the result was wrong in both batch and live mode. The DAG now carries an
  event-time watermark (a monotone lower bound on future frame indices) that gates a
  bounded reorder buffer at every fan-in, releasing frames in global index order once
  every input has advanced past them. `Filter` and `Dropna` forward the watermark past
  dropped frames, and `Resample` closes its windows on it, in both index and count modes.

### Documentation

* Full pass over the notebooks, guides, and every function reference page: corrected
  formulas and claims that did not match the implementation, and rewrote descriptions to
  lead with what each operator computes. Added a documentation writing guide to
  `CONTRIBUTING.md`.

0.13.0 - 2026-07-24
------------

### Changed (breaking)

* `EwKyleLambda` now accepts only the standard EW mutex (`com`, `span`, `halflife`,
  `alpha`), exactly one required. The previous positional `window` parameter is removed.
* `Propagator` parameter renamed from `window` to `window_size` for consistency with
  other windowed operators. Passing `window=` raises `TypeError`.
* `BayesianRegression` output columns reordered to `[slope, intercept, pred_mean, pred_std]`
  (was `[pred_mean, pred_std, slope, intercept]`), so slope and intercept sit at columns
  0 and 1 as in `RollingLinearRegression`.
* `Hampel`, `ImpulseClip`, `RollingSigmaClip`, `RollingOU`: the `output` parameter now
  accepts a string mode name instead of an integer. Passing an integer raises an error.
  Valid values: `Hampel`/`ImpulseClip` use `"cleaned"`, `"flag"`, `"nan"`;
  `RollingSigmaClip` uses `"clipped"`, `"mean"`, `"std"`, `"nan"`;
  `RollingOU` uses `"mrr"`, `"mean"`, `"relmean"`, `"std"`.

### Added

* `RollingSigmaClip` now exposes `start_policy` in its binding (previously omitted),
  defaulting to `"strict"` to match the other rolling operators.

### Fixed

* `EwMean`, `EwStd`, `BayesianRegression` and all other `Ew*` operators now raise
  `ValueError` when `alpha` is non-finite (`NaN` or `inf`).
* Removed contradictory "NaN values should be preprocessed" sentences from 16 docs pages
  that also carried a generated `Policy: ignore` footnote. The footnote is the authoritative
  statement; the body sentences were stale.
* Fixed incorrect comment in `bindings/bindings_misc.cpp` that said NaN "propagates" for
  `CumSum`/`CumProd`/`CumMax`/`CumMin`; they follow the standard `ignore` policy.

0.12.0 - 2026-07-23
------------

### Added

* `BayesianRegression`: online Bayesian univariate regression that emits a causal
  one-step-ahead predictive mean and standard deviation plus the current slope and
  intercept. Normal-Inverse-Gamma posterior (noise learned online, Student-t
  predictive) with exponential forgetting (`com`/`span`/`halflife`/`alpha`) and a weak
  prior, so estimates and intervals are defined from the first sample.
* Example notebook 16, "Supervised forecasting with `forecast_pairs`": build a
  leak-safe training set, fit a least-squares model, check it out of sample, and
  tie the prediction into a backtest.

### Fixed

* `forecast_pairs(count=..., dropna=True)` now also drops rows whose target is NaN
  (not only feature-warmup rows), matching `duration=` mode, so it returns a clean
  training set.

### Docs

* Documentation rewritten to the NumPy and scikit-learn voice. Removed the repeated
  batch-vs-live-stream property from per-operator docs and notebook closers (it is
  stated once, where streaming is the subject), the verification cells that only
  demonstrated equivalence, and the closing summary sections.

0.11.0 - 2026-07-22
------------

### Added

* Backtesting: engines are renamed and reorganised into a
  `Backtest<DataModel><OrderDef>` grid. The data model (Price, OHLC, Trades,
  L1, L1Trades) names the market feed; the order definition (Target, Orders)
  names the strategy's output contract. Eight engines fill eight of the ten
  cells of the 5x2 matrix; `BacktestPriceOrders` and `BacktestL1TradesTarget`
  are intentionally not provided (see `choosing_a_backtest_engine`).
* **Target engines** (`BacktestPriceTarget`, `BacktestOHLCTarget`,
  `BacktestTradesTarget`, `BacktestL1Target`): receive a scalar target
  position each event, compute the delta to the current position, and take
  liquidity to reach it. The target is clamped to the static
  `[min_position, max_position]` cap before sizing the order.
  `BacktestOHLCTarget` defers execution to the next bar's open (causal;
  no manual lag needed). `BacktestTradesTarget` and `BacktestL1Target` execute
  immediately against the current event.
* **Orders engines** (`BacktestOHLCOrders`, `BacktestTradesOrders`,
  `BacktestL1Orders`, `BacktestL1TradesOrders`): receive a two-sided resting
  quote `(bid_price, bid_size, ask_price, ask_size)` each event plus the
  market data columns. Either or both sides can fill on the same event. A
  quote submitted already crossing the spread is a taker.
* New engines completing the useful grid cells:
  `BacktestOHLCTarget` and `BacktestOHLCOrders` (replacing `BacktestOHLC` /
  `BacktestOHLCMaker`), `BacktestTradesTarget` and `BacktestTradesOrders`
  (replacing `BacktestTrades` / `BacktestTradesMaker`), `BacktestL1Target` and
  `BacktestL1Orders` (replacing `BacktestL1`), `BacktestL1TradesOrders`
  (replacing `BacktestL1Trades`), and `BacktestPriceTarget` (replacing
  `BacktestSignal`).
* All engines accept `min_position` and `max_position` (default unbounded).
  Fills are capped by a three-way minimum: order size, participation limit
  (where applicable), and remaining room to the position bound.
* MARKET/NaN/inf encoding shared across all engines: a finite price is a
  resting limit (maker); `NaN` is a side-agnostic market order (taker);
  `+inf` / `screamer.MARKET` is a market buy (never-fill sell); `-inf` is a
  market sell (never-fill buy).
* Docs: `choosing_a_backtest_engine` grid overview page with the 5x2 matrix,
  order-definition interfaces, MARKET encoding table, and fill-cap rule.
* `SchmittTrigger` gains an `initial` latch seed (`0.0`, `1.0`, or `NaN`).
* `Delay(duration)` stream op: re-stamp each event's index by a time offset (the
  time-based counterpart of `Lag`). Requires an explicit index; lossless, 1:1,
  no warmup.
* `screamer.supervised.forecast_pairs(X, y, count=|duration=)`: build a forecasting
  training set by lagging features to align with a future causal target. Returns
  `(X_shifted, y, as_of)`; `count=` is event-based, `duration=` is time-based (uses
  `Delay`, needs an index). Fully causal (lags X, never leads y).

### Changed (breaking)

* `SchmittTrigger` now seeds its latch with `initial` (default `0.0`, low) instead
  of `NaN`, so a signal that starts inside the dead band reads low during warmup
  rather than `NaN`. Pass `initial=nan` to restore the previous undefined-until-first-crossing
  behavior; pass `initial=1.0` to start high.

0.10.0 - 2026-07-18
------------

### Added

* Risk statistics: `RollingDownsideDeviation` (downside semideviation, the Sortino
  denominator), `RollingOmega` (Omega ratio of gains to losses about a threshold),
  and `RollingCVaR` (historical Conditional Value-at-Risk / Expected Shortfall,
  the mean loss in the worst alpha tail; VaR is `-RollingQuantile`). Each with a
  reference page, a plotted example, and tests.
* Backtesting: a suite of five causal C++ engines named by the market data they
  consume, all sharing one accounting core (`detail::PnLAccount`) and the
  `[equity, pnl, position, cost]` output schema. Fills follow one rule set: a
  trade or quote *through* your price fills the full remaining, a fill *at* your
  price is `min(remaining, participation_ratio * available_size)`, and a
  marketable order fills fully with `tick_size` slippage. Resting orders fill at
  their limit price (maker); an order submitted already crossing is a taker.
  * `BacktestSignal` (2 inputs): marks a position signal to a price, with a
    fractional `spread` (crossing cost) and `fee`.
  * `BacktestOHLC` (6 inputs): a directional target-position strategy on OHLC
    bars, with market orders (fill at the open, crossing half the `spread`, paying
    `taker_fee`) and limit orders (`"touch"`/`"breach"` of the bar range, paying
    `maker_fee`); bars fill in full. Causal by design: the target decided on a
    bar's close is deferred and traded on the next bar, so no manual lag is needed.
  * `BacktestTrades` (4 inputs): a resting limit order against the trade tape;
    a through-print sweeps the full order, an at-print fills a `participation_ratio`
    share.
  * `BacktestL1` (8 inputs): a two-sided maker against top-of-book quotes only.
    Fills are a documented heuristic: `"breach"` (default) fills on a through,
    `"touch"` adds a participation partial once per lock episode. Inventory cap,
    `taker_fee`, and `tick_size` for marketable orders.
  * `BacktestL1Trades` (10 inputs): the preferred maker engine. Quotes mark the
    position, the trade tape drives fills (each trade on its own event row, so no
    fill-versus-cancel ambiguity), a quote cross with no explaining trade is the
    run-over fallback.
  * `BacktestReport` (4 inputs, 6 outputs): the C++ node that turns an engine's
    `[equity, pnl, position, cost]` into the running report columns (dollar
    drawdown, cumulative cost, turnover, trade count, running max drawdown, and
    running Sharpe), so pure-C++ callers get the statistics too.
  The `backtest_report` helper wraps `BacktestReport`, labeling its columns and
  reading the last row into a summary (total PnL, max drawdown, cost, turnover,
  trades, Sharpe). It returns plain dicts of numpy arrays and needs no pandas.
  Reference pages with plotted examples, tests, and two demo notebooks (a signal
  on bars, and the event-driven engines on a real tape).

[0.9.0] - 2026-07-17
--------------------

More microstructure: order-flow toxicity, book pressure, and spread
decomposition, plus a re-classified docs taxonomy.

### Added

* Microstructure tranche 2, order-flow toxicity and book features:
  * `VPIN` (Easley-Lopez de Prado-O'Hara 2012): order-flow toxicity over a
    self-contained volume clock (equal-volume buckets, boundary-splitting).
  * `MicroPrice` (Stoikov 2018, first-order): the imbalance-weighted mid, a fair
    value that leans toward the thinner side of the book.
  * `QueueImbalance`: L1 book (queue) imbalance, a documented synonym of `OFI`
    applied to resting bid/ask sizes.
* Microstructure tranche 3, order-book flow and spread decomposition:
  * `ContOFI` (Cont-Kukanov-Stoikov 2014): the canonical order-flow imbalance
    from L1 book events, distinct from the trade-flow `OFI`.
  * `EffectiveSpread` (`2*|price - mid|`) and `RealizedSpread` (the liquidity
    part kept after the price moves); their difference is the price-impact /
    adverse-selection component.
* Each ships with a reference page, a plotted usage example, and tests.
* Docs: the flat "Microstructure" function topic is split into four groups -
  Trade signing, Order-flow imbalance, Price impact & liquidity, and Order-flow
  arrivals - with "microstructure" kept as a search tag.
* Docs: a third microstructure notebook (toxicity, book pressure, and spreads)
  covering VPIN, QueueImbalance, MicroPrice, ContOFI, and the effective/realized
  spread decomposition.

[0.8.0] - 2026-07-16
--------------------

Microstructure and order-flow operators, and the lazy Pipeline path moved into
the C++ core.

### Added

* Microstructure and order-flow operators, all implemented as C++ core nodes:
  * Trade signing: `TickRuleSign`, `LeeReadySign`, `SignedVolume`, and the
    bar-level `BulkVolumeClassifier` (BVC).
  * Order flow: `OFI` (order-flow imbalance) and `RollingOrderImbalance`.
  * Price impact and liquidity: `RollingKyleLambda`, `EwKyleLambda`,
    `AmihudIlliquidity`, `RollSpread` (Roll effective spread), and the Bouchaud
    `Propagator`.
  * Event intensity: `HawkesIntensity`, a self-exciting arrival-rate model.
* Two demo notebooks driven by a small committed real-data slice (six hours of
  Deribit BTC- and ETH-perpetual trades): order flow and trade signing, and
  price impact and liquidity.

### Internal

* The lazy (Python-iterator) `Pipeline` path now runs in the C++ core. A C++
  driver merges the input feeds, drives the compiled graph, and runs the
  multi-output watermark as-of join, replacing the previous Python driver. The
  `batch == lazy == graph` invariant is now enforced by one implementation
  rather than by parallel ones agreeing. The k-way merge refills lazily, so it
  never reads further ahead than requested (safe for endless streams).
* Eager `combine_latest` coalesces per index in C++ (the Python dedup is gone),
  and `split` runs in C++.

[0.7.0] - 2026-07-12
--------------------

The v2 API: one consistent call shape, streams as plain tuples, and pipelines.

### Changed (breaking)

* Stream operators are now CamelCase config-first classes, called like the
  functors as `Op(config)(data)`: `Resample`, `Dropna`, `Select`, `Filter`,
  `CombineLatest`, `Merge`. The lowercase function forms are removed from the
  public API.
* The `Stream` class is removed. A stream is a plain `(values, index)` tuple;
  `to_pandas` / `from_pandas` convert to and from pandas.
* `Dag` is renamed to `Pipeline` (`from screamer import Pipeline`).
* `Resample` takes `freq=` (a time window) or `count=` (a number of arrivals),
  and `agg=` a functor or string. The `every=`, `func=`, and `agg={dict}` forms
  are gone; compose several reducers with `CombineLatest` instead.
* `replay` and `multi_resample` are removed; compose the existing operators.

### Added

* Comparison and logic operators: `GreaterThan`, `LessThan`, `GreaterEqual`,
  `LessEqual`, `Equal`, `NotEqual`, `And`, `Or`, `Not`, `Where`, `IsNan`,
  `IsFinite`. These build the masks the new `Filter` gate consumes.
* Expanding whole-history statistics: `ExpandingMean`, `ExpandingVar`,
  `ExpandingStd`, `ExpandingSkew`, `ExpandingKurt`, `ExpandingSlope`, and the
  running `ExpandingSum` / `ExpandingMax` / `ExpandingMin` / `ExpandingProd`.
* `PosPart` (`max(x, 0)`) and `NegPart` (`max(-x, 0)`).
* OHLC bar aggregations for `Resample`: `agg="ohlc"`, `"ohlcv"`, `"ohlcv2"`.

### Internal

* All stream-operator compute moved into C++ (dropna, select, filter, and merge
  as graph nodes), so the Python bindings stay thin.

### Fixed

* `Clip` vectorizes to SIMD min/max, about 2x faster; it had regressed to 2x
  slower than `np.clip`.


[0.6.0] - 2026-07-06
--------------------

Multi-stream and pipeline infrastructure.

### Added

* The streams layer, for aligning and reshaping feeds that do not tick together:
  `CombineLatest`, `Merge`, `Dropna`, `Filter`, `Select`, `split`, and time- or
  count-based `Resample`.
* Pipelines (then named `Dag`): wire operators into a reusable graph, run it in
  batch or live with identical results, serialize it to JSON, and visualize it as
  a text tree or Graphviz.
* Despiking functors: `RollingMedianAD`, `Hampel`, `ImpulseClip`.
* A topic taxonomy and unified frontmatter across all function reference pages.


[0.5.0] - 2026-05-21
--------------------

### Added

* `SchmittTrigger`, a hysteresis comparator.

### Changed

* The `ignore` NaN policy is applied consistently across the library: a `NaN`
  input is skipped and never corrupts a function's internal state.


[0.4.0] - 2026-05-20
-------------------------

### Changed (breaking - JSON consumers)

* `screamer/data/help.json` schema: the freeform `body_markdown` field has been
  removed and replaced with two structured fields:
  - `details` (string) - markdown prose, guaranteed to contain no fenced
    code blocks. Use this when rendering the description / math / notes.
  - `examples` (list of `{language, caption, code}`) - extracted code
    examples, one entry per `### Caption` heading in the source markdown.
    `{eval-rst} .. plotly::` directives are unwrapped to plain python.
  
  Consumers that read `body_markdown` must switch to `details` (and
  optionally render `examples` separately). No backwards-compatibility
  shim is provided.

### Changed

* Function reference docs (`docs/functions_*/<Name>.md`) now follow a
  canonical layout: prose lives under H2 sub-headings (Description,
  Formula, …), examples live under a single `## Examples` H2 with one
  `### Caption` per example. The sphinx-rendered pages adopt the same
  structure.

[0.3.0] - 2026-05-11
--------------------

This release more than doubles the indicator surface (67 → 153 exposed
classes) and closes six of the seven roadmap sections. Almost every
new class is cross-validated against TA-Lib, pandas, scipy, or
pandas-ta-classic to floating-point precision; documented divergences
(EMA convention, Cutler vs Wilder RSI) are pinned by tests so future
drift fires. Full test suite: 2126 passing.

### Added

#### Dispatcher infrastructure

* **Plan E (`N→M` dispatch)** completed. `FunctorBase` now supports
  any combination of multi-input + multi-output classes (1→1, N→1,
  1→M, N→M). First consumers: `Cart2Polar` / `Polar2Cart` (2→2),
  `KeltnerChannels` (3→3), `RollingLinearRegression` (2→4).
* New shared primitives: `detail::MonotonicDeque<bool IsMax>` (six+
  classes share it), `detail::WilderSmoother`.
* Documented multi-output shape rule
  `output.shape == single_input.shape + (M,)` with column-by-column
  pairing in `docs/polymorphic_api.md`.

#### Math

* Element-wise: `Floor`, `Ceil`, `Round` (banker's rounding),
  `Square`, `Cube`, `Sin`, `Cos`, `Atan`, `Asin`, `Acos`, `Identity`,
  `Hypot` (2→1), `Atan2` (2→1).
* `Linear2(a, b, c)` - two-input affine combination
  `a*x + b*y + c`, pairs with `Sign`/`Relu`/`Sigmoid` for compact
  composed expressions.
* `Cart2Polar` / `Polar2Cart` - 2→2 coordinate conversions; first
  consumers of the N→M dispatcher.

#### Misc / data transforms

* `CumSum`, `CumProd`, `CumMax`, `CumMin` - `O(1)` running
  reductions matching numpy.
* `Diff2` - second-order finite difference (discrete second
  derivative).
* `Detrend(window)` - `x − rolling_mean(x)`.
* `Momentum(k)` - alias of `Diff(k)` (TA-Lib's `MOM`).

#### Rolling-window statistics

* `RollingMad`, `RollingIqr`, `RollingRange` - composition of
  existing primitives or sharper variants.
* `RollingArgmin`, `RollingArgmax` - window-offset of the
  rolling extremum (TA-Lib's `MININDEX` / `MAXINDEX`).
* `RollingRank`, `RollingPercentile` - pandas-style position
  metrics with average tie rule.

#### Exponentially-weighted statistics

* `EwCov`, `EwCorr`, `EwBeta` - 2-input pair statistics. Matches
  pandas `ewm(adjust=True).cov / .corr` bit-exactly. EwBeta follows
  the CAPM `(target, regressor)` convention.

#### Moving averages

* `WMA` - linearly-weighted moving average, O(1) per step via the
  identity `W[t] − W[t−1] = w·x[t] − S[t−1]`.
* `DEMA`, `TEMA` - Mulloy's double/triple EMA compositions.
* `TRIMA` - triangular MA (`SMA(SMA(x))`).
* `HullMA` - `WMA(2·WMA(n/2) − WMA(n), √n)`.
* `KAMA` - Kaufman Adaptive MA with O(1) per step; matches TA-Lib
  bit-exactly.

#### Momentum / oscillators

* `MACD` (1→3), `WilliamsR` (3→1), `Stoch` (3→2, fast and slow
  via `smooth_k`), `StochRSI` (1→2), `TRIX`, `BOP` (4→1), `CCI`
  (3→1), `UltimateOscillator` (3→1), `ADX` (3→3 returning
  `+DI` / `-DI` / `ADX`).
* `ROC`, `ROCP`, `ROCR` - TA-Lib rate-of-change family.
* `RollingRSI` default changed to **Wilder's smoothing** (matches
  TA-Lib and pandas-ta); `method="cutler"` preserves the old
  Cutler form. Earlier versions diverged from TA-Lib's `RSI` by
  ~11 RSI points -- now bit-exact.

#### Volatility / range

* Range-based volatility quartet: **Parkinson**, **Garman-Klass**,
  **Rogers-Satchell**, **Yang-Zhang**. Each ships in `Var` and
  `Vol` variants, with rolling and EW smoothing (for the first
  three) -- 14 classes total.
* `TrueRange`, `ATR(window)`, `NATR(window)` - Wilder's
  bar-aware volatility family.
* `DonchianChannels` (2→3), `KeltnerChannels` (3→3) - channel
  indicators.

#### Volume-aware

* `RollingVWAP`, `OBV`, `AD`, `ADOSC`, `MFI` - first volume-aware
  primitives in screamer; all bit-exact to TA-Lib counterparts
  (except `ADOSC`, which inherits the documented EMA-convention
  divergence).

#### Performance / risk

* `Drawdown`, `MaxDrawdown`, `RollingMaxDrawdown`,
  `RollingSharpe(window, periods_per_year)`,
  `RollingSortino(window, ppy, target)`,
  `RollingInfoRatio(window, ppy)`,
  `RollingCalmar(window, ppy)`,
  `RollingHitRate(window)` - backtest-evaluation metrics. None of
  these are in TA-Lib; they're a real differentiator for screamer
  in trading pipelines.

#### Statistical / regression

* `RollingAlpha` - companion intercept to `RollingBeta`.
* `RollingResidualStd` - std of the per-bar `RollingSpread`.
* `RollingLinearRegression` (2→4) - full OLS fit returning
  `(slope, intercept, r², stderr)`. First 2→4 consumer of the
  N→M dispatcher. `stderr` is the RMSE of residuals (standard
  error of estimate, not slope-stderr).
* `RollingTSF` - TA-Lib's Time-Series Forecast (regression vs
  time projected one step ahead), bit-exact to `talib.TSF`.
* `RollingHurst(window, min_scale=4, method='rs')` - rolling Hurst
  exponent via Anis-Lloyd corrected rescaled-range analysis at
  dyadic scales. Bit-exact to the reference Python implementation;
  ~0.5 on white noise, >0.5 on integrated processes.

#### Signal processing

* `ButterHighpass`, `ButterBandpass`, `ButterBandstop` -
  high-pass, band-pass, band-stop Butterworth IIR filters.
  Added the underlying `lp2hp_zpk` / `lp2bp_zpk` / `lp2bs_zpk`
  ZPK transformations so future Bessel/Cheby/Elliptic families
  also get all four btypes once their prototypes are written.
* `MovingAverage(taps)` - FIR with arbitrary user-supplied taps
  (pair with `np.hamming` / `np.kaiser` / `scipy.signal.firwin`).
* `KalmanFilter(process_var, observation_var)` - scalar 1-D
  Kalman, O(1) per step.

### Validation

* New `tests/test_third_party_alignment.py` runs against **TA-Lib**
  and **pandas-ta-classic** under the new optional `validation`
  install group. Tests are skipped gracefully when the libraries
  are unavailable.
* New `docs/conventions.md` documents the few deliberate
  divergences (EMA `adjust=True` vs TA-Lib's adjust=False + SMA
  seed; `ddof=1` vs ddof=0 in `RollingStd`; Wilder vs Cutler RSI).
  Each divergence is asserted in the expected direction so future
  drift in either screamer or the third-party library trips the
  test.

### Changed

* `RollingRSI` default is now Wilder smoothing (was Cutler);
  `method="cutler"` preserves the previous behaviour.
* Pre-existing pending items now part of this release: general-
  order Butterworth filter, `RollingOU`, refactored devtools.

Version v0.1.46 (2024-11-02)
-------------------------

### Added

* RollingSigmaClip
* Relu
* Elu
* Selu
* Sigmoid
* Tanh
* Softsign
* Linear
* Power

Version v0.1.35 (2024-11-01)
-------------------------

* Improved documentation

Version v0.1.34 (2024-10-31)
-------------------------

Version v0.1.33 (2024-10-31)
-------------------------

### Added

#### Data handeling

* fillna
* ffill
* clip

#### Math

* Abs
* Sign
* Exp
* Log
* Sqrt
* Erf
* Erfc

#### Simple transforms

* Diff
* Lag
* Return 
* LogReturn

#### Rolling window functions

* rolling std
* rolling skew
* rolling kurtosis
* rolling zscore
* rolling min
* rolling max
* rolling median
* rolling quantile
* rolling rms
* rolling poly1, 1rst order polynomial fit
* rolling poly2, 2nd order polynomial fit

#### Exponentiually weighted functions

* EwMean
* EwStd
* EwVar
* EwSkew
* EwKurt
* EwRms
  
#### Filters

* 2nd order Butterworth


#### Interface
* support for iterator / generator processing

### Fixed
* Fixed incorrect results when applying transforms to a view on a numpy array.

### Changed
* removed the transform member functions

Version v0.1.32 (2024-10-20)
-------------------------

### Added

* Differences, Simple Return, Log Return, Rolling Sum, Simple Moving Average

### Changed
* removed initial_value from the constructor, we (for now) return NaN values when we cant resolve indicators.

Version v0.1.31 (2024-10-20)
-------------------------

### Added
* The indicator.transform() member functions can now transform multi-dimensional numpy arrays.
* Added documentation.


