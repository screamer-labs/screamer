# Changelog

All notable changes to this project are documented in this file.

[Unreleased]
------------

### Added

* Example notebook 16, "Supervised forecasting with `forecast_pairs`": build a
  leak-safe training set, fit a least-squares model, check it out of sample, and
  tie the prediction into a backtest.

### Fixed

* `forecast_pairs(count=..., dropna=True)` now also drops rows whose target is NaN
  (not only feature-warmup rows), matching `duration=` mode, so it returns a clean
  training set.

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


