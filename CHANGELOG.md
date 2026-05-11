# Changelog

All notable changes to this project are documented in this file.
 
[Unreleased] - yyyy-mm-dd
-------------------------

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
* `Linear2(a, b, c)` — two-input affine combination
  `a*x + b*y + c`, pairs with `Sign`/`Relu`/`Sigmoid` for compact
  composed expressions.
* `Cart2Polar` / `Polar2Cart` — 2→2 coordinate conversions; first
  consumers of the N→M dispatcher.

#### Misc / data transforms

* `CumSum`, `CumProd`, `CumMax`, `CumMin` — `O(1)` running
  reductions matching numpy.
* `Diff2` — second-order finite difference (discrete second
  derivative).
* `Detrend(window)` — `x − rolling_mean(x)`.
* `Momentum(k)` — alias of `Diff(k)` (TA-Lib's `MOM`).

#### Rolling-window statistics

* `RollingMad`, `RollingIqr`, `RollingRange` — composition of
  existing primitives or sharper variants.
* `RollingArgmin`, `RollingArgmax` — window-offset of the
  rolling extremum (TA-Lib's `MININDEX` / `MAXINDEX`).
* `RollingRank`, `RollingPercentile` — pandas-style position
  metrics with average tie rule.

#### Exponentially-weighted statistics

* `EwCov`, `EwCorr`, `EwBeta` — 2-input pair statistics. Matches
  pandas `ewm(adjust=True).cov / .corr` bit-exactly. EwBeta follows
  the CAPM `(target, regressor)` convention.

#### Moving averages

* `WMA` — linearly-weighted moving average, O(1) per step via the
  identity `W[t] − W[t−1] = w·x[t] − S[t−1]`.
* `DEMA`, `TEMA` — Mulloy's double/triple EMA compositions.
* `TRIMA` — triangular MA (`SMA(SMA(x))`).
* `HullMA` — `WMA(2·WMA(n/2) − WMA(n), √n)`.
* `KAMA` — Kaufman Adaptive MA with O(1) per step; matches TA-Lib
  bit-exactly.

#### Momentum / oscillators

* `MACD` (1→3), `WilliamsR` (3→1), `Stoch` (3→2, fast and slow
  via `smooth_k`), `StochRSI` (1→2), `TRIX`, `BOP` (4→1), `CCI`
  (3→1), `UltimateOscillator` (3→1), `ADX` (3→3 returning
  `+DI` / `-DI` / `ADX`).
* `ROC`, `ROCP`, `ROCR` — TA-Lib rate-of-change family.
* `RollingRSI` default changed to **Wilder's smoothing** (matches
  TA-Lib and pandas-ta); `method="cutler"` preserves the old
  Cutler form. Earlier versions diverged from TA-Lib's `RSI` by
  ~11 RSI points -- now bit-exact.

#### Volatility / range

* Range-based volatility quartet: **Parkinson**, **Garman-Klass**,
  **Rogers-Satchell**, **Yang-Zhang**. Each ships in `Var` and
  `Vol` variants, with rolling and EW smoothing (for the first
  three) -- 14 classes total.
* `TrueRange`, `ATR(window)`, `NATR(window)` — Wilder's
  bar-aware volatility family.
* `DonchianChannels` (2→3), `KeltnerChannels` (3→3) — channel
  indicators.

#### Volume-aware

* `RollingVWAP`, `OBV`, `AD`, `ADOSC`, `MFI` — first volume-aware
  primitives in screamer; all bit-exact to TA-Lib counterparts
  (except `ADOSC`, which inherits the documented EMA-convention
  divergence).

#### Performance / risk

* `Drawdown`, `MaxDrawdown`, `RollingMaxDrawdown`,
  `RollingSharpe(window, periods_per_year)`,
  `RollingSortino(window, ppy, target)`,
  `RollingInfoRatio(window, ppy)`,
  `RollingCalmar(window, ppy)`,
  `RollingHitRate(window)` — backtest-evaluation metrics. None of
  these are in TA-Lib; they're a real differentiator for screamer
  in trading pipelines.

#### Statistical / regression

* `RollingAlpha` — companion intercept to `RollingBeta`.
* `RollingResidualStd` — std of the per-bar `RollingSpread`.
* `RollingLinearRegression` (2→4) — full OLS fit returning
  `(slope, intercept, r², stderr)`. First 2→4 consumer of the
  N→M dispatcher. `stderr` is the RMSE of residuals (standard
  error of estimate, not slope-stderr).
* `RollingTSF` — TA-Lib's Time-Series Forecast (regression vs
  time projected one step ahead), bit-exact to `talib.TSF`.
* `RollingHurst(window, min_scale=4, method='rs')` — rolling Hurst
  exponent via Anis-Lloyd corrected rescaled-range analysis at
  dyadic scales. Bit-exact to the reference Python implementation;
  ~0.5 on white noise, >0.5 on integrated processes.

#### Signal processing

* `ButterHighpass`, `ButterBandpass`, `ButterBandstop` —
  high-pass, band-pass, band-stop Butterworth IIR filters.
  Added the underlying `lp2hp_zpk` / `lp2bp_zpk` / `lp2bs_zpk`
  ZPK transformations so future Bessel/Cheby/Elliptic families
  also get all four btypes once their prototypes are written.
* `MovingAverage(taps)` — FIR with arbitrary user-supplied taps
  (pair with `np.hamming` / `np.kaiser` / `scipy.signal.firwin`).
* `KalmanFilter(process_var, observation_var)` — scalar 1-D
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


