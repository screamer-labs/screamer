# Roadmap: new functions

This document is an inventory of what `screamer` ships today, mapped against
what comparable signal-processing and financial-indicator libraries have,
with a list of additions worth picking up next. It serves two purposes.
For users: a quick view of what's coming. For contributors: a prioritised
backlog of useful work.

Reference libraries used for the comparison:

- **TA-Lib** (`talib`): the canonical 150-indicator financial library.
- **pandas / `pandas-ta` / `ta`**: Python-native technical analysis.
- **scipy.signal**: classical DSP (filters, transforms, spectra).
- **numpy / pandas.rolling**: the rolling-window baseline.
- **FilterPy**: adaptive filters (Kalman family).

Quadrant labels in the gap tables refer to the `FunctorBase<_, N, M>`
template (see [`polymorphic_api.md`](polymorphic_api.md)):

- `1→1` is the standard one-stream-in, one-stream-out case
  (`ScreamerBase` or `FunctorBase<_, 1, 1>`). Already implemented.
- `N→1` is multi-input, one output (Plan D done).
- `1→M` is one input, multi-output (Plan C done, M up to 3 used).
- `N→M` is multi-input multi-output (no consumer yet, dispatcher throws).

Priority: 🔴 high, 🟡 medium, ⚪ low.


## Math / element-wise transforms

### What we have

| | |
|---|---|
| `Abs`, `Sign`, `Exp`, `Log`, `Sqrt`, `Erf`, `Erfc` | scalar functions |
| `Tanh`, `Sigmoid`, `Softsign`, `Relu`, `Selu`, `Elu` | activations (more useful for ML pipelines than finance) |
| `Linear(a, b)` | `a * x + b` |
| `Power(p)` | `x^p` |

### Gaps

| Function | Description | Quadrant | Priority | Note |
|---|---|---|---|---|
| `Floor`, `Ceil`, `Round` | rounding | 1→1 | ⚪ | thin wrappers, easy |
| `Square`, `Cube` | precomposed common powers | 1→1 | ⚪ | `Power(2)` covers it |
| `Sin`, `Cos`, `Atan` | trig (useful for cyclical features) | 1→1 | ⚪ | mostly relevant for time-of-day features |
| `Compose(f, g)` | function composition object | 1→1 | ⚪ | Python lambdas already work; would be a perf optimisation |


## Misc / simple transforms

### What we have

| | |
|---|---|
| `Diff(k)` | `x[t] - x[t-k]` |
| `Lag(k)` | `x[t-k]` |
| `Clip(low, high)` | clamp |
| `Ffill` | forward-fill NaN |
| `FillNa(value)` | replace NaN with constant |

### Gaps

| Function | Description | Quadrant | Priority | Note |
|---|---|---|---|---|
| `CumSum`, `CumProd` | running sums | 1→1 | 🟡 | very common; useful for PnL aggregation |
| `CumMax`, `CumMin` | running extrema | 1→1 | 🟡 | needed for drawdown |
| `Detrend(window)` | subtract rolling mean | 1→1 | 🟡 | composite of `RollingMean` + subtraction; convenient |
| `Diff2`, `Pct` | second difference, % change | 1→1 | ⚪ | composable from `Diff` and arithmetic |
| `Identity` | pass-through | 1→1 | ⚪ | mostly useful as a placeholder in pipelines |


## Rolling-window statistics

### What we have

`RollingMean`, `RollingSum`, `RollingVar`, `RollingStd`, `RollingSkew`,
`RollingKurt`, `RollingZscore`, `RollingMin`, `RollingMax`,
`RollingMinMax`, `RollingMedian`, `RollingQuantile`, `RollingRms`,
`RollingPoly1`, `RollingPoly2`, `RollingSigmaClip`, `RollingOU`.

This is a strong set; pandas `Series.rolling.*` is essentially fully
covered.

### Gaps

| Function | Description | Quadrant | Priority | Note |
|---|---|---|---|---|
| `RollingMad` | Mean absolute deviation, `mean(|x - rolling_mean|)` | 1→1 | 🟡 | robust scale measure; talib has `MEDPRICE` siblings |
| `RollingArgmin`, `RollingArgmax` | index of min/max within window | 1→1 | 🟡 | TA-Lib's `MAXINDEX`, `MININDEX` |
| `RollingRange` | `max − min` | 1→1 | 🟡 | trivial composite of `RollingMinMax` |
| `RollingIqr` | inter-quartile range | 1→1 | ⚪ | from `RollingQuantile` |
| `RollingApply(fn, window)` | user-supplied Python function | 1→1 | ⚪ | breaks the all-C++ guarantee; only worth it if users complain |
| `RollingTrimmedMean` | trimmed-mean for robust mean | 1→1 | ⚪ | rare in finance |
| `Expanding{Mean,Var,...}` | growing-window variants (no max size) | 1→1 | 🟡 | the `start_policy="expanding"` param partially serves this; a dedicated class would have cleaner semantics |


## Exponentially-weighted statistics

### What we have

`EwMean`, `EwVar`, `EwStd`, `EwZscore`, `EwSkew`, `EwKurt`, `EwRms`.

### Gaps

| Function | Description | Quadrant | Priority | Note |
|---|---|---|---|---|
| `EwCov`, `EwCorr` | exponentially-weighted covariance / correlation of two streams | 2→1 | 🟡 | matches `pd.Series.ewm.cov/corr`; useful for fast-moving regimes |
| `EwBeta` | EW slope of x on y | 2→1 | 🟡 | composable from `EwCov` and `EwVar` |
| `EwMin`, `EwMax` | non-trivial in EW form; usually approximated by other means | 1→1 | ⚪ | rarely asked for |


## Moving averages and trend (technical-analysis "overlap studies")

### What we have

| | |
|---|---|
| `RollingMean` | Simple Moving Average (SMA) |
| `EwMean` | Exponential Moving Average (EMA) |
| `RollingPoly1`, `RollingPoly2` | linear / quadratic regression fit at the window endpoint (gives smoothed trend, slope, second derivative) |

### Gaps

| Function | Description | Quadrant | Priority | Note |
|---|---|---|---|---|
| `WMA` | Weighted Moving Average (linear weights) | 1→1 | 🟡 | one of the three classics with SMA / EMA |
| `DEMA`, `TEMA` | Double / Triple Exponential MA | 1→1 | 🟡 | composites: `DEMA = 2·EMA − EMA(EMA)` |
| `KAMA` | Kaufman Adaptive Moving Average | 1→1 | 🟡 | adaptive smoothing based on volatility-to-noise ratio |
| `TRIMA` | Triangular Moving Average | 1→1 | ⚪ | `RollingMean` of a `RollingMean` |
| `MAMA` (MESA) | Hilbert-transform-based adaptive MA | 1→1 | ⚪ | requires Hilbert; specialised |
| `HullMA` | Hull Moving Average | 1→1 | ⚪ | composite of WMAs |
| `SavGol(window, order)` | Savitzky-Golay filter | 1→1 | 🟡 | `RollingPoly{1,2}` is the order=1/2 version centred at the right endpoint; `SavGol` would generalise to user-specified polynomial order and be the proper "savgol" |


## Momentum / oscillators

### What we have

| | |
|---|---|
| `RollingRSI` | Relative Strength Index |
| `Return`, `LogReturn` | simple / log returns over a delay |

### Gaps

| Function | Description | Quadrant | Priority | Note |
|---|---|---|---|---|
| `MACD` | Moving Average Convergence Divergence | 1→3 | 🔴 | `(macd, signal, histogram)`; TA-Lib's most-used non-MA indicator |
| `Stoch` | Stochastic oscillator | 1→2 | 🔴 | `(%K, %D)` over rolling min/max range |
| `StochRSI` | Stochastic of RSI | 1→2 | 🟡 | composite |
| `WilliamsR` | Williams %R | 1→1 | 🟡 | normalised position within rolling range |
| `ROC`, `ROCP`, `ROCR` | rate-of-change variants | 1→1 | 🟡 | `Return` covers ROCP; explicit names would help discoverability |
| `Momentum(k)` | `x[t] - x[t-k]` | 1→1 | 🟡 | identical to `Diff(k)`; alias would aid TA-Lib porting |
| `CCI` | Commodity Channel Index | 3→1 | 🟡 | needs (high, low, close) typical-price |
| `ADX` | Average Directional Index | 3→1 | 🟡 | needs high/low/close; multi-step but fits the dispatcher |
| `TRIX` | triple-smoothed momentum | 1→1 | ⚪ | composite of EMAs |
| `BOP` | Balance of Power | 4→1 | ⚪ | needs OHLC |
| `UltimateOscillator` | weighted average of three timeframes | 3→1 | ⚪ | niche |


## Volatility / range

### What we have

| | |
|---|---|
| `RollingStd`, `EwStd` | standard deviations |
| `RollingVar`, `EwVar` | variances |
| `BollingerBands(window, num_std)` | `(lower, mid, upper)` |
| `RollingMinMax` | range bounds |

### Gaps

| Function | Description | Quadrant | Priority | Note |
|---|---|---|---|---|
| `TrueRange` | `max(high - low, |high - prev_close|, |low - prev_close|)` | 3→1 | 🔴 | the building block for ATR; the canonical OHLC-aware volatility |
| `ATR(window)` | Average True Range = rolling mean of TrueRange | 3→1 | 🔴 | the canonical financial volatility measure |
| `NATR` | Normalised ATR (`ATR / close * 100`) | 3→1 | 🟡 | trivial wrapper |
| `KeltnerChannels(window, num_atr)` | `(lower, mid, upper)` like Bollinger but using ATR for bandwidth | 3→3 | 🟡 | needs `N→M` dispatch ("Plan E"); useful test case for the M>1 N>1 quadrant |
| `DonchianChannels(window)` | `(lower, mid, upper)` from rolling min, mid (avg of min/max), max | 1→3 | 🟡 | already feasible today (M=3 dispatch is done) |
| `ParkinsonVol` | high-low range volatility estimator | 2→1 | 🟡 | quant-finance staple |
| `GarmanKlassVol` | OHLC volatility estimator | 4→1 | ⚪ | niche; needs OHLC |


## Volume-aware indicators (price + volume)

### What we have

None. All the volume-based indicators below are genuinely missing.

### Gaps

| Function | Description | Quadrant | Priority | Note |
|---|---|---|---|---|
| `VWAP(window)` | Volume-Weighted Average Price | 2→1 | 🔴 | extremely common; trivial: `RollingSum(price · vol) / RollingSum(vol)` |
| `OBV` | On-Balance Volume (cumulative signed volume) | 2→1 | 🟡 | streaming-friendly |
| `AD` | Chaikin Accumulation / Distribution Line | 4→1 | 🟡 | needs OHLCV |
| `ADOSC` | Chaikin A/D Oscillator (EMA difference of AD) | 4→1 | 🟡 | composite |
| `MFI` | Money Flow Index | 4→1 | 🟡 | RSI-like but volume-weighted |


## Statistical / regression

### What we have

| | |
|---|---|
| `RollingCov`, `RollingCorr` | pairwise covariance / correlation |
| `RollingBeta` | regression slope of x on y |
| `RollingSpread` | hedge-adjusted residual `x − β·y` |
| `RollingPoly1`, `RollingPoly2` | OLS fit at window endpoint, returning value or derivatives |
| `RollingZscore`, `EwZscore` | normalised position |
| `RollingFracDiff` | fractional differentiation |
| `RollingOU` | Ornstein-Uhlenbeck parameter fit |

### Gaps

| Function | Description | Quadrant | Priority | Note |
|---|---|---|---|---|
| `RollingAlpha` | regression intercept companion to `RollingBeta` | 2→1 | 🟡 | `α = mean(x) − β · mean(y)` |
| `RollingResidualStd` | std of residuals from `RollingSpread`, useful for normalised pairs trading | 2→1 | 🟡 | composite |
| `RollingLinearRegression` | `(slope, intercept, r_squared, std_err)` of x on y | 2→4 | 🟡 | matches TA-Lib's `LINEARREG_*`; needs N>1 M>1 ("Plan E") |
| `RollingTSF` | Time-Series Forecast (linear regression projected one step) | 1→1 | ⚪ | similar to `RollingPoly1` with derivative + endpoint extrapolation |
| `RollingRank` | rank of latest value within window | 1→1 | 🟡 | also called `quantile_rank`; useful for cross-sectional features |
| `RollingPercentile` | inverse of `RollingQuantile`: where does the current value sit? | 1→1 | 🟡 | very common in cross-sectional finance |
| `RollingCovMatrix` | full Gram matrix of K series | K→K² | ⚪ | huge output; specialised dispatcher needed; probably better delivered as a separate batch tool |


## Performance / risk

These are present in `pyfolio`, `quantstats`, and `pandas` recipes but
not as streaming primitives in any reference library. Adding them to
`screamer` is a real differentiator.

### Gaps

| Function | Description | Quadrant | Priority | Note |
|---|---|---|---|---|
| `Drawdown` | running drawdown from running peak | 1→1 | 🔴 | needs `CumMax`; one of the most-asked-for metrics |
| `MaxDrawdown(window)` | rolling max-drawdown | 1→1 | 🔴 | depth + duration are both useful |
| `RollingSharpe(window, periods_per_year)` | rolling annualised Sharpe ratio | 1→1 | 🔴 | the canonical risk-adjusted return |
| `RollingSortino` | downside-only-volatility variant | 1→1 | 🟡 | |
| `RollingInfoRatio` | active return / tracking error (vs benchmark) | 2→1 | 🟡 | |
| `RollingCalmar` | annualised return / max-drawdown | 1→1 | ⚪ | composite |
| `RollingHitRate` | fraction of positive samples in the window | 1→1 | 🟡 | useful for strategy-evaluation streams |


## Signal processing

### What we have

| | |
|---|---|
| `Butter(order, cutoff)` | low-pass Butterworth filter |

### Gaps

| Function | Description | Quadrant | Priority | Note |
|---|---|---|---|---|
| `Butter(...)` extensions | high-pass, band-pass, band-stop variants | 1→1 | 🟡 | one parameter from current implementation |
| `Bessel`, `Cheby1`, `Cheby2`, `Elliptic` | other classical IIR families | 1→1 | ⚪ | scipy.signal coverage |
| `MovingAverage(taps)` | finite-impulse-response (FIR) with arbitrary taps | 1→1 | 🟡 | covers Hamming / Blackman / Kaiser windows |
| `Hilbert` | Hilbert transform (analytic signal) | 1→2 | ⚪ | foundation for instantaneous amplitude / frequency, MAMA, etc. |
| `KalmanFilter` | classical 1-D Kalman filter | 1→1 | ⚪ | several adaptive financial techniques rely on this; FilterPy has it offline |


## Scope deliberately excluded

These are popular elsewhere but do not fit the streaming / `O(1)`-per-step
model that defines `screamer`, and are better served by other libraries.

- **Candlestick patterns** (`CDL_DOJI`, `CDL_HAMMER`, etc.). These are
  pattern-matching against fixed K-bar shapes; closer to a rules engine
  than a streaming algorithm. Use TA-Lib directly when needed.
- **Cycle indicators** (`HT_DCPERIOD`, `HT_DCPHASE`). All Hilbert-based
  and only meaningful over substantial history; the streaming form is
  awkward.
- **Cointegration tests** (Johansen, ADF). Statistical tests rather than
  rolling indicators; tests need full-window data and don't decompose
  cleanly into per-step updates.
- **Pattern recognition / regime detection**. Ill-defined as a
  fixed-shape streaming function.


## Suggested next batches

If we tackle this in iterations of "5-ish related additions per release",
sensible groupings:

1. **OHLC volatility batch**: `TrueRange` → `ATR` → `NATR` → `KeltnerChannels` (forces Plan E for `N→M`) → `DonchianChannels`. Three-input dispatcher gets a heavy real workout.
2. **MA family batch**: `WMA`, `DEMA`, `TEMA`, `KAMA`, `SavGol`. All `1→1`, easy, fills the most-requested TA-Lib gap.
3. **Risk metrics batch**: `CumMax`, `CumMin`, `Drawdown`, `MaxDrawdown`, `RollingSharpe`. Differentiates `screamer` from TA-Lib.
4. **Volume batch**: `VWAP`, `OBV`, `AD`, `MFI`. Adds whole new feature category.
5. **Momentum batch**: `MACD`, `Stoch`, `WilliamsR`, `CCI`, `ROC`. Restores TA-Lib coverage for the most popular oscillators.

Each batch is one focused session. Before any of them, the multi-input
auto-test harness (currently `tests/test_rolling_two_input.py` and
`tests/test_one_input_multi_output.py`) is enough infrastructure to
verify a batch end-to-end.

If the next batch needs `N→M` dispatch (Keltner, full LinearRegression),
a "Plan E" extension to `FunctorBase` is the prerequisite. The shape
rule would be the natural product: `output.shape == input.shape + (M,)`
for each of the `N` parallel input streams' time axes, with the
column-by-column pairing already used in the `N→1` path.
