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
- `N→M` is multi-input multi-output (Plan E done; `Cart2Polar`/`Polar2Cart` are the first 2→2 consumers).

Priority: 🔴 high, 🟡 medium, ⚪ low.


## Math / element-wise transforms

### What we have

| | |
|---|---|
| `Abs`, `Sign`, `Exp`, `Log`, `Sqrt` | scalar functions |
| `Square`, `Cube` | precomposed `x^2`, `x^3` (faster than `Power(2)`/`Power(3)`) |
| `Floor`, `Ceil`, `Round` | rounding (`Round` uses banker's rounding, matching `numpy.round`) |
| `Sin`, `Cos`, `Atan` | trig (useful for cyclical features) |
| `Asin`, `Acos` | inverse trig (NaN outside `[-1, 1]`) |
| `Hypot(x, y)`, `Atan2(y, x)` | 2-input Euclidean distance and signed angle |
| `Cart2Polar`, `Polar2Cart` | 2→2 inverse-pair coordinate transforms |
| `Erf`, `Erfc` | error function and complement |
| `Tanh`, `Sigmoid`, `Softsign`, `Relu`, `Selu`, `Elu` | activations (more useful for ML pipelines than finance) |
| `Linear(a, b)` | `a * x + b` |
| `Power(p)` | `x^p` |

### Gaps

| Function | Description | Quadrant | Priority | Note |
|---|---|---|---|---|
| `Compose(f, g, ...)` | declarative composition | 1→1 | ⚪ | superseded by the planned **DAG-of-transforms** design; tracked separately |


## Misc / simple transforms

### What we have

| | |
|---|---|
| `Diff(k)` | `x[t] - x[t-k]` |
| `Lag(k)` | `x[t-k]` |
| `Clip(low, high)` | clamp |
| `Ffill` | forward-fill NaN |
| `FillNa(value)` | replace NaN with constant |
| `CumSum`, `CumProd` | running sums (matches `numpy.cumsum`/`cumprod`) |
| `CumMax`, `CumMin` | running extrema (matches `numpy.maximum/minimum.accumulate`) |
| `Detrend(window)` | `x[t]` minus a rolling-mean baseline |
| `Diff2` | second-order finite difference (discrete second derivative) |
| `Identity` | pass-through, useful as a pipeline placeholder |

### Gaps

| Function | Description | Quadrant | Priority | Note |
|---|---|---|---|---|
| `Pct` | percent change | 1→1 | ⚪ | already covered by `Return(k)` in the financial section |


## Rolling-window statistics

### What we have

`RollingMean`, `RollingSum`, `RollingVar`, `RollingStd`, `RollingSkew`,
`RollingKurt`, `RollingZscore`, `RollingMin`, `RollingMax`,
`RollingMinMax`, `RollingMedian`, `RollingQuantile`, `RollingRms`,
`RollingPoly1`, `RollingPoly2`, `RollingSigmaClip`, `RollingOU`,
`RollingMad`, `RollingArgmin`, `RollingArgmax`, `RollingRange`,
`RollingIqr`. The monotonic-deque primitive is centralised in
`detail::MonotonicDeque<bool IsMax>`, shared by `Min`/`Max`/`MinMax`/
`Argmin`/`Argmax`/`Range`. `RollingIqr` shares a single OST instead
of running two `RollingQuantile`s (half the memory and inserts).

This is a strong set; pandas `Series.rolling.*` is essentially fully
covered.

### Gaps

| Function | Description | Quadrant | Priority | Note |
|---|---|---|---|---|
| `RollingApply(fn, window)` | user-supplied Python function | 1→1 | ⚪ | breaks the all-C++ guarantee; only worth it if users complain |
| `RollingTrimmedMean` | trimmed-mean for robust mean | 1→1 | ⚪ | rare in finance; dedicated O(log W) needs OST extension with subtree sums |
| `Expanding{Mean,Var,...}` | growing-window variants (no max size) | 1→1 | 🟡 | the `start_policy="expanding"` param partially serves this; a dedicated class would have cleaner semantics |


## Exponentially-weighted statistics

### What we have

`EwMean`, `EwVar`, `EwStd`, `EwZscore`, `EwSkew`, `EwKurt`, `EwRms`,
`EwCov`, `EwCorr`, `EwBeta`. The pair statistics use the same
bias-corrected convention as `EwVar` and match
`pandas.Series.ewm(adjust=True, ...).cov / .corr` to floating-point
precision (verified in `tests/test_ew_pair.py`).

### Gaps

No prioritised gaps remain. (`EwMin` and `EwMax` were considered and
dropped: the running min/max under exponential weighting has no
agreed-upon definition and is rarely asked for in practice.)


## Moving averages and trend (technical-analysis "overlap studies")

### What we have

| | |
|---|---|
| `RollingMean` | Simple Moving Average (SMA) |
| `EwMean` | Exponential Moving Average (EMA) |
| `WMA` | Linearly-weighted MA (newest = weight w, oldest = 1). O(1) per step via the identity `W[t] - W[t-1] = w·x[t] - S[t-1]` where `S` is the simple rolling sum of the previous window |
| `DEMA`, `TEMA` | Double / Triple Exponential MA (Mulloy 1994). Pure compositions: `DEMA = 2·EMA − EMA(EMA)`, `TEMA = 3·EMA − 3·EMA(EMA) + EMA(EMA(EMA))` |
| `TRIMA` | Triangular MA. `SMA(SMA(x, n_inner), n_outer)` with TA-Lib's window split |
| `HullMA` | Hull MA (Hull 2005). `WMA(2·WMA(n/2) − WMA(n), √n)` |
| `KAMA` | Kaufman Adaptive MA (Kaufman 1998). SC adapts to the efficiency ratio (net displacement / total absolute travel). Validated bit-for-bit against TA-Lib and pandas-ta to ~1e-15 |
| `RollingPoly1`, `RollingPoly2` | linear / quadratic regression fit at the window endpoint (gives smoothed trend, slope, second derivative) |

### Gaps

| Function | Description | Quadrant | Priority | Note |
|---|---|---|---|---|
| `MAMA` (MESA) | Hilbert-transform-based adaptive MA | 1→1 | ⚪ | requires Hilbert primitive we don't have; specialised |
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
| `KeltnerChannels(window, num_atr)` | `(lower, mid, upper)` like Bollinger but using ATR for bandwidth | 3→3 | 🟡 | uses the now-available `N→M` dispatch (Plan E) |
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
| `RollingLinearRegression` | `(slope, intercept, r_squared, std_err)` of x on y | 2→4 | 🟡 | matches TA-Lib's `LINEARREG_*`; uses the now-available `N→M` dispatch (Plan E) |
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
3. **Risk metrics batch**: `Drawdown`, `MaxDrawdown`, `RollingSharpe`, `RollingSortino`. `CumMax`/`CumMin` are already in place, so `Drawdown` is now a one-liner. Differentiates `screamer` from TA-Lib.
4. **Volume batch**: `VWAP`, `OBV`, `AD`, `MFI`. Adds whole new feature category.
5. **Momentum batch**: `MACD`, `Stoch`, `WilliamsR`, `CCI`, `ROC`. Restores TA-Lib coverage for the most popular oscillators.

Each batch is one focused session. The multi-input auto-test harness
(currently `tests/test_rolling_two_input.py`,
`tests/test_one_input_multi_output.py`, and `tests/test_geometry.py`)
is enough infrastructure to verify a batch end-to-end.

All four `FunctorBase` quadrants are implemented: 1→1, N→1, 1→M, and
N→M. The shape rule for any input/output combination is
`output.shape == single_input.shape + (M,)` with column-by-column
pairing across the `N` inputs.
