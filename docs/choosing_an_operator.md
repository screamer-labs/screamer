---
name: choosing_an_operator
title: Choosing an operator
kind: guide
short: A task-oriented map from what you want to do to the operator that does it.
topics:
- getting-started
---

# Choosing an operator

Each section below maps a common signal-processing task to the operators that cover it, with a one-line note on when to prefer each.

## Smooth or denoise a series

| Intent | Operator(s) | Note |
|:-------|:------------|:-----|
| Simple trailing mean, fast and predictable lag | `RollingMean` | The baseline; use when you want a plain symmetric window. |
| Exponentially weighted mean, more recent samples weighted higher | `EwMean` | No fixed window boundary; weight decays geometrically. |
| Robust smooth insensitive to outliers | `RollingMedian` | Slower than `RollingMean` but not pulled by spikes. |
| Weighted moving average with a linear ramp | `WMA` | Puts more weight on recent samples inside the window. |
| Double exponential moving average, less lag than `EwMean` | `DEMA` | Trades some smoothness for faster response. |
| Triple exponential moving average | `TEMA` | Further reduces lag at the cost of overshoot on sharp moves. |
| Triangular moving average, double-smoothed for extra noise rejection | `TRIMA` | Effective on slow, noisy signals; longer effective lag. |
| Hull moving average, low lag with reduced whipsaw | `HullMA` | Good balance of lag and smoothness for trend following. |
| Adaptive moving average that slows during ranging markets | `KAMA` | Speeds up when the move is directional, slows when it chops (the efficiency ratio). |
| Frequency-domain Butterworth low-pass filter | `Butter` | Set `cutoff_freq` to keep only slow components. |
| Minimum-phase IIR filter for any custom passband | `MovingAverage` | General-purpose FIR/IIR; see its page for coefficient inputs. |
| Optimal linear filter tracking a latent state with noise | `KalmanFilter` | Use when signal and observation noise variances are known or can be estimated; see [notebook 17](notebooks/17-filtering-and-forecasting-with-uncertainty). |

## Remove outliers or despike

| Intent | Operator(s) | Note |
|:-------|:------------|:-----|
| Replace spikes with window median, with statistical detection | `Hampel` | Uses median absolute deviation to flag and replace outliers. |
| Clip samples that fall outside a rolling sigma band | `RollingSigmaClip` | Replaces values beyond N standard deviations of the local window. |
| Remove isolated single-sample spikes by comparing neighbours | `ImpulseClip` | Designed for sharp, one-sample impulses in otherwise smooth data. |
| Hard absolute clip to a fixed range | `Clip` | Simple clamp; use when the valid range is known a priori. |

## Measure dispersion or volatility

| Intent | Operator(s) | Note |
|:-------|:------------|:-----|
| Rolling standard deviation or variance | `RollingStd`, `RollingVar` | Standard close-to-close estimators over a trailing window. |
| Rolling interquartile range | `RollingIqr` | Robust to outliers; appropriate when distribution is heavy-tailed. |
| Rolling min-to-max range | `RollingRange` | Useful as a quick spread measure or normalizer. |
| Rolling median absolute deviation | `RollingMad` | Robust location-free dispersion; pairs well with `RollingMedian`. |
| Exponentially weighted standard deviation or variance | `EwStd`, `EwVar` | Continuously weighted; more weight on recent samples. |
| OHLC Garman-Klass volatility (rolling or EW) | `RollingGarmanKlassVol`, `EwGarmanKlassVol` | Uses open/high/low/close to reduce variance of the estimator. |
| OHLC Parkinson volatility (rolling or EW) | `RollingParkinsonVol`, `EwParkinsonVol` | High/low range estimator; assumes no overnight gap. |
| OHLC Rogers-Satchell volatility (rolling or EW) | `RollingRogersSatchellVol`, `EwRogersSatchellVol` | Corrects for drift; generally preferred over Parkinson. |
| Yang-Zhang volatility (rolling) | `RollingYangZhangVol` | Combines overnight gap and intraday range; lowest variance among OHLC estimators. |

## Fit a local trend or slope

| Intent | Operator(s) | Note |
|:-------|:------------|:-----|
| Linear regression over a trailing window (slope and intercept) | `RollingLinearRegression` | Returns slope, intercept, and forecast for each window. |
| Polynomial fit of degree 1 or 2 over a trailing window | `RollingPoly1`, `RollingPoly2` | Use `derivative_order=1` to extract the slope directly. |
| Whole-history expanding linear slope | `ExpandingSlope` | Slope over all data seen so far; grows less noisy as N increases. |
| Time-series forecast (endpoint of regression line) | `RollingTSF` | Returns the fitted value at the window's right edge. |
| Triple-smoothed rate of change (momentum) | `TRIX` | One-period rate of change of a triple-exponential average; filters short-term noise. |

## Relate two series

| Intent | Operator(s) | Note |
|:-------|:------------|:-----|
| Rolling Pearson correlation | `RollingCorr` | Normalized correlation in [-1, 1] over a trailing window. |
| Rolling regression slope (beta) | `RollingBeta` | OLS slope of y on x over a trailing window. |
| Rolling regression covariance | `RollingCov` | Unnormalized; divide by rolling variance to get beta. |
| Rolling regression spread (residual) | `RollingSpread` | The residual after hedging y by x at the rolling beta. |
| Rolling regression intercept (alpha) | `RollingAlpha` | The drift component of a rolling linear regression. |
| Exponentially weighted beta, correlation, or covariance | `EwBeta`, `EwCorr`, `EwCov` | Continuous weighting; more responsive to recent co-movement changes. |

For a worked pairs-trading example using `RollingBeta`, `RollingSpread`, `RollingAlpha`, and `RollingOU`, see [notebook 18](notebooks/18-pairs-and-mean-reversion).

## Mean reversion and forecasting

| Intent | Operator(s) | Note |
|:-------|:------------|:-----|
| Ornstein-Uhlenbeck mean-reversion parameters from a spread | `RollingOU` | Fits OU to a trailing window; `output=` selects "mrr"/"mean"/"relmean"/"std". |
| Online Bayesian regression with calibrated prediction intervals | `BayesianRegression` | Returns slope, intercept, predictive mean, and predictive std per sample; interval widens early and tightens as evidence accumulates. See [notebook 17](notebooks/17-filtering-and-forecasting-with-uncertainty). |
| Kalman filter as a recursive forecaster | `KalmanFilter` | Tracks a latent level and returns the filtered estimate; tune `process_var` and `observation_var` to set the lag-vs-noise trade-off. |

## Detect cycles and regimes

| Intent | Operator(s) | Note |
|:-------|:------------|:-----|
| Measure the dominant cycle period | `DominantCycle` | Ehlers' homodyne discriminator; the period in samples. |
| Instantaneous phase, frequency, or amplitude of a cycle | `CyclePhase`, `CycleFrequency`, `CycleAmplitude` | From the analytic signal; `HilbertPhasor` gives its in-phase and quadrature parts. |
| Sinewave turning-point indicator | `CycleSine` | Sine and lead-sine lines; a crossing marks a cycle turn. |
| Classify trending versus cycling | `TrendMode` | 1.0 while trending, 0.0 while cycling. |
| Isolate a band of cycle periods | `RoofingFilter` | A highpass then a `SuperSmoother`; feed its output to the cycle operators. |
| Low-lag smoothing or trend | `SuperSmoother`, `Decycler`, `InstantaneousTrendline` | A 2-pole lowpass, a trend with short cycles removed, and a trendline that adapts to the measured cycle. |
| Adaptive moving average pair | `MAMA` | MAMA and FAMA lines whose speed follows the phase rate. |
| Turn a noisy indicator into a stable on/off state | `SchmittTrigger` | Hysteresis between two thresholds avoids flip-flopping. |

For a worked example see [notebook 19](notebooks/19-cycle-and-spectral-analysis).

## Analyse order flow and liquidity

| Intent | Operator(s) | Note |
|:-------|:------------|:-----|
| Sign trades as buyer- or seller-initiated | `TickRuleSign`, `LeeReadySign`, `BulkVolumeClassifier` | Tick rule from prices only; Lee-Ready with quotes; bulk classification from bar returns. |
| Signed volume and order-flow imbalance | `SignedVolume`, `OFI`, `RollingOrderImbalance` | Net aggressive flow per trade, normalized, or summed over a window. |
| Book imbalance and fair value | `QueueImbalance`, `MicroPrice`, `ContOFI` | From L1 quotes: size imbalance, an imbalance-weighted mid, and order-flow imbalance from book events. |
| Trading cost | `EffectiveSpread`, `RealizedSpread`, `RollSpread` | Cost against the mid, its liquidity part, and a spread estimate from trade prices alone. |
| Price impact | `RollingKyleLambda`, `EwKyleLambda`, `Propagator`, `AmihudIlliquidity` | Impact slope of return on signed flow, a decaying-kernel impact model, and illiquidity per unit of notional. |
| Flow toxicity and clustering | `VPIN`, `HawkesIntensity` | Probability of informed trading, and the intensity of self-exciting arrivals. |

See the [Market microstructure](microstructure) guide and notebooks
[11](notebooks/11-microstructure-order-flow),
[12](notebooks/12-microstructure-price-impact), and
[13](notebooks/13-microstructure-toxicity-and-book).

## Measure risk and performance

| Intent | Operator(s) | Note |
|:-------|:------------|:-----|
| Drawdown from the running peak | `Drawdown`, `MaxDrawdown`, `RollingMaxDrawdown` | The current drawdown, the worst so far, or the worst inside a window. |
| Risk-adjusted return over a window | `RollingSharpe`, `RollingSortino`, `RollingCalmar`, `RollingInfoRatio`, `RollingOmega` | Annualised Sharpe, Sortino, Calmar, and information ratios, and the Omega ratio. |
| Tail and downside risk | `RollingCVaR`, `RollingDownsideDeviation` | Expected shortfall in the worst tail, and the deviation of returns below a threshold. |
| Share of winning periods | `RollingHitRate` | Fraction of positive samples in the window. |
| Persistence or mean reversion of a series | `RollingHurst` | Hurst exponent: above 0.5 trends, below 0.5 reverts. |

## Clean and prepare data

| Intent | Operator(s) | Note |
|:-------|:------------|:-----|
| Fill missing values | `Ffill`, `FillNa` | Carry the last finite value forward, or replace NaN with a constant. |
| Drop missing events from a stream | `Dropna` | Removes events instead of filling them. |
| Mark missing or infinite values | `IsNan`, `IsFinite` | 1.0 or 0.0 masks, for `Filter` or `Where`. |
| Make a series stationary but keep its memory | `FracDiff` | Fractional differencing removes the trend with less information loss than a first difference. |
| Standardise or rank | `RollingZscore`, `EwZscore`, `RollingRank` | Scale-free features for models and thresholds. |
| Shift a series in time | `Lag`, `Delay` | `Lag` shifts by samples; `Delay` shifts an indexed stream by a time offset. |
| Build a supervised training set | `forecast_pairs` | Aligns lagged features with a future target without lookahead; see [notebook 16](notebooks/16-supervised-forecasting). |

## Build bars from ticks

`Resample` turns a tick stream into bars. Use `freq=` for time bars or `count=`
for bars of a fixed number of events, and `agg=` to choose the reducer:
`"first"`, `"max"`, `"min"`, and `"last"` give open, high, low, and close. See
notebooks [9](notebooks/09-bars-from-ticks) and
[10](notebooks/10-custom-and-multi-column-bars).

## Expanding (whole-history) vs rolling (trailing-window)

Many operators come in both flavors. A rolling operator sees only the most recent `window_size` samples, so its estimates adapt quickly but can be noisy when the window is short. An expanding operator (prefix "Expanding") uses all data from the start of the stream, which gives a more stable estimate but never forgets early, possibly unrepresentative, observations. For stationary series where the distribution is unlikely to shift, expanding operators are often preferable because they use more information. For non-stationary series such as prices, regime-shifting volatility, or any signal expected to drift, a rolling window keeps estimates current.

## Align or combine streams

| Intent | Operator(s) | Note |
|:-------|:------------|:-----|
| Combine two or more streams, emitting on every new event with the last-known value for silent streams | `CombineLatest` | The main operator for merging asynchronous feeds. |
| Interleave several index-sorted streams into one timeline | `Merge` | Keeps every event, tagged with its source; each input must be individually index-sorted. |
| Resample a stream to a regular frequency or event count | `Resample` | Use `freq=` for time-based resampling or `count=` for event-count-based. |
| Drop rows that contain NaN from any input | `Dropna` | Useful after alignment when some inputs have not yet ticked. |
| Keep only rows that satisfy a boolean condition | `Filter` | Passes through rows where the condition stream is true. |
| Pick a subset of columns from a multi-column stream | `Select` | Projects a named or indexed subset of output columns. |

For full documentation of multi-stream operators, including the timestamp model and the causal alignment semantics, see the [Streams, values, and alignment](multistream) guide.

## Backtest a strategy

To evaluate a trading strategy on historical data, use one of screamer's backtest engines. The choice depends on the data feed you have (price, OHLC, trades, or L1) and the order type your strategy produces (target position or two-sided quotes). For the full decision grid see the [Choosing a backtest engine](functions_fin/choosing_a_backtest_engine) guide.

To check whether a backtest result is better than luck, `position_test` compares
the strategy's timing with randomly timed copies of the same trades and returns
a p-value. See [`position_test`](functions_fin/position_test).

<!-- HELP_END -->
