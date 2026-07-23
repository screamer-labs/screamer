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
| Adaptive moving average that slows during ranging markets | `KAMA` | Adjusts its speed to market volatility automatically. |
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

## Expanding (whole-history) vs rolling (trailing-window)

Many operators come in both flavors. A rolling operator sees only the most recent `window_size` samples, so its estimates adapt quickly but can be noisy when the window is short. An expanding operator (prefix "Expanding") uses all data from the start of the stream, which gives a more stable estimate but never forgets early, possibly unrepresentative, observations. For stationary series where the distribution is unlikely to shift, expanding operators are often preferable because they use more information. For non-stationary series such as prices, regime-shifting volatility, or any signal expected to drift, a rolling window keeps estimates current.

## Align or combine streams

| Intent | Operator(s) | Note |
|:-------|:------------|:-----|
| Combine two or more streams, emitting on every new event with the last-known value for silent streams | `CombineLatest` | The main operator for merging asynchronous feeds. |
| Concatenate or interleave streams in arrival order | `Merge` | Preserves all events from all inputs; output may be unsorted if feeds are not synchronized. |
| Resample a stream to a regular frequency or event count | `Resample` | Use `freq=` for time-based resampling or `count=` for event-count-based. |
| Drop rows that contain NaN from any input | `Dropna` | Useful after alignment when some inputs have not yet ticked. |
| Keep only rows that satisfy a boolean condition | `Filter` | Passes through rows where the condition stream is true. |
| Pick a subset of columns from a multi-column stream | `Select` | Projects a named or indexed subset of output columns. |

For full documentation of multi-stream operators, including the timestamp model and the causal alignment semantics, see the [Streams, values, and alignment](multistream) guide.

## Backtest a strategy

To evaluate a trading strategy on historical data, use one of screamer's backtest engines. The choice depends on the data feed you have (price, OHLC, trades, or L1) and the order type your strategy produces (target position or two-sided quotes). For the full decision grid see the [Choosing a backtest engine](functions_fin/choosing_a_backtest_engine) guide.

<!-- HELP_END -->
