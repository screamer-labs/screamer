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
| Typical value or spread of each hour of the day | `RollingSeasonalMean`, `RollingSeasonalStd`, `RollingSeasonalZscore` | The same slot of the last `window_size` cycles; the input must be on a regular clock. |
| Variance without the jumps | `RollingBipowerVar` | Bipower variation of returns, on the scale of the realized variance. |
| Average true range, for stops and position sizing | `TrueRange`, `ATR`, `NATR` | The true range of each bar, its Wilder average, and that average as a percentage of the close. |

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
| Hedge ratio that drifts over time, without a window | `KalmanRegression` | Kalman filter on a random-walk intercept and slope; also returns the spread and its predicted deviation. |
| Hedge ratio that treats both series alike | `RollingTLSBeta` | Total least squares: the ratio does not depend on which series is the dependent one. |

## Trade a pair

| Intent | Operator(s) | Note |
|:-------|:------------|:-----|
| Estimate the hedge ratio | `KalmanRegression`, `RollingTLSBeta`, `RollingBeta` | Kalman adapts without a window; TLS is symmetric; OLS is the plain baseline. |
| Test that the spread still reverts | `RollingADF` | Augmented Dickey-Fuller statistic over a window; below about -2.86 rejects a unit root at 5%. |
| Measure how fast it reverts | `RollingOU` | Ornstein-Uhlenbeck fit: mean-reversion rate, mean, and deviation. |
| Standardize the spread | `RollingZscore`, `EwZscore` | The input of a band rule. |
| Enter and exit on bands | `BandSignal` | Short above `+entry`, long below `-entry`, flat inside `exit`, with an optional stop. |
| Backtest both legs from one account | `BacktestPortfolioOHLCTarget`, `BacktestPortfolioTarget` | Weights of equity per leg, with per-asset costs. |

The [pairs trade recipe](recipes/pairs-trade.md) puts these together on BTC and
ETH, and [notebook 18](notebooks/18-pairs-and-mean-reversion) compares the hedge
ratio estimators.

## Rank and weight assets

| Intent | Operator(s) | Note |
|:-------|:------------|:-----|
| Rank each asset against the others at each step | `CrossRank` | Scaled to [-1, 1] or [0, 1]. |
| Standardize or center scores across assets | `CrossZscore`, `CrossDemean` | Z-score or mean removal over the assets of one step. |
| Measure the spread of values across assets | `CrossStd` | One value per step, such as the return dispersion. |
| Limit outliers across assets | `CrossWinsorize` | Clips each row to its lower and upper quantiles. |
| Long the top, short the bottom | `CrossTopBottom` | +1 for the `count` highest scores, -1 for the lowest. |
| Scale scores into portfolio weights | `CrossNormalize` | Gross exposure 1 by default, with an optional net exposure. |
| Backtest the book | `BacktestPortfolioOHLCTarget`, `PortfolioReport` | One shared account, rebalancing rules, exposure limits, and book statistics. |

The [cross-sectional momentum recipe](recipes/cross-sectional-momentum.md) and
the [portfolios and pairs guide](portfolio_backtesting.md) show the whole path.

## Turn a signal into a position

| Intent | Operator(s) | Note |
|:-------|:------------|:-----|
| Long or short on the sign of a score | `Sign` | +1, 0, or -1 per step. |
| Enter and exit at different levels | `BandSignal`, `SchmittTrigger` | Hysteresis stops a score near a threshold from switching the position on and off. |
| Hold a signal for a fixed number of bars | `Hold` | Latches a nonzero input and keeps it for `n` bars. |
| Hold a signal, then ignore new ones for a while | `Hold(cooldown=)` | After a hold expires, triggers in the next cooldown steps are ignored. |
| Size by volatility | `RollingStd`, `EwStd`, `ATR` | Divide a risk budget by the volatility; see the [volatility sizing recipe](recipes/volatility-sizing.md). |

## Mean reversion and forecasting

| Intent | Operator(s) | Note |
|:-------|:------------|:-----|
| Ornstein-Uhlenbeck mean-reversion parameters from a spread | `RollingOU` | Fits OU to a trailing window; `output=` selects "mrr"/"mean"/"relmean"/"std". |
| Online Bayesian regression with calibrated prediction intervals | `BayesianRegression` | Returns slope, intercept, predictive mean, and predictive std per sample; interval widens early and tightens as evidence accumulates. See [notebook 17](notebooks/17-filtering-and-forecasting-with-uncertainty). |
| Kalman filter as a recursive forecaster | `KalmanFilter` | Tracks a latent level and returns the filtered estimate; tune `process_var` and `observation_var` to set the lag-vs-noise trade-off. |

## Detect cycles

| Intent | Operator(s) | Note |
|:-------|:------------|:-----|
| Measure the dominant cycle period | `DominantCycle` | Ehlers' homodyne discriminator; the period in samples. |
| Instantaneous phase, frequency, or amplitude of a cycle | `CyclePhase`, `CycleFrequency`, `CycleAmplitude` | From Ehlers' Hilbert transform; `HilbertPhasor` gives its in-phase and quadrature parts. |
| Sinewave turning-point indicator | `CycleSine` | Sine and lead-sine lines; a crossing marks a cycle turn. |
| Classify trending versus cycling | `TrendMode` | Ehlers' trend mode for price series: 1.0 while trending, 0.0 while cycling. |
| Isolate a band of cycle periods | `RoofingFilter` | A highpass then a `SuperSmoother`; feed its output to the cycle operators. |
| Low-lag smoothing or trend | `SuperSmoother`, `Decycler`, `InstantaneousTrendline` | A 2-pole lowpass, a trend with short cycles removed, and the mean over one measured cycle. |
| Adaptive moving average pair | `MAMA` | MAMA and FAMA lines whose smoothing follows the phase rate. |
| Turn a noisy indicator into a stable on/off state | `SchmittTrigger` | Hysteresis between two thresholds avoids flip-flopping. |

For a worked example see [notebook 19](notebooks/19-cycle-and-spectral-analysis).

## Detect regimes and change points

| Intent | Operator(s) | Note |
|:-------|:------------|:-----|
| Probability of a volatile, or of a trending, regime | `RollingHMM` | Two-state hidden Markov model refit on a trailing window; `prob` is the filtered probability of the high-variance state, or with `emission="ar1"` of the more persistent state. |
| Detect that the mean or spread of a series changed | `BayesianChangepoint` | Expected run length since the last change, and a predictive mean and standard deviation; feed a standardized series. |
| Sample events at cumulative moves | `CusumFilter` | +1 or -1 when the move since the last event passes `threshold`; label or trade at the events. |
| Test whether returns trend or revert | `RollingVarianceRatio`, `RollingHurst` | A variance ratio above 1 trends, below 1 reverts; `output="zscore"` gives the robust test statistic. |
| Measure how random a series is | `RollingPermutationEntropy` | 1 for noise, lower when some orderings of consecutive values are more common. |
| Share of the variance from jumps | `RollingJumpShare`, `RollingBipowerVar` | Bipower variation estimates the variance without the jumps. |

## Analyse order flow and liquidity

| Intent | Operator(s) | Note |
|:-------|:------------|:-----|
| Sign trades as buyer- or seller-initiated | `TickRuleSign`, `LeeReadySign`, `BulkVolumeClassifier` | Tick rule from prices only; Lee-Ready with quotes; bulk classification from bar returns. |
| Signed volume and order-flow imbalance | `SignedVolume`, `OFI`, `RollingOrderImbalance` | Net aggressive flow per trade, normalized, or summed over a window. |
| Book imbalance and fair value | `QueueImbalance`, `MicroPrice`, `ContOFI` | From L1 quotes: size imbalance, an imbalance-weighted mid, and order-flow imbalance from book events. |
| Trading cost | `EffectiveSpread`, `RealizedSpread`, `RollSpread` | Cost against the mid, its liquidity part, and a spread estimate from trade prices alone. |
| Spread from OHLC bars | `CorwinSchultzSpread`, `AbdiRanaldoSpread` | Bid-ask spread estimated from bar high and low (Corwin-Schultz) or close, high and low (Abdi-Ranaldo). |
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
| Spread of the gains above a threshold | `RollingUpsideDeviation` | The mirror of `RollingDownsideDeviation`. |
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
| Build a supervised training set | `forecast_pairs` | Pairs the features at each step with the target over the horizon after it; see [Build a training set](#build-a-training-set). |

## Build a training set

A training row holds the features at step `t` and the target over the steps
after it. The target is a causal series, such as `LogReturn(h)` of the price,
and `forecast_pairs(X, y, count=h)` pairs it with the features where its
horizon starts.

| Intent | Operator(s) | Note |
|:-------|:------------|:-----|
| Pair features with a target | `forecast_pairs` | `count=` for a horizon in events, `duration=` for one in index units. |
| Label which of two price levels is reached first | `TripleBarrier`, `TripleBarrierOHLC` | `+1`, `-1`, or `0` per step, with the return and the holding time. |
| A return, direction, volatility, or range target | `LogReturn`, `Sign`, `RollingStd`, `RollingMax`, `RollingMin` | See the chooser in [Labels and training sets](labels.md). |

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

| Intent | Operator(s) | Note |
|:-------|:------------|:-----|
| One asset, target position, bar data | `BacktestOHLCTarget`, `BacktestPriceTarget` | Fills at the next open, or at the price you pass. |
| Many assets from one account | `BacktestPortfolioOHLCTarget`, `BacktestPortfolioTarget` | Weights, units, or notional per asset, with rebalancing rules. |
| Funding, borrow, or financing | `BacktestCarry` | Charges carry on the position of any engine's output; see the [funding recipe](recipes/funding-carry.md). |
| A list of the trades, or results per month | `backtest_trades`, `backtest_report(by=)` | One row per round trip; per-group totals, Sharpe, and drawdown. |
| Summary statistics | `backtest_report`, `BacktestStats`, `PortfolioReport` | `backtest_report` gives a running table and a summary in one call; `BacktestStats` adds round-trip trade statistics as a node; `PortfolioReport` reports a portfolio book. |

## Is it more than luck?

| Intent | Operator(s) | Note |
|:-------|:------------|:-----|
| Compare a strategy with randomly timed copies of its trades | `position_test` | Shifts the positions in time and returns a p-value; works on portfolios too. |
| Compare it with the same rule on randomized markets | `strategy_test` | Reruns the strategy on permuted or shuffled prices. |
| Correct for the number of variants tried | `deflated_sharpe`, `probabilistic_sharpe` | The Sharpe ratio a best-of-many search would reach by luck. |
| Choose parameters out of sample | `walk_forward` | Picks the best variant on each training window and trades it on the next test window. |
| Select between variants with funding, or between portfolio weightings | `walk_forward(carry=)`, `walk_forward` with a portfolio engine | Every run is scored after `BacktestCarry`; a (T, K, C) target is scored on the book. |

See the [significance recipe](recipes/significance-test.md), the
[walk-forward recipe](recipes/walk-forward.md), and
[backtesting best practices](backtesting_best_practices.md).

<!-- HELP_END -->
