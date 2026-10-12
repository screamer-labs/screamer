# Conventions and divergences from third-party libraries

For trading-decision use cases, knowing exactly which convention an indicator follows matters as much as the formula itself. Two libraries can both implement "RSI" or "DEMA" correctly and disagree by several percent because they made different defensible choices about smoothing, bias correction, or warmup seeding.

This page documents:

1. The indicators where `screamer` matches third-party libraries exactly (so you can swap freely).
2. The indicators where `screamer` *deliberately* diverges from a particular library, what the divergence is, and why.

All claims here are continuously checked by `tests/test_third_party_alignment.py`, which runs against TA-Lib and `pandas-ta-classic`. Run that file directly to see the live alignment numbers.

## Where we match (post-warmup, to ~1e-12)

| Indicator | TA-Lib | pandas-ta-classic | Notes |
|---|---|---|---|
| `WMA` | `WMA` | `wma` | linear weights, identical formula |
| `TRIMA` | `TRIMA` | `trima` | TA-Lib's asymmetric inner/outer split for even windows |
| `RollingMean` | `SMA` | `sma` | |
| `RollingMin` / `RollingMax` | `MIN` / `MAX` | - | |
| `RollingArgmin` / `RollingArgmax` | `MININDEX` / `MAXINDEX` | - | TA-Lib returns *absolute* sample index; we return *window offset*. The two are deterministically related: `talib_idx = our_offset + (t − w + 1)`. The test transforms one into the other. |
| `RollingMedian` | - | `median` | |
| `HullMA` | - | `hma` | TA-Lib does not have HullMA |
| `BollingerBands` middle band | `BBANDS` (mid) | `bbands` | |
| `RollingStd` | `STDDEV` (ddof=1) | - | We follow pandas's `ddof=1` (sample std). See divergence below for `pandas-ta-classic.stdev`. |
| `RollingRSI` (default: Wilder) | `RSI` | `rsi` | Default smoothing is Wilder's, matching TA-Lib and pandas-ta. Pass `method="cutler"` for the SMA-smoothed variant; see divergence below. |
| `KAMA` | `KAMA` | `kama` | Kaufman's defaults (`fast=2`, `slow=30`) match TA-Lib. First valid output is at sample index `window_size`, seeded with `KAMA[n-1] = x[n-1]`. |

## Where we deliberately diverge

These are not bugs - they are well-known convention splits in the technical-analysis ecosystem. Knowing which side a library sits on is part of using it correctly.

### `EwMean` (and therefore `DEMA`, `TEMA`) - bias-corrected EMA vs Wilder/recursive EMA

`screamer.EwMean` uses pandas's default `adjust=True` form:

$$
\text{EwMean}[t] = \frac{\sum_{k=0}^{t} (1-\alpha)^{t-k} \cdot x_k}{\sum_{k=0}^{t} (1-\alpha)^{t-k}}
$$

This is the bias-corrected weighted mean (the denominator absorbs the missing past-infinity tail). It is well-defined from sample t=0 (returns `x[0]`).

TA-Lib's `EMA` uses the recursive form:

$$
\text{EMA}[t] = \alpha \cdot x[t] + (1-\alpha) \cdot \text{EMA}[t-1]
$$

with the seed `EMA[w-1] = SMA(x[0..w-1])` - a window-sized SMA "warmup" plus undefined output for the first `w-1` samples. The two converge as `t → ∞` but disagree for early samples; for `span=10` the gap is on the order of a few percent for the first 30-50 samples.

`DEMA`, `TEMA`, and `MACD` are pure compositions of `EwMean`, so they inherit this divergence. Each of them matches the equivalent `pandas.Series.ewm(...).mean()` composition bit-exactly; each differs from TA-Lib by a few percent during early samples, converging as `t → ∞`. The decision is deliberate: TA-Lib's SMA-seeded recursive form splices uniform weights onto exponential weights at an arbitrary cutoff, so it agrees with the pure exponential form only as `t → ∞`. Our default uses the pure bias-corrected exponential weighting throughout. If you need TA-Lib-bit-exact output for a backtest, file an issue.

### `RollingStd` - ddof=1 (sample) vs ddof=0 (population)

`screamer.RollingStd` follows pandas's default of `ddof=1` (the unbiased sample estimator):

$$
\sigma^2 = \frac{1}{n - 1} \sum_i (x_i - \bar{x})^2
$$

`pandas-ta-classic.stdev` defaults to `ddof=0` (the maximum-likelihood / population estimator: dividing by `n` instead of `n - 1`). On a window of 10 samples the two differ by a factor of `sqrt(10/9) ≈ 1.054`, which is what the test asserts. Both are correct, just different conventions.

This divergence cascades into `BollingerBands`: the middle band (an SMA) matches TA-Lib exactly, but the upper/lower bands inherit the std difference (TA-Lib uses ddof=0 in `BBANDS` by default).

### `RollingRSI(method="cutler")` - the opt-in alternative to Wilder

`screamer.RollingRSI` defaults to **Wilder's** smoothing (matching TA-Lib and pandas-ta-classic exactly). The constructor argument `method="cutler"` switches to **Cutler's RSI** - a simple-moving-average smoothing of gains and losses:

$$
\text{avg\_gain}_W[t] = \frac{(w-1) \cdot \text{avg\_gain}_W[t-1] + \text{gain}[t]}{w}
\qquad\text{(Wilder, default)}
$$

$$
\text{avg\_gain}_C[t] = \frac{1}{w} \sum_{k=0}^{w-1} \text{gain}[t-k]
\qquad\text{(Cutler, opt-in)}
$$

The two methods disagree by up to ~10 RSI points during early periods and converge only slowly. In the wider literature both definitions are common; Wilder's is what almost every charting platform shows, Cutler's is what you get from a naive `pandas.Series.diff().rolling(w).mean()` decomposition and is used in some quantitative-research papers because it is a direct rolling-mean decomposition. There is also a one-sample offset: Wilder's first valid output is at sample index `n` (zero-indexed); Cutler's is at `n - 1`.

## Accuracy at large levels

The moment operators compute their statistics from sums of powers of the samples. Taken about zero, these sums cancel when the mean is large next to the spread: for a price near 1000 that moves by about 1 per step, the fourth central moment keeps about 4 of its 16 significant digits, and a kurtosis computed from such sums can be wrong in its second digit.

screamer takes the sums about an origin close to the data instead. The estimator is the same, but every term is of the size of the spread, so nothing cancels:

- The rolling operators start the origin at the first sample. They move it to the window mean, recomputing the sums from the window, when the mean has drifted from the origin by more than four standard deviations (after a level change), when the sample that leaves the window was far out next to the samples that remain (a spike, or the last sample of an old level), when an infinity leaves, and in any case once every `window_size` steps.
- The expanding and exponentially weighted operators also start the origin at the first sample. It moves to the current mean when the mean has drifted from it by more than one standard deviation (three for the two-input weighted operators, which use only second moments), checked every fourth sample.

`tests/test_moment_stability.py` checks each operator against its estimator evaluated in exact rational arithmetic, at levels 0, 1e3 and 1e6, on noise, on a drifting walk, and for the rolling operators with a spike of 1e4 standard deviations and a level jump of 1e5 standard deviations (both directions, windows 50 and 1000). The largest error measured is 3e-13 relative to the statistic's scale (a kurtosis right after a level jump, window 1000); most errors are below 1e-14. On noise the result is the same at every level. An output that is itself of the size of the level, such as a mean, a band or a spread `x - beta * y`, keeps the rounding of that level.

On stationary data a rolling operator recomputes its sums about once per window, so its cost per step is constant on average; outliers that leave while others remain in the window do not dominate it and cost nothing extra, but the step that recomputes takes one pass over the window (two after a far sample left). With `window_size=10000` a streaming step usually takes a few nanoseconds; the step that recomputes takes about 8 microseconds, and up to about 23 microseconds after a spike or a level jump (Apple M3).

The operators that work this way are `RollingVar`, `RollingStd`, `RollingSkew`, `RollingKurt`, `RollingZscore`, `RollingSharpe`, `BollingerBands`, `RollingSigmaClip`, `RollingCov`, `RollingCorr`, `RollingBeta`, `RollingSpread`, `RollingAlpha`, `RollingTLSBeta`, `RollingLinearRegression`, `RollingResidualStd`, `ExpandingVar`, `ExpandingStd`, `ExpandingSkew`, `ExpandingKurt`, `EwVar`, `EwStd`, `EwSkew`, `EwKurt`, `EwZscore`, `EwCov`, `EwCorr`, `EwBeta`, `EwSpread`, `EwAlpha`, `EwResidualStd`, `EwYangZhangVar` and `EwYangZhangVol`.

## How to verify these claims yourself

```bash
poetry install --with validation        # installs TA-Lib + pandas-ta-classic
poetry run pytest tests/test_third_party_alignment.py -v
```

If a library cannot be installed (TA-Lib needs its C dependency), the relevant tests skip with a clear message rather than fail.
