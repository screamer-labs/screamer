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
| `RollingMin` / `RollingMax` | `MIN` / `MAX` | -- | |
| `RollingArgmin` / `RollingArgmax` | `MININDEX` / `MAXINDEX` | -- | TA-Lib returns *absolute* sample index; we return *window offset*. The two are deterministically related: `talib_idx = our_offset + (t − w + 1)`. The test transforms one into the other. |
| `RollingMedian` | -- | `median` | |
| `HullMA` | -- | `hma` | TA-Lib does not have HullMA |
| `BollingerBands` middle band | `BBANDS` (mid) | `bbands` | |
| `RollingStd` | `STDDEV` (ddof=1) | -- | We follow pandas's `ddof=1` (sample std). See divergence below for `pandas-ta-classic.stdev`. |
| `RollingRSI` (default: Wilder) | `RSI` | `rsi` | Default smoothing is Wilder's, matching TA-Lib and pandas-ta. Pass `method="cutler"` for the SMA-smoothed variant; see divergence below. |
| `KAMA` | `KAMA` | `kama` | Kaufman's defaults (`fast=2`, `slow=30`) match TA-Lib. First valid output is at sample index `window_size`, seeded with `KAMA[n-1] = x[n-1]`. |

## Where we deliberately diverge

These are not bugs -- they are well-known convention splits in the technical-analysis ecosystem. Knowing which side a library sits on is part of using it correctly.

### `EwMean` (and therefore `DEMA`, `TEMA`) -- bias-corrected EMA vs Wilder/recursive EMA

`screamer.EwMean` uses pandas's default `adjust=True` form:

$$
\text{EwMean}[t] = \frac{\sum_{k=0}^{t} (1-\alpha)^{t-k} \cdot x_k}{\sum_{k=0}^{t} (1-\alpha)^{t-k}}
$$

This is the bias-corrected weighted mean (the denominator absorbs the missing past-infinity tail). It is well-defined from sample t=0 (returns `x[0]`).

TA-Lib's `EMA` uses the recursive form:

$$
\text{EMA}[t] = \alpha \cdot x[t] + (1-\alpha) \cdot \text{EMA}[t-1]
$$

with the seed `EMA[w-1] = SMA(x[0..w-1])` -- a window-sized SMA "warmup" plus undefined output for the first `w-1` samples. The two converge as `t → ∞` but disagree for early samples; for `span=10` the gap is on the order of a few percent for the first 30-50 samples.

`DEMA`, `TEMA`, and `MACD` are pure compositions of `EwMean`, so they inherit this divergence. Each of them matches the equivalent `pandas.Series.ewm(...).mean()` composition bit-exactly; each differs from TA-Lib by a few percent during early samples, converging as `t → ∞`. The decision is deliberate -- TA-Lib's SMA-seeded recursive form is a useful engineering shortcut but is not a statistically clean choice (it splices uniform weights onto exponential weights at an arbitrary cutoff). Our default favours the principled formula. If you need TA-Lib-bit-exact output for a backtest, file an issue.

### `RollingStd` -- ddof=1 (sample) vs ddof=0 (population)

`screamer.RollingStd` follows pandas's default of `ddof=1` (the unbiased sample estimator):

$$
\sigma^2 = \frac{1}{n - 1} \sum_i (x_i - \bar{x})^2
$$

`pandas-ta-classic.stdev` defaults to `ddof=0` (the maximum-likelihood / population estimator: dividing by `n` instead of `n - 1`). On a window of 10 samples the two differ by a factor of `sqrt(10/9) ≈ 1.054`, which is what the test asserts. Both are correct, just different conventions.

This divergence cascades into `BollingerBands`: the middle band (an SMA) matches TA-Lib exactly, but the upper/lower bands inherit the std difference (TA-Lib uses ddof=0 in `BBANDS` by default).

### `RollingRSI(method="cutler")` -- the opt-in alternative to Wilder

`screamer.RollingRSI` defaults to **Wilder's** smoothing (matching TA-Lib and pandas-ta-classic exactly). The constructor argument `method="cutler"` switches to **Cutler's RSI** -- a simple-moving-average smoothing of gains and losses:

$$
\text{avg\_gain}_W[t] = \frac{(w-1) \cdot \text{avg\_gain}_W[t-1] + \text{gain}[t]}{w}
\qquad\text{(Wilder, default)}
$$

$$
\text{avg\_gain}_C[t] = \frac{1}{w} \sum_{k=0}^{w-1} \text{gain}[t-k]
\qquad\text{(Cutler, opt-in)}
$$

The two methods disagree by up to ~10 RSI points during early periods and converge only slowly. In the wider literature both definitions are common; Wilder's is what almost every charting platform shows, Cutler's is what you get from a naive `pandas.Series.diff().rolling(w).mean()` decomposition and is used in some quantitative-research papers because the algebra is cleaner. There is also a one-sample offset: Wilder's first valid output is at sample index `n` (zero-indexed); Cutler's is at `n - 1`.

## How to verify these claims yourself

```bash
poetry install --with validation        # installs TA-Lib + pandas-ta-classic
poetry run pytest tests/test_third_party_alignment.py -v
```

If a library cannot be installed (TA-Lib needs its C dependency), the relevant tests skip with a clear message rather than fail.
