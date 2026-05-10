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

`DEMA` and `TEMA` are pure compositions of `EwMean`, so they inherit this divergence. `screamer.DEMA(span=10)(x)` matches pandas-ta-classic's `dema` (which also uses recursive EMA only when configured to) only via the `adjust=True` path; against TA-Lib's `DEMA` we differ by up to ~2% during warmup.

**If you need TA-Lib parity for DEMA/TEMA**, file an issue -- it is implementable as an `adjust=False` mode on `EwMean`, just not currently exposed.

### `RollingStd` -- ddof=1 (sample) vs ddof=0 (population)

`screamer.RollingStd` follows pandas's default of `ddof=1` (the unbiased sample estimator):

$$
\sigma^2 = \frac{1}{n - 1} \sum_i (x_i - \bar{x})^2
$$

`pandas-ta-classic.stdev` defaults to `ddof=0` (the maximum-likelihood / population estimator: dividing by `n` instead of `n - 1`). On a window of 10 samples the two differ by a factor of `sqrt(10/9) ≈ 1.054`, which is what the test asserts. Both are correct, just different conventions.

This divergence cascades into `BollingerBands`: the middle band (an SMA) matches TA-Lib exactly, but the upper/lower bands inherit the std difference (TA-Lib uses ddof=0 in `BBANDS` by default).

### `RollingRSI` -- Cutler's vs Wilder's

`screamer.RollingRSI` uses simple-moving-average smoothing of gains and losses (sometimes called "Cutler's RSI"). TA-Lib's `RSI` uses Wilder's smoothing -- a recursive form roughly equivalent to an EMA with `α = 1/period`:

$$
\text{avg\_gain}_W[t] = \frac{(w-1) \cdot \text{avg\_gain}_W[t-1] + \text{gain}[t]}{w}
$$

The two methods disagree by up to ~10 RSI points during early periods and converge slowly. In the wider literature both definitions are common; Wilder's is more popular in TA-Lib-compatible workflows, Cutler's is what you get from a naive `pandas.Series.diff().rolling(w).mean()` decomposition.

## How to verify these claims yourself

```bash
poetry install --with validation        # installs TA-Lib + pandas-ta-classic
poetry run pytest tests/test_third_party_alignment.py -v
```

If a library cannot be installed (TA-Lib needs its C dependency), the relevant tests skip with a clear message rather than fail.
