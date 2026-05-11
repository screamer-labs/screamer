# Regression and statistical position metrics

Six classes covering rolling-window linear regression plus pandas-style position metrics. The 2-input regression family follows `RollingBeta`'s convention: **first argument is the dependent (target), second is the regressor**.

## `RollingAlpha(window_size, start_policy="strict")` -- regression intercept

$$
\alpha[t] = \overline{y}_w - \beta[t] \cdot \overline{x}_w
$$

Companion to `RollingBeta`. Same `(target, regressor)` input convention. Composes `RollingBeta` + two `RollingMean` instances. O(1) per step.

## `RollingResidualStd(window_size, start_policy="strict")` -- spread std

$$
\sigma_\text{spread}[t] = \text{RollingStd}\big(\text{RollingSpread}(y, x)\big)[t]
$$

Standard deviation of the per-bar hedge-adjusted spread `y − β·x` over the trailing window (ddof=1). Useful for pairs-trading z-score normalisation: `z = (current_spread - mean_spread) / RollingResidualStd`.

Composes `RollingSpread` + `RollingStd`. O(1) per step. NaN-poisoning during `RollingSpread`'s own warmup is gated explicitly so the std accumulator stays clean.

## `RollingLinearRegression(window_size, start_policy="strict")` -- 2→4 OLS fit

$$
y \approx \text{slope} \cdot x + \text{intercept} + \varepsilon
$$

Returns the four-tuple **(slope, intercept, r_squared, stderr)** per step. Stat definitions:

| Output | Formula |
|---|---|
| `slope` | $(n S_{xy} - S_x S_y) / (n S_{xx} - S_x^2)$ |
| `intercept` | $(S_y - \text{slope} \cdot S_x) / n$ |
| `r_squared` | $(n S_{xy} - S_x S_y)^2 / [(n S_{xx} - S_x^2)(n S_{yy} - S_y^2)]$ |
| `stderr` | $\sqrt{\text{SSE} / (n-2)}$ -- **RMSE of residuals** ("standard error of estimate") |

**Note on `stderr`**: this is the standard error of the *estimate* (the RMSE of the fit), not the standard error of the *slope coefficient*. The two are related by `slope_stderr = stderr / sqrt(Σ(x − mean_x)²)` -- multiply by that scale factor if you need slope confidence.

Output is 2→4 (`FunctorBase<_, 2, 4>`). First valid output at sample index `window_size - 1`. The first 2→4 consumer of the Plan E `N→M` dispatcher in screamer.

## `RollingTSF(window_size)` -- Time-Series Forecast

$$
\text{TSF}[t] = \text{linear regression of } y \text{ on time, projected one step ahead}
$$

Equivalent to fitting `y = slope * t + intercept` on the trailing window {(0, y_{t-w+1}), …, (w-1, y_t)} and returning the line evaluated at the *next* bar (local t = w).

1→1. Bit-exact match to `talib.TSF(real, timeperiod)`. Composes two `detail::RollingSum` buffers + precomputed time-axis constants. O(1) per step.

## `RollingRank(window_size)` and `RollingPercentile(window_size)` -- position metrics

Where does the current value sit within the trailing window?

$$
\text{rank}[t] = (\text{#values} < y_t) + 1 + \tfrac{1}{2}(\text{#ties} - 1)
$$

Pandas's "average" tie-breaking rule (mean rank among tied values). `RollingRank` returns a 1-based rank in `[1, w]`; `RollingPercentile` returns `rank / w` in `[1/w, 1]` (matching `pandas.Series.rolling(w).rank(pct=True)`).

Both are 1→1. Circular window buffer + per-step counting sweep; O(W) per step.

## Usage Example

```python
import numpy as np
from screamer import (
    RollingAlpha, RollingResidualStd, RollingLinearRegression,
    RollingTSF, RollingRank, RollingPercentile,
)

rng = np.random.default_rng(0)
asset = rng.standard_normal(500)
market = 0.5 * asset + 0.3 * rng.standard_normal(500)

# Pairs-trading z-score: (spread - mean_spread) / std_spread
alpha = RollingAlpha(60)(asset, market)
sigma = RollingResidualStd(60)(asset, market)
# ...

# Full regression stats
lr = RollingLinearRegression(60)(asset, market)
slope, intercept, r2, rmse = lr[:, 0], lr[:, 1], lr[:, 2], lr[:, 3]

# One-step-ahead forecast on the asset series itself
forecast = RollingTSF(20)(asset)

# Where does the latest return sit in its recent distribution?
pct = RollingPercentile(60)(asset)  # in [1/60, 1]
```

## Reference

| Class | Reference | Status |
|---|---|---|
| `RollingAlpha` | manual `mean(y) − β·mean(x)` | bit-exact (≤ 1e-12) |
| `RollingResidualStd` | `RollingStd(RollingSpread)` composition | bit-exact (≤ 1e-12) |
| `RollingLinearRegression` slope / intercept / r² | `scipy.stats.linregress` | bit-exact (≤ 1e-10) |
| `RollingLinearRegression` stderr | manual `sqrt(SSE / (n-2))` | bit-exact |
| `RollingTSF` | `talib.TSF` | bit-exact (≤ 1e-14) |
| `RollingRank` | `pandas.Series.rolling(w).rank()` | bit-exact (0.0) |
| `RollingPercentile` | `pandas.Series.rolling(w).rank(pct=True)` | bit-exact (0.0) |
