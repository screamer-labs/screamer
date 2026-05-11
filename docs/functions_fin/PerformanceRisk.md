# Performance and risk metrics

Eight streaming primitives for backtest evaluation: drawdowns plus the canonical risk-adjusted-return ratios. None of these are in TA-Lib (which is purely indicator-focused), so they are a genuine differentiator -- you can compute them per-bar in the same pipeline that produces your signals, without spinning up pandas-pyfolio-quantstats.

## Drawdown family

### `Drawdown()` -- running drawdown from cumulative peak

$$
\text{Drawdown}[t] = \frac{\text{price}[t]}{\text{CumMax}(\text{price})[t]} - 1
$$

A flat or new-high series gives `0`. A 30% loss from the prior peak gives `-0.30`. **1-input, 1-output.** Composes `CumMax`. No warmup. Bit-exact to `pandas.Series.cummax`-based reference.

### `MaxDrawdown()` -- largest drawdown so far

$$
\text{MaxDrawdown}[t] = \min_{k \le t}\ \text{Drawdown}[k]
$$

The worst peak-to-trough loss experienced since reset. **1-input, 1-output.** Composes `Drawdown` + `CumMin`. Monotonically non-increasing in time.

### `RollingMaxDrawdown(window_size)` -- worst drawdown within the trailing window

The maximum peak-to-trough loss observed inside the last `w` bars. Different from `MaxDrawdown` (which is the worst loss EVER since reset).

**1-input, 1-output.** Algorithm: maintain a circular buffer of the last `w` prices; per step, sweep the buffer tracking a within-window running peak and the worst drawdown from that peak. **O(window_size) per step** -- there's no cheap-O(1)-amortised algorithm for the standard definition (the rolling peak can be anywhere in the window).

If you want the cheaper "current drawdown vs. rolling-window peak" approximation, compose it directly:

```python
rolling_dd = price / RollingMax(window)(price) - 1
```

## Risk-adjusted-return ratios

All take a *returns* series. The `periods_per_year` parameter is the annualisation factor (`252` for daily, `52` for weekly, `12` for monthly, `1` for no annualisation).

### `RollingSharpe(window_size, periods_per_year=1.0)`

$$
\text{Sharpe}[t] = \sqrt{ppy}\ \cdot\ \frac{\text{RollingMean}(r)}{\text{RollingStd}(r)}
$$

Composes `RollingMean` + `RollingStd` (ddof=1, pandas default). Returns NaN where the std is zero.

### `RollingSortino(window_size, periods_per_year=1.0, target=0.0)`

$$
\text{Sortino}[t] = \sqrt{ppy}\ \cdot\ \frac{\text{mean}(r) - \text{target}}{\sqrt{\text{mean}(\min(r - \text{target},\ 0)^2)}}
$$

Same as Sharpe but the denominator is the *downside* deviation -- only the bars below `target` contribute, so upside variability is not penalised. O(W) per step (the downside-RMS sweep is necessary; there is no closed-form O(1) update for the squared-piecewise-min).

### `RollingInfoRatio(window_size, periods_per_year=1.0)`

$$
\text{IR}[t] = \sqrt{ppy}\ \cdot\ \frac{\text{mean}(r - b)}{\text{std}(r - b)}
$$

Information Ratio against a benchmark. **2-input, 1-output** on `(returns, benchmark)`. Effectively `RollingSharpe` applied to the active-return series.

### `RollingCalmar(window_size, periods_per_year=1.0)`

$$
\text{Calmar}[t] = \frac{ppy \cdot \text{mean}(r)}{\big|\text{RollingMaxDrawdown}(\text{implied price})\big|}
$$

Calmar Ratio: annualised return divided by the worst rolling drawdown. Takes a *returns* series; internally reconstructs the implied price path as a cumulative product `price *= (1 + r)` (starting from 1.0) so the drawdown calculation is well-defined. Returns NaN when there is no drawdown in the window (i.e. the path is monotonic up).

If you already have a price path and want Calmar over it, compose by hand:

```python
calmar = ppy * RollingMean(window)(returns) / abs(RollingMaxDrawdown(window)(price))
```

### `RollingHitRate(window_size)`

$$
\text{HitRate}[t] = \frac{1}{w}\ \text{count}(r_i > 0,\ i \in \text{window})
$$

Fraction of strictly-positive samples in the trailing window. Output in `[0, 1]`. Composes `detail::RollingSum` over the indicator `(r > 0)`.

## Usage Example

```python
import numpy as np
from screamer import (
    Drawdown, MaxDrawdown, RollingMaxDrawdown,
    RollingSharpe, RollingSortino, RollingCalmar, RollingHitRate,
)

# Synthetic daily strategy returns.
rng = np.random.default_rng(0)
returns = rng.normal(0.0008, 0.015, 1000)
price = 100 * np.cumprod(1 + returns)

mdd_curve = MaxDrawdown()(price)
rolling_mdd = RollingMaxDrawdown(60)(price)
sharpe = RollingSharpe(60, periods_per_year=252)(returns)
sortino = RollingSortino(60, periods_per_year=252)(returns)
calmar = RollingCalmar(60, periods_per_year=252)(returns)
hit_rate = RollingHitRate(60)(returns)
```

## Reference

There is no canonical TA-Lib counterpart for any of these (they are not technical indicators). The classes are validated against hand-rolled pandas references to 1e-12 to 1e-14 across the suite. See `tests/test_performance_risk.py`.
