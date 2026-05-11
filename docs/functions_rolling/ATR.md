# `TrueRange`, `ATR`, `NATR`

## Description

This is the Wilder family of OHLC-aware volatility -- the dominant tradition in technical analysis for measuring "average bar size", complementary to the statistical range-based estimators (Parkinson, Garman-Klass, Rogers-Satchell).

### TrueRange

The per-bar quantity that accounts for overnight gaps:

$$
\text{TR}[t] = \max\big(\ H - L,\ \ |H - C_{t-1}|,\ \ |L - C_{t-1}|\ \big)
$$

`TrueRange` is a 3-input, 1-output functor on `(high, low, close)`. The first sample returns `NaN` (no previous close). Otherwise stateless.

### ATR -- Average True Range (Wilder, 1978)

Wilder-smoothed rolling average of `TR`:

$$
\begin{aligned}
\text{ATR}[w] &= \frac{1}{w} \sum_{i=1}^{w} \text{TR}[i] \quad\text{(SMA seed)} \\
\text{ATR}[t] &= \frac{(w - 1) \cdot \text{ATR}[t - 1] + \text{TR}[t]}{w} \quad\text{for } t > w
\end{aligned}
$$

`ATR(window_size=14)` is the conventional setting (Wilder's original choice). First valid output at sample index `window_size`. Matches `talib.ATR` bit-exactly post-warmup.

### NATR -- Normalised ATR

`ATR` scaled to a percentage of the current close:

$$
\text{NATR}[t] = 100 \cdot \frac{\text{ATR}[t]}{C[t]}
$$

Useful for cross-instrument comparison (an ATR of \$1 on a \$10 stock is much larger than the same ATR on a \$1000 stock).

## Output shape

All three are `FunctorBase<_, 3, 1>` on `(high, low, close)`:

| You pass... | You get back... |
|---|---|
| three scalars | `float` |
| three 1D arrays of shape `(T,)` | array of shape `(T,)` |
| three 2D arrays of shape `(T, K)` | array of shape `(T, K)` |

## Implementation Details

`TrueRange` holds a single scalar `prev_close`. `ATR` adds an internal accumulator for the SMA seed (used only during warmup) and the recursive Wilder state. `NATR` simply composes `ATR` and divides. All three are O(1) per step.

## Usage Example

```python
import numpy as np
from screamer import ATR

rng = np.random.default_rng(0)
n = 200
close = 100 + np.cumsum(rng.normal(0, 1, n))
high = close + np.abs(rng.normal(0, 0.5, n))
low = close - np.abs(rng.normal(0, 0.5, n))

atr = ATR(14)(high, low, close)
# NaN for indices 0..13; valid from 14 onward.
```

## Relation to the range-based estimators

| Family | Convention | Smoothing |
|---|---|---|
| `Parkinson`, `GarmanKlass`, `RogersSatchell` | per-bar variance estimator (squared, ln-based) | rolling **mean** or EW mean |
| `TrueRange`, `ATR`, `NATR` | per-bar range (linear, max-based) | **Wilder** smoothing (EMA with alpha = 1/n) |

They measure related but not identical quantities. ATR is a price-scaled "typical bar size"; the range-based estimators are estimators of *return* variance. Both have their place in trading systems.

## Reference

Bit-exact matches to `talib.TRANGE`, `talib.ATR`, `talib.NATR` post-warmup (verified to floating-point precision in `tests/test_third_party_alignment.py`).
