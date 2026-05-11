# `TRIX`

## Description

`TRIX(span)` is the one-step rate of change of a **triple-smoothed** EMA -- a momentum oscillator that filters short-term noise via two extra layers of exponential smoothing before measuring change.

$$
\begin{aligned}
\text{ema}_1[t] &= \text{EwMean}(\text{span})(x)[t] \\
\text{ema}_2[t] &= \text{EwMean}(\text{span})(\text{ema}_1)[t] \\
\text{ema}_3[t] &= \text{EwMean}(\text{span})(\text{ema}_2)[t] \\
\text{TRIX}[t]  &= 100 \cdot \frac{\text{ema}_3[t] - \text{ema}_3[t-1]}{\text{ema}_3[t-1]}
\end{aligned}
$$

*Parameters*:

- `span` (int, positive): the span for every EMA stage (all three use the same value).

*Warmup*: only the first sample is NaN (no `prev_ema3` yet). Each `EwMean` is well-defined from `t=0`, so from `t=1` onward TRIX is finite.

*NaN handling*: NaN inputs propagate through the EMA arithmetic.

## Implementation Details

Pure composition of three chained `EwMean` instances plus a single scalar holding `prev_ema3`. O(1) per step.

The underlying EMA is pandas's `adjust=True` (bias-corrected weighted mean -- the form we use throughout the library). TA-Lib's TRIX uses `adjust=False` with an SMA-seeded warmup, so ours differs from TA-Lib by a few percent during early samples; see [conventions](../conventions.md).

## Usage Example

```python
import numpy as np, pandas as pd
from screamer import TRIX

x = 100 + np.cumsum(np.random.default_rng(0).standard_normal(200))
trix = TRIX(14)(x)

# Algorithmically equivalent composition (pandas adjust=True):
s = pd.Series(x)
e3 = s.ewm(span=14, adjust=True).mean().ewm(span=14, adjust=True).mean().ewm(span=14, adjust=True).mean()
np.testing.assert_allclose(trix, 100 * e3.pct_change(), equal_nan=True, atol=1e-12)
```

## Reference

Equivalent to `pandas.Series.ewm(span=..., adjust=True).mean()` triple composition followed by `100 * pct_change`. Cross-validated in `tests/test_third_party_alignment.py`: bit-exact vs pandas composition; documented divergence vs TA-Lib's `TRIX`.
