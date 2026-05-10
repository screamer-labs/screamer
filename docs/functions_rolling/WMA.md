# `WMA`

## Description

`WMA` computes the **linearly-weighted moving average**: the most recent sample carries weight `w`, the next-most-recent weight `w-1`, ..., the oldest in the window weight `1`. The denominator is the triangular number `w*(w+1)/2`.

$$
\text{WMA}[t] = \frac{1 \cdot x[t-w+1] \;+\; 2 \cdot x[t-w+2] \;+\; \dots \;+\; w \cdot x[t]}{w \, (w+1) / 2}
$$

WMA is one of the three classical moving averages alongside `RollingMean` (SMA) and `EwMean` (EMA), and is a common preprocessing step in technical analysis (Hull, KAMA, etc. are built on top of it).

*Parameters*:

- `window_size` (int, positive).
- `start_policy` (str, optional): `"strict"` (default), `"expanding"`, or `"zero"`. See **Warmup behaviour** below.

*NaN handling*: NaN values should be preprocessed; an NaN input poisons subsequent outputs through the rolling sum.

## Implementation Details

### Why O(1) per step?

WMA admits a closed-form rolling recurrence. Let `W[t] = 1·x[t-w+1] + ... + w·x[t]` be the linear-weighted sum, and `S[t-1]` the simple rolling sum of the *previous* window (i.e. `x[t-1] + x[t-2] + ... + x[t-w]`). Then

$$
W[t] - W[t-1] \;=\; w \cdot x[t] \;-\; S[t-1]
$$

(every old weight drops by 1, contributing `−S[t-1]`; the new sample enters with weight `w`). The class therefore holds a `detail::RollingSum` (for `S`) and a single `double` (for `W`).

### Complexity

* Time complexity: `O(1)` per step.
* Space complexity: `O(window_size)` for the rolling-sum buffer.

### Warmup behaviour

While the window is filling (n samples seen, `n < window_size`):

| Policy | Output during warmup |
|---|---|
| `"strict"` (default) | `NaN` |
| `"expanding"` | partial-window WMA: weights `1, 2, ..., n`, divisor `n(n+1)/2` |
| `"zero"` | weights `1, 2, ..., w` with implicit zeros for missing past, full divisor `w(w+1)/2` (output is dampened during warmup) |

The warmup numerators agree exactly at the moment the window first fills, so the transition to the post-warmup recurrence is seamless.

## Usage Example

```python
import numpy as np
import pandas as pd
from screamer import WMA

rng = np.random.default_rng(0)
x = rng.standard_normal(100)
w = 10

ours = WMA(w)(x)

# Reference: explicit per-window dot product (same definition pandas uses)
ref = pd.Series(x).rolling(w).apply(
    lambda v: np.dot(v, np.arange(1, w + 1)) / (w * (w + 1) / 2),
    raw=True,
).to_numpy()
np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-12)
```

## Reference

Equivalent to `pandas.Series.rolling(w).apply(lambda v: np.dot(v, np.arange(1, w+1)) / (w*(w+1)/2))` and to TA-Lib's `WMA`. Validated in `tests/test_wma.py` against three brute-force per-window references (one per policy) and against pandas to floating-point precision.
