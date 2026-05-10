# `EwCov`

## Description

`EwCov` computes the exponentially weighted moving covariance of two streams. Same bias-correction convention as `EwVar`: the formula matches `pandas.Series.ewm(adjust=True, bias=False).cov(other)`. This is a **2-input, 1-output** function (`FunctorBase<_, 2, 1>`).

## Parameters

Specify exactly one of the following to set the smoothing factor `alpha`:

- **`com`**: Center of mass. `alpha = 1 / (1 + com)`
- **`span`**: Span. `alpha = 2 / (span + 1)`
- **`halflife`**: Half-life. `alpha = 1 - exp(-log(2) / halflife)`
- **`alpha`**: Directly sets the smoothing factor, `0 < alpha < 1`

The first sample is `NaN` (need `n_eff > 1` for the bias correction).

## Formula

Tracks five running sums per step. With $\bar{x} = S_x / S_w$ and $\bar{y} = S_y / S_w$:

$$
\text{EwCov} = \left( \frac{S_{xy}}{S_w} - \bar{x}\,\bar{y} \right) \cdot \frac{N_{\text{eff}}}{N_{\text{eff}} - 1}
$$

where $N_{\text{eff}} = S_w^2 / S_{ww}$ is the effective sample size, computed exactly as in `EwVar`. The bias correction makes the estimator unbiased under independent sampling.

## Usage Example

```python
import numpy as np
import pandas as pd
from screamer import EwCov

rng = np.random.default_rng(0)
x = rng.standard_normal(200)
y = 0.7 * x + 0.3 * rng.standard_normal(200)

# Streaming
ours = EwCov(span=20)(x, y)

# Pandas reference (matches to ~1e-12)
ref = pd.Series(x).ewm(span=20).cov(pd.Series(y)).to_numpy()
np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-12)
```

## Numerical caveat

The recurrence uses uncentered sums ($S_{xy}$, $\bar{x}$, $\bar{y}$ separately). For inputs with very small variance, the cancellation `S_{xy}/S_w - mean_x·mean_y` is ill-conditioned. In typical use the residual error is `O(1e-12)`; for *exactly* constant inputs the error can climb to `O(1e-9)` and the value won't be exactly zero. Pandas uses centered updates internally and avoids this. The behaviour difference only matters for degenerate inputs.

## Reference

Equivalent to `pandas.Series.ewm(adjust=True, bias=False).cov(other)`.
