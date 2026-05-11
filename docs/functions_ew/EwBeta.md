---
name: EwBeta
title: Exponentially-weighted beta
implementation_family: ew
topics:
- correlation
- regression
tags:
- ew
- beta
- pair
- capm
short: 'EW CAPM beta: cov(target, regressor) / var(regressor).'
inputs: 2
outputs: 1
parameters:
- name: com
  type: float
  default: null
  description: Center of mass (alpha = 1 / (1 + com)). Exclusive with span/halflife/alpha.
- name: span
  type: float
  default: 20.0
  description: Span (alpha = 2 / (span + 1)). Default smoothing parameter. Exclusive
    with com/halflife/alpha.
- name: halflife
  type: float
  default: null
  description: Halflife (alpha = 1 - 0.5^(1/halflife)). Exclusive with com/span/alpha.
- name: alpha
  type: float
  default: null
  description: Smoothing parameter directly. Exclusive with com/span/halflife.
---

# `EwBeta`

## Description

`EwBeta` computes the exponentially weighted regression slope of `x` on `y`:

$$
\text{EwBeta}(x, y) = \frac{\operatorname{Cov}(x, y)}{\operatorname{Var}(y)}
$$

This is a **2-input, 1-output** function (`FunctorBase<_, 2, 1>`).

## Argument-order convention

The **first argument is the dependent (target)**, the **second is the regressor**. Calling `EwBeta(asset_returns, market_returns)` returns the asset's beta with respect to the market, matching the CAPM definition and the existing `RollingBeta` convention.

This is the *opposite* of pandas (`pandas.ols`-style and most pandas docs use slope of `y` on `x`).

## Parameters

Specify exactly one of the following to set the smoothing factor `alpha`:

- **`com`**: Center of mass. `alpha = 1 / (1 + com)`
- **`span`**: Span. `alpha = 2 / (span + 1)`
- **`halflife`**: Half-life. `alpha = 1 - exp(-log(2) / halflife)`
- **`alpha`**: Directly sets the smoothing factor, `0 < alpha < 1`

The first sample is `NaN`. Subsequent NaNs occur when the regressor has zero variance over the effective window.

## Formula

The bias-correction factor that `EwVar` and `EwCov` apply cancels between numerator and denominator, so `EwBeta` uses the simpler unbiased form. With $\bar{x} = S_x / S_w$ and $\bar{y} = S_y / S_w$:

$$
\text{EwBeta} = \frac{\dfrac{S_{xy}}{S_w} - \bar{x}\,\bar{y}}
                     {\dfrac{S_{yy}}{S_w} - \bar{y}^2}
$$

If the denominator is zero or non-positive, the output is `NaN`.

## Identity check

By definition `EwBeta(x, y) == EwCov(x, y) / EwVar(y)`. Verified to ~1e-12 in the test suite.

<!-- HELP_END -->

## Usage Example

```python
import numpy as np
import pandas as pd
from screamer import EwBeta

rng = np.random.default_rng(0)
market = rng.standard_normal(500)
# An asset with true beta = 1.5 plus idiosyncratic noise
asset = 1.5 * market + 0.5 * rng.standard_normal(500)

# CAPM convention: dependent first, regressor second
beta = EwBeta(span=60)(asset, market)

# Validate against pandas cov/var
ref = (
    pd.Series(asset).ewm(span=60).cov(pd.Series(market)).to_numpy()
    / pd.Series(market).ewm(span=60).var().to_numpy()
)
np.testing.assert_allclose(beta, ref, equal_nan=True, atol=1e-10)
```

## Reference

Equivalent to `pandas.Series.ewm(adjust=True).cov(market) / market.ewm(adjust=True).var()`. Pandas does not expose an `ewm.beta()` method directly.
