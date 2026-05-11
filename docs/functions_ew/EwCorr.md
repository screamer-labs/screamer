---
name: EwCorr
title: Exponentially-weighted correlation
implementation_family: ew
topics:
- correlation
- statistics
tags:
- ew
- correlation
- pair
short: EW Pearson correlation of two parallel streams.
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

# `EwCorr`

## Description

`EwCorr` computes the exponentially weighted moving Pearson correlation of two streams. Bounded to $[-1, 1]$. Matches `pandas.Series.ewm(adjust=True).corr(other)`. This is a **2-input, 1-output** function (`FunctorBase<_, 2, 1>`).

## Parameters

Specify exactly one of the following to set the smoothing factor `alpha`:

- **`com`**: Center of mass. `alpha = 1 / (1 + com)`
- **`span`**: Span. `alpha = 2 / (span + 1)`
- **`halflife`**: Half-life. `alpha = 1 - exp(-log(2) / halflife)`
- **`alpha`**: Directly sets the smoothing factor, `0 < alpha < 1`

The first sample is `NaN`. Subsequent NaNs occur when either input has zero variance over the effective window.

## Formula

The bias-correction factor that `EwVar` and `EwCov` apply cancels in the numerator-vs-denominator ratio, so `EwCorr` uses the simpler unbiased form. With $\bar{x} = S_x / S_w$ and $\bar{y} = S_y / S_w$:

$$
\text{EwCorr} =
\frac{\dfrac{S_{xy}}{S_w} - \bar{x}\,\bar{y}}
     {\sqrt{\left(\dfrac{S_{xx}}{S_w} - \bar{x}^2\right)\left(\dfrac{S_{yy}}{S_w} - \bar{y}^2\right)}}
$$

If either denominator factor is zero (a constant input over the effective window), the output is `NaN`.

## Identity check

`EwCorr` is exactly `EwCov / sqrt(EwVar(x) · EwVar(y))` because the bias factors cancel:

```python
from screamer import EwCorr, EwCov, EwVar
denom = np.sqrt(EwVar(alpha=0.1)(x) * EwVar(alpha=0.1)(y))
np.testing.assert_allclose(
    EwCorr(alpha=0.1)(x, y),
    EwCov(alpha=0.1)(x, y) / denom,
    equal_nan=True, atol=1e-12,
)
```

<!-- HELP_END -->

## Usage Example and Plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import EwCorr

    rng = np.random.default_rng(0)
    n = 400
    # Two series whose correlation regime changes mid-sample.
    x = rng.standard_normal(n)
    y = np.empty(n)
    y[:n//2] =  0.8 * x[:n//2] + 0.2 * rng.standard_normal(n//2)   # high corr
    y[n//2:] = -0.4 * x[n//2:] + 0.6 * rng.standard_normal(n//2)   # negative corr

    rho = EwCorr(span=30)(x, y)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=rho, mode='lines',
                             name='EwCorr(span=30)',
                             line=dict(color='steelblue')))
    fig.add_hline(y=0, line=dict(color='gray', dash='dot'))
    fig.update_layout(
        title="EwCorr: Tracking a Regime Shift in Correlation",
        xaxis_title="Index",
        yaxis_title="Correlation",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.show()
```

## Reference

Equivalent to `pandas.Series.ewm(adjust=True).corr(other)`.
