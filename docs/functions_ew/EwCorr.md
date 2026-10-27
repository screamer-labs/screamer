---
name: EwCorr
title: Exponentially-weighted correlation
implementation_family: ew
topics:
- regression
tags:
- ew
- correlation
- corr
- pearson
- exponential
- ewm corr
short: EW Pearson correlation of two parallel streams.
inputs: 2
outputs: 1
parameters:
- name: com
  type: float
  default: null
  description: Center of mass c, giving alpha = 1 / (1 + c). Must be positive. Give exactly
    one of com, span, halflife and alpha.
- name: span
  type: float
  default: 20.0
  description: Span s in samples, giving alpha = 2 / (s + 1). Must be greater than 1. Give
    exactly one of com, span, halflife and alpha. The default shown is an example value; the
    constructor has no default.
- name: halflife
  type: float
  default: null
  description: Half-life h in samples, giving alpha = 1 - exp(-ln(2) / h). Must be positive.
    Give exactly one of com, span, halflife and alpha.
- name: alpha
  type: float
  default: null
  description: Smoothing factor, with 0 < alpha < 1; a larger alpha gives recent samples more
    weight. Give exactly one of com, span, halflife and alpha.
see_also:
- EwCov
- EwBeta
- RollingCorr
nan_policy: ignore
---

# `EwCorr`

## Description

`EwCorr` computes the exponentially weighted Pearson correlation of two series, bounded to $[-1, 1]$. The smoothing factor `alpha` is set by exactly one of `com`, `span`, `halflife` or `alpha`. The result matches pandas `ewm(adjust=True).corr(other)`.

With $\bar{x} = S_x / S_w$ and $\bar{y} = S_y / S_w$, where every running sum decays by $(1-\alpha)$ per step:

$$
\text{EwCorr}[t] =
\frac{\dfrac{S_{xy}}{S_w} - \bar{x}\,\bar{y}}
     {\sqrt{\left(\dfrac{S_{xx}}{S_w} - \bar{x}^2\right)\left(\dfrac{S_{yy}}{S_w} - \bar{y}^2\right)}}
$$

This equals `EwCov(x, y) / sqrt(EwVar(x) * EwVar(y))`: the bias-correction factors of the covariance and the variances cancel, so they are left out. The output is `NaN` when either input is constant over the effective window, which includes the first sample. It is also `NaN` once either weighted variance is below the smallest normal double (about 2.2e-308), where it has lost its precision: for a unit-variance input that has stopped varying, after about 6700 rows at alpha = 0.1. The implementation takes the sums about origins close to the means, which gives the same result without losing precision when the levels are large next to the spread (see [Accuracy at large levels](../conventions.md#accuracy-at-large-levels)).

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

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

<!-- HELP_END -->
