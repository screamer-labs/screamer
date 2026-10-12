---
name: EwCov
title: Exponentially-weighted covariance
implementation_family: ew
topics:
- regression
tags:
- ew
- covariance
- cov
- exponential
- ewm cov
short: EW covariance of two parallel streams.
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
- EwCorr
- EwVar
- EwBeta
- RollingCov
nan_policy: ignore
---

# `EwCov`

## Description

`EwCov` computes the exponentially weighted covariance of two series, with the same bias correction as [`EwVar`](EwVar.md). The smoothing factor `alpha` is set by exactly one of `com`, `span`, `halflife` or `alpha`. The result matches pandas `ewm(adjust=True, bias=False).cov(other)`.

With $\bar{x} = S_x / S_w$ and $\bar{y} = S_y / S_w$, where every running sum decays by $(1-\alpha)$ per step:

$$
\text{EwCov}[t] = \left( \frac{S_{xy}}{S_w} - \bar{x}\,\bar{y} \right) \frac{N_\text{eff}}{N_\text{eff} - 1},
\qquad N_\text{eff} = \frac{S_w^2}{S_{ww}}
$$

The first output is `NaN`, because the bias correction needs $N_\text{eff} > 1$. The implementation takes the sums about origins close to the means, which gives the same result without losing precision when the levels are large next to the spread (see [Accuracy at large levels](../conventions.md#accuracy-at-large-levels)).

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
    from plotly.subplots import make_subplots
    from screamer import EwCov

    rng = np.random.default_rng(0)
    N = 300
    x = rng.standard_normal(N)
    y = 1.5 * x + 0.5 * rng.standard_normal(N)
    ewcov = EwCov(span=60)(x, y)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=x, name='x', line=dict(color='steelblue')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=y, name='y = 1.5x + noise', line=dict(color='orange')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=ewcov, name='EwCov(span=60)',
                             line=dict(color='crimson')), row=2, col=1)
    fig.update_layout(
        title='EwCov: exponentially weighted covariance of two return streams',
        yaxis=dict(title='returns'),
        yaxis2=dict(title='covariance'),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Numerical accuracy

The recurrence uses uncentered sums, so the difference $S_{xy}/S_w - \bar{x}\bar{y}$ loses precision when the variance is small relative to the mean. In typical use the error is about `1e-12`. For exactly constant inputs it can reach about `1e-9`, so the result is not exactly zero. pandas uses centered updates and does not have this error; the difference matters only for such degenerate inputs.
