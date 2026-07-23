---
name: EwCov
title: Exponentially-weighted covariance
implementation_family: ew
topics:
- regression
tags:
- ew
- covariance
- pair
short: EW covariance of two parallel streams.
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
nan_policy: ignore
---

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


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
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

## Numerical caveat

The recurrence uses uncentered sums ($S_{xy}$, $\bar{x}$, $\bar{y}$ separately). For inputs with very small variance, the cancellation `S_{xy}/S_w - mean_x·mean_y` is ill-conditioned. In typical use the residual error is `O(1e-12)`; for *exactly* constant inputs the error can climb to `O(1e-9)` and the value won't be exactly zero. Pandas uses centered updates internally and avoids this. The behaviour difference only matters for degenerate inputs.

## Reference

Equivalent to `pandas.Series.ewm(adjust=True, bias=False).cov(other)`.
