---
name: RollingTLSBeta
title: Rolling total least squares hedge ratio
implementation_family: fin
topics:
- regression
tags:
- total least squares
- orthogonal regression
- hedge ratio
- pairs
- beta
short: "Rolling total least squares (orthogonal regression) slope of the first input on the second: a hedge ratio that does not depend on which leg is called dependent."
inputs: 2
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Window length.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: Warmup behavior. "strict" returns NaN until the window is full, "expanding" uses the samples seen so far, "zero" treats the window as pre-filled with zeros.
nan_policy: ignore
see_also:
- RollingBeta
- KalmanRegression
- RollingSpread
---

# `RollingTLSBeta`

## Description

`RollingTLSBeta(window_size)(a, b)` is the slope of `a` on `b` that minimizes the
perpendicular distances of the points to the line, over the last
`window_size` steps. With the centered sums `Saa`, `Sbb`, and `Sab` of the
window,

    beta = (Saa - Sbb + sqrt((Saa - Sbb)^2 + 4 Sab^2)) / (2 Sab)

Ordinary least squares ([`RollingBeta`](RollingBeta.md)) assumes all the noise
is in the dependent series, so the slope of `a` on `b` is not the inverse of
the slope of `b` on `a`, and a pairs trade's hedge ratio changes with the
choice of dependent leg. Total least squares treats both series alike: the
slope of `a` on `b` is exactly one over the slope of `b` on `a`. It assumes the
two series have noise of the same size, so use it on series of the same scale,
such as log prices. The result is `NaN` when the window has no linear relation
(`Sab = 0`).

<!-- NAN_FOOTNOTE_START -->
## NaN handling

A `NaN` in either input skips the step: the output is `NaN` and the window is unchanged.
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import RollingBeta, RollingTLSBeta

    rng = np.random.default_rng(4)
    n = 1500
    factor = np.cumsum(rng.standard_normal(n)) * 0.02            # what both legs follow
    log_a = 0.5 + 1.1 * factor + rng.standard_normal(n) * 0.02   # each leg with its own noise
    log_b = 4.0 + factor + rng.standard_normal(n) * 0.02

    fig = go.Figure()
    fig.add_hline(y=1.1, line_dash='dash', line_color='gray', annotation_text='true ratio 1.1')
    fig.add_trace(go.Scatter(y=RollingTLSBeta(250)(log_a, log_b), name='TLS, a on b'))
    fig.add_trace(go.Scatter(y=1 / RollingTLSBeta(250)(log_b, log_a), name='TLS, 1 / (b on a)', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(y=RollingBeta(250)(log_a, log_b), name='OLS, a on b'))
    fig.add_trace(go.Scatter(y=1 / RollingBeta(250)(log_b, log_a), name='OLS, 1 / (b on a)'))
    fig.update_yaxes(range=[0.7, 1.5])
    fig.update_layout(title='Total least squares gives one hedge ratio whichever leg is dependent',
                      xaxis_title='Step', yaxis_title='Hedge ratio', margin=dict(l=20, r=20, t=80, b=20))
    fig.show()
```

<!-- HELP_END -->
