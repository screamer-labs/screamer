---
name: EwZscore
title: Exponentially-weighted z-score
implementation_family: ew
topics:
- standardization
tags:
- ew
- zscore
- z-score
- standardize
- standard score
- normalize
- standardization
- anomaly
short: Latest sample standardized by the EW mean and std.
inputs: 1
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
- EwMean
- EwStd
- RollingZscore
- CrossZscore
nan_policy: ignore
---

# `EwZscore`

## Description

`EwZscore` standardizes each sample by the exponentially weighted mean and standard deviation of the series, both computed with the same smoothing factor. The smoothing factor `alpha` is set by exactly one of `com`, `span`, `halflife` or `alpha`.

$$
\text{EwZscore}[t] = \frac{x_t - \text{EwMean}[t]}{\text{EwStd}[t]}
$$

The mean and standard deviation at step $t$ include $x_t$ itself. $\text{EwStd}$ is the bias-corrected estimate described on [`EwStd`](EwStd.md), so the first output is `NaN`. The output is also `NaN` where the weighted variance is 0: for a constant series, and for a series that has been constant long enough for its variance to underflow.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Description

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import EwZscore

    np.random.seed(0)  # reproducible example data

    data = np.cumsum(np.random.normal(size=300))
    ewzscore_data = EwZscore(span=20)(data)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[1/2, 1/2],
        vertical_spacing=0.1
    )

    fig.add_trace(go.Scatter(y=data, mode='lines', name='Original Data'), row=1, col=1)
    fig.add_trace(go.Scatter(y=ewzscore_data, mode='lines', name='EwZscore', line=dict(color='red')), row=2, col=1)

    fig.update_layout(
        title="Exponentially Weighted Moving Z-Score",
        xaxis_title="Index",
        yaxis=dict(title="Original Data"),
        yaxis2=dict(title="EwZscore", range=[-4, 4]),
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)        
    )

    fig.show()
```

<!-- HELP_END -->
