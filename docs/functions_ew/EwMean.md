---
name: EwMean
title: Exponentially-weighted mean
implementation_family: ew
topics:
- smoothing
- statistics
tags:
- ema
- exponential moving average
- ewma
- exponential
- mean
- moving-average
- ew
short: Exponentially-weighted moving average (pandas adjust=True).
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
- EwStd
- EwZscore
- RollingMean
- DEMA
- KAMA
nan_policy: ignore
---

# `EwMean`

## Description

`EwMean` computes the exponentially weighted mean of the input (the exponential moving average, EMA). The smoothing factor `alpha` is set by exactly one of `com`, `span`, `halflife` or `alpha`; a larger `alpha` gives recent samples more weight. The result matches pandas `ewm(adjust=True).mean()`.

Each step decays two running sums by $(1 - \alpha)$ and adds the new sample:

$$
S_x \leftarrow (1-\alpha) S_x + x_t, \qquad S_w \leftarrow (1-\alpha) S_w + 1, \qquad
\text{EwMean}[t] = \frac{S_x}{S_w}
$$

Dividing by the weight sum $S_w$ is the pandas `adjust=True` normalization: the first output equals the first sample, and early outputs are not pulled toward zero.

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
    from screamer import EwMean

    np.random.seed(0)  # reproducible example data

    # Generate example data
    data = np.cumsum(np.random.normal(size=300))

    # Compute exponentially weighted mean with a span of 20
    ewmean_data = EwMean(span=20)(data)

    # Create subplots with original and transformed data
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[1/2, 1/2], vertical_spacing=0.1)

    fig.add_trace(go.Scatter(y=data, mode='lines', name='Original Data'), row=1, col=1)
    fig.add_trace(go.Scatter(y=ewmean_data, mode='lines', name='EwMean', line=dict(color='red')), row=2, col=1)

    fig.update_layout(
        title="Exponentially Weighted Moving Mean",
        xaxis_title="Index",
        yaxis=dict(title="Original Data"),
        yaxis2=dict(title="EwMean"),
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)        
    )

    fig.show()
```

<!-- HELP_END -->
