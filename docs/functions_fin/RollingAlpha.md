---
name: RollingAlpha
title: Rolling alpha (regression intercept)
implementation_family: fin
topics:
- regression
tags:
- alpha
- jensen alpha
- regression
- intercept
- pair
short: Rolling OLS intercept of a target regressed on a regressor (companion to RollingBeta).
inputs: 2
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of most recent (target, regressor) pairs in the regression window.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before the window is full: "strict" gives NaN, "expanding" computes
    from the samples seen so far, and "zero" treats the window as pre-filled with zeros.'
nan_policy: ignore
see_also:
- RollingBeta
- RollingLinearRegression
- RollingSpread
- EwAlpha
---

# `RollingAlpha`

## Description

`RollingAlpha` computes the intercept of the rolling least-squares regression of a target on a regressor. With target `y`, regressor `x` and the slope `beta[t]` from `RollingBeta`:

$$
\alpha[t] = \overline{y}_w - \beta[t]\ \cdot\ \overline{x}_w
$$

where the means are taken over the last `window_size` samples. It takes two inputs, `(target, regressor)`, in the same order as [`RollingBeta`](RollingBeta.md), and costs O(1) per step.


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
    from screamer import RollingAlpha

    np.random.seed(0)
    market_ret = np.random.normal(0.0004, 0.012, size=300)
    asset_ret = 0.6 * market_ret + np.random.normal(0.0003, 0.008, size=300)
    alpha = RollingAlpha(window_size=63)(asset_ret, market_ret)   # regression intercept of asset on market

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=asset_ret, mode="lines", name="asset returns"), row=1, col=1)
    fig.add_trace(go.Scatter(y=market_ret, mode="lines", name="market returns"), row=1, col=1)
    fig.add_trace(go.Scatter(y=alpha, mode="lines", name="alpha",
                             line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Rolling alpha of asset on market over 63 bars (RollingAlpha)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="returns", row=1, col=1)
    fig.update_yaxes(title_text="alpha", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
