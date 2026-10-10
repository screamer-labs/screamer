---
name: RollingResidualStd
title: Rolling residual std
implementation_family: fin
topics:
- regression
tags:
- residual
- spread
- std
- spread volatility
- pair
- pairs-trading
short: Standard deviation of the rolling hedge-adjusted residual y - beta * x.
inputs: 2
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of most recent (y, x) pairs used for both the hedge ratio and the standard
    deviation.
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
- RollingSpread
- RollingBeta
- RollingZscore
- EwResidualStd
---

# `RollingResidualStd`

## Description

`RollingResidualStd` is the sample standard deviation (ddof=1) of the hedge-adjusted spread `y − β·x` over the trailing window. It takes two inputs, `(y, x)`, in the same order as [`RollingSpread`](RollingSpread.md):

$$
\sigma_\text{spread}[t] = \text{RollingStd}\big(\text{RollingSpread}(y, x)\big)[t]
$$

Each step costs O(1). The spread is `NaN` during its own warmup, so the standard deviation starts accumulating once the spread is defined. Divide the spread by this value to get a pairs z-score.


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
    from screamer import RollingResidualStd

    np.random.seed(0)
    x = np.random.normal(0.0004, 0.012, size=300)
    y = 0.8 * x + np.random.normal(0.0003, 0.007, size=300)
    resid = RollingResidualStd(window_size=63)(y, x)   # std of the hedge-adjusted residual y - beta*x

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=y, mode="lines", name="y returns"), row=1, col=1)
    fig.add_trace(go.Scatter(y=x, mode="lines", name="x returns"), row=1, col=1)
    fig.add_trace(go.Scatter(y=resid, mode="lines", name="residual std",
                             line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Rolling residual std over 63 bars (RollingResidualStd)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="returns", row=1, col=1)
    fig.update_yaxes(title_text="residual std", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
