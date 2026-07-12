---
name: RollingMaxDrawdown
title: Rolling maximum drawdown
implementation_family: fin
topics:
- risk
tags:
- drawdown
- max-drawdown
- rolling
short: Worst peak-to-trough drawdown inside a trailing window.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 252
  min: 2
  description: Trailing-window length (252 = one trading year, default).
nan_policy: ignore
---

# `RollingMaxDrawdown`

## Description

The worst peak-to-trough loss observed inside the last `window_size` bars. Different from
`MaxDrawdown` (which is the worst loss EVER since reset).

## Implementation

Maintains a circular buffer of the last `w` prices and, on each step, sweeps the buffer
tracking a within-window running peak and the worst drawdown from that peak. **O(window_size)
per step** -- there is no cheap amortised algorithm for the standard definition because the
in-window peak can sit anywhere in the window.

If you want the cheaper "current drawdown vs. rolling-window peak" approximation, compose it
directly:


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
    from screamer import RollingMaxDrawdown

    np.random.seed(0)
    price = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, size=300)))
    rmdd = RollingMaxDrawdown(window_size=63)(price)   # worst drawdown in the last 63 bars

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=price, mode="lines", name="price"), row=1, col=1)
    fig.add_trace(go.Scatter(y=rmdd, mode="lines", name="rolling max drawdown",
                             line=dict(color="red"), fill="tozeroy"), row=2, col=1)
    fig.update_layout(title="Worst drawdown in a trailing window (RollingMaxDrawdown)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="rolling max drawdown", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
