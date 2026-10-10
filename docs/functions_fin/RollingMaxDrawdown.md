---
name: RollingMaxDrawdown
title: Rolling maximum drawdown
implementation_family: fin
topics:
- risk
tags:
- drawdown
- max-drawdown
- max drawdown
- mdd
- rolling
short: Worst peak-to-trough drawdown inside a trailing window.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 252
  min: 2
  description: Number of most recent prices searched for the worst peak-to-trough loss (252
    is one trading year of daily data).
nan_policy: ignore
see_also:
- MaxDrawdown
- Drawdown
- RollingCalmar
- RollingMax
---

# `RollingMaxDrawdown`

## Description

`RollingMaxDrawdown` is the worst peak-to-trough loss, as a negative fraction, among the last `window_size` prices. Both the peak and the trough must lie inside the window. [`MaxDrawdown`](MaxDrawdown.md) is the worst loss since the start of the stream.

Each step scans the window, so the cost is O(window_size) per step. The in-window peak can sit anywhere in the window, which rules out a cheaper update.


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
