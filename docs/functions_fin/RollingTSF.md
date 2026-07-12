---
name: RollingTSF
title: Rolling Time-Series Forecast (TSF)
implementation_family: fin
topics:
- trend
- regression
tags:
- tsf
- forecast
- regression
- talib
short: Linear regression of y on time, projected one step ahead. TA-Lib's TSF.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Trailing-window length.
nan_policy: ignore
---

# `RollingTSF`

## Description

TA-Lib's *Time-Series Forecast*: fits `y = slope · t + intercept` on the trailing window
`{(0, y_{t-w+1}), …, (w-1, y_t)}` and returns the line evaluated at the *next* bar (local
`t = w`):

$$
\text{TSF}[t] = \text{intercept} + \text{slope} \cdot w
$$

1→1. Bit-exact match to `talib.TSF(real, timeperiod)`. Composes two `detail::RollingSum`
buffers + precomputed time-axis constants. O(1) per step.


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
    from screamer import RollingTSF

    np.random.seed(0)
    price = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, size=300)))
    tsf = RollingTSF(window_size=20)(price)   # one-step-ahead value of the 20-bar trend line

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=price, mode="lines", name="price"), row=1, col=1)
    fig.add_trace(go.Scatter(y=tsf, mode="lines", name="TSF",
                             line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Time-series forecast over 20 bars (RollingTSF)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="forecast", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
