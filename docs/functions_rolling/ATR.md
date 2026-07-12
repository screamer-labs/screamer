---
name: ATR
title: Average True Range (ATR)
implementation_family: rolling
topics:
- volatility
tags:
- wilder
- atr
- true-range
- talib
- hlc
short: Wilder-smoothed average of TrueRange.
inputs: 3
outputs: 1
parameters:
- name: window_size
  type: int
  default: 14
  min: 2
  description: Wilder smoothing period (Wilder's original choice is 14).
nan_policy: ignore
---

# `ATR`

## Description

Wilder-smoothed rolling average of `TrueRange`:

$$
\begin{aligned}
\text{ATR}[w] &= \frac{1}{w} \sum_{i=1}^{w} \text{TR}[i] \quad\text{(SMA seed)} \\
\text{ATR}[t] &= \frac{(w - 1) \cdot \text{ATR}[t - 1] + \text{TR}[t]}{w} \quad\text{for } t > w
\end{aligned}
$$

**3-input, 1-output** on `(high, low, close)`. First valid output at sample index
`window_size`. Matches `talib.ATR` bit-exactly post-warmup.


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
    from screamer import ATR

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(np.random.normal(0, 0.4, size=300))
    high = np.maximum(open_, close) + wick
    low  = np.minimum(open_, close) - wick
    out = ATR(14)(high, low, close)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=high, name="high", line=dict(color="#888")), row=1, col=1)
    fig.add_trace(go.Scatter(y=low, name="low", line=dict(color="#bbb")), row=1, col=1)
    fig.add_trace(go.Scatter(y=close, name="close", line=dict(color="royalblue")), row=1, col=1)
    fig.add_trace(go.Scatter(y=out, name="ATR(14)", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Average true range (ATR)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="ATR", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
