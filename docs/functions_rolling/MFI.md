---
name: MFI
title: Money Flow Index (MFI)
implementation_family: rolling
topics:
- momentum
- volume
tags:
- mfi
- money-flow
- volume-rsi
- talib
- ohlcv
short: Volume-weighted analogue of RSI on the typical price.
inputs: 4
outputs: 1
parameters:
- name: window_size
  type: int
  default: 14
  min: 2
  description: Lookback period (Wilder's default is 14).
nan_policy: ignore
---

# `MFI`

## Description

Money Flow Index - a volume-weighted analogue of RSI on the typical price:

$$
\begin{aligned}
\text{TP}[t]     &= (H + L + C) / 3 \\
\text{MF}[t]     &= \text{TP} \cdot V \\
\text{pos\_MF}_w &= \sum_w \text{MF}[\text{where}\ \text{TP} > \text{TP}_{t-1}] \\
\text{neg\_MF}_w &= \sum_w \text{MF}[\text{where}\ \text{TP} < \text{TP}_{t-1}] \\
\text{MFI}[t]    &= 100\ \cdot\ \dfrac{\text{pos\_MF}_w}{\text{pos\_MF}_w + \text{neg\_MF}_w}
\end{aligned}
$$

**4-input, 1-output** on `(high, low, close, volume)`. First valid output at sample index
`window_size`. Bit-exact to `talib.MFI` (~1e-14).


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
    from screamer import MFI

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(np.random.normal(0, 0.4, size=300))
    high = np.maximum(open_, close) + wick
    low  = np.minimum(open_, close) - wick
    volume = np.random.uniform(1e5, 5e5, size=300)
    out = MFI(14)(high, low, close, volume)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=close, name="close", line=dict(color="royalblue")), row=1, col=1)
    fig.add_trace(go.Scatter(y=out, name="MFI(14)", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Money flow index (MFI)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="close", row=1, col=1)
    fig.update_yaxes(title_text="MFI (0-100)", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
