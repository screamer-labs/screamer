---
name: KeltnerChannels
title: Keltner Channels
implementation_family: rolling
topics:
- volatility
- bands
tags:
- keltner
- channels
- envelope
- atr-based
- keltner channel
- atr bands
- volatility bands
short: 'Volatility-adapted envelope: EMA midline plus/minus a multiple of ATR.'
inputs: 3
outputs: 3
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Period for both the EMA midline and the ATR offset.
- name: num_atr
  type: float
  default: 2.0
  min: 0.0
  description: Distance from the middle line to each outer line, in multiples of the ATR.
output_names:
- lower
- middle
- upper
see_also:
- ATR
- EwMean
- BollingerBands
- DonchianChannels
references:
- Keltner, C. W. (1960). *How to Make Money in Commodities*. The Keltner Statistical Service.
nan_policy: ignore
---

# `KeltnerChannels`

## Description

`KeltnerChannels` is a volatility-adapted envelope. The middle line is an EMA of the
close, and the upper and lower lines are offset from it by a multiple of the ATR:

$$
\begin{aligned}
\text{mid}[t]   &= \text{EMA}(\text{close},\ \text{window\_size}) \\
\text{atr}[t]   &= \text{ATR}(\text{high},\ \text{low},\ \text{close},\ \text{window\_size}) \\
\text{upper}[t] &= \text{mid} + \text{num\_atr}\ \cdot\ \text{atr} \\
\text{lower}[t] &= \text{mid} - \text{num\_atr}\ \cdot\ \text{atr}
\end{aligned}
$$

Inputs are `(high, low, close)`. The EMA is `EwMean(span=window_size)` and the range
is `ATR(window_size)`. The first valid output is at sample index `window_size`. Each
step costs `O(1)`.

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
    from screamer import KeltnerChannels

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(np.random.normal(0, 0.4, size=300))
    high = np.maximum(open_, close) + wick
    low  = np.minimum(open_, close) - wick
    out = np.asarray(KeltnerChannels(20)(high, low, close))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=close, name="close", line=dict(color="royalblue")), row=1, col=1)
    fig.add_trace(go.Scatter(y=close, name="close", line=dict(color="royalblue")), row=2, col=1)
    fig.add_trace(go.Scatter(y=out[:, 2], name="upper", line=dict(color="red")), row=2, col=1)
    fig.add_trace(go.Scatter(y=out[:, 1], name="mid", line=dict(color="gray")), row=2, col=1)
    fig.add_trace(go.Scatter(y=out[:, 0], name="lower", line=dict(color="green")), row=2, col=1)
    fig.update_layout(title="Keltner channels (KeltnerChannels)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="close", row=1, col=1)
    fig.update_yaxes(title_text="close and bands", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
