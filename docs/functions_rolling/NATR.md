---
name: NATR
title: Normalized ATR (NATR)
implementation_family: rolling
topics:
- volatility
- volatility-indicators
tags:
- wilder
- natr
- atr-normalized
- talib
- hlc
- normalized average true range
- atr percent
- volatility
short: ATR scaled to a percentage of the current close.
inputs: 3
outputs: 1
parameters:
- name: window_size
  type: int
  default: 14
  min: 2
  description: 'Wilder smoothing period: the smoother seeds on the first window_size values,
    then each step keeps (window_size - 1) / window_size of the previous value.'
see_also:
- ATR
- TrueRange
- KeltnerChannels
references:
- Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
nan_policy: ignore
---

# `NATR`

## Description

`NATR` is [`ATR`](ATR.md) as a percentage of the current close:

$$
\text{NATR}[t] = 100\ \cdot\ \frac{\text{ATR}[t]}{C[t]}
$$

Expressing the range relative to price makes it comparable across instruments: an ATR
of 1 on a price of 10 is a much larger move than the same ATR on a price of 1000.

Inputs are `(high, low, close)`. The output matches `talib.NATR` exactly.

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
    from screamer import NATR

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(np.random.normal(0, 0.4, size=300))
    high = np.maximum(open_, close) + wick
    low  = np.minimum(open_, close) - wick
    out = NATR(14)(high, low, close)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=high, name="high", line=dict(color="#888")), row=1, col=1)
    fig.add_trace(go.Scatter(y=low, name="low", line=dict(color="#bbb")), row=1, col=1)
    fig.add_trace(go.Scatter(y=close, name="close", line=dict(color="royalblue")), row=1, col=1)
    fig.add_trace(go.Scatter(y=out, name="NATR(14)", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Normalized average true range (NATR)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="NATR (%)", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
