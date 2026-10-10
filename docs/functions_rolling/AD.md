---
name: AD
title: Accumulation/Distribution Line (Chaikin)
implementation_family: rolling
topics:
- cumulative
- volume
tags:
- ad
- accumulation-distribution
- chaikin
- talib
- ohlcv
- accumulation distribution line
- a/d line
short: Chaikin Accumulation/Distribution Line.
inputs: 4
outputs: 1
parameters: []
see_also:
- ADOSC
- OBV
- MFI
- SignedVolume
nan_policy: ignore
---

# `AD`

## Description

`AD` is Chaikin's Accumulation/Distribution line: a running sum of volume, each bar's
volume weighted by where the close sits within the bar's range.

$$
\text{CLV}[t] = \dfrac{(C - L) - (H - C)}{H - L}
\qquad
\text{AD}[t]  = \text{AD}[t-1] + \text{CLV}\ \cdot\ V[t]
$$

The close location value (CLV) lies in `[-1, +1]`: +1 means the bar closed at its high
(full accumulation), -1 at its low (full distribution). When `high == low` the CLV is
undefined and the AD line is unchanged, following TA-Lib.

Inputs are `(high, low, close, volume)`. The line is cumulative from the first sample,
with no window and no warmup, and matches `talib.AD` exactly.

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
    from screamer import AD

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(np.random.normal(0, 0.4, size=300))
    high = np.maximum(open_, close) + wick
    low  = np.minimum(open_, close) - wick
    volume = np.random.uniform(1e5, 5e5, size=300)
    out = AD()(high, low, close, volume)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=close, name="close", line=dict(color="royalblue")), row=1, col=1)
    fig.add_trace(go.Scatter(y=out, name="AD line", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Accumulation/distribution line (AD)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="close", row=1, col=1)
    fig.update_yaxes(title_text="AD line", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
