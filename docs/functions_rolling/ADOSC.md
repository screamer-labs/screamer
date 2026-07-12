---
name: ADOSC
title: Chaikin A/D Oscillator (ADOSC)
implementation_family: rolling
topics:
- momentum
- volume
tags:
- adosc
- chaikin
- oscillator
- talib
- ohlcv
short: Difference of fast and slow EMA of the Accumulation/Distribution line.
inputs: 4
outputs: 1
parameters:
- name: fast
  type: int
  default: 3
  min: 2
  description: Fast EMA period.
- name: slow
  type: int
  default: 10
  min: 2
  description: Slow EMA period.
nan_policy: ignore
---

# `ADOSC`

## Description

Chaikin A/D Oscillator: difference of two EMAs of the `AD` line.

$$
\text{ADOSC}[t] = \text{EMA}(\text{AD},\ \text{fast})[t] - \text{EMA}(\text{AD},\ \text{slow})[t]
$$

**4-input, 1-output** on `(high, low, close, volume)`. Default `(fast=3, slow=10)` matches
TA-Lib.

The underlying EMA is `screamer.EwMean` (pandas `adjust=True`), so `ADOSC` inherits the
same documented divergence from TA-Lib's `ADOSC` as `DEMA`/`TEMA`/`MACD`/`TRIX`. The class
matches the explicit pandas-composition reference bit-exactly. See `docs/conventions.md`
for the divergence detail.


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
    from screamer import ADOSC

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(np.random.normal(0, 0.4, size=300))
    high = np.maximum(open_, close) + wick
    low  = np.minimum(open_, close) - wick
    volume = np.random.uniform(1e5, 5e5, size=300)
    out = ADOSC(3, 10)(high, low, close, volume)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=close, name="close", line=dict(color="royalblue")), row=1, col=1)
    fig.add_trace(go.Scatter(y=out, name="ADOSC(3,10)", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Accumulation/distribution oscillator (ADOSC)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="close", row=1, col=1)
    fig.update_yaxes(title_text="ADOSC", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
