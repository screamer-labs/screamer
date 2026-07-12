---
name: TrueRange
title: True Range (Wilder)
implementation_family: rolling
topics:
- volatility
tags:
- wilder
- true-range
- tr
- talib
- hlc
short: Per-bar true range accounting for overnight gaps (Wilder, 1978).
inputs: 3
outputs: 1
parameters: []
nan_policy: ignore
---

# `TrueRange`

## Description

The per-bar true-range quantity that accounts for overnight gaps:

$$
\text{TR}[t] = \max\big(\ H - L,\ \ |H - C_{t-1}|,\ \ |L - C_{t-1}|\ \big)
$$

**3-input, 1-output** on `(high, low, close)`. The first sample returns `NaN` (no previous
close). Otherwise stateless. Bit-exact to `talib.TRANGE`.


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
    from screamer import TrueRange

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(np.random.normal(0, 0.4, size=300))
    high = np.maximum(open_, close) + wick
    low  = np.minimum(open_, close) - wick
    out = TrueRange()(high, low, close)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=high, name="high", line=dict(color="#888")), row=1, col=1)
    fig.add_trace(go.Scatter(y=low, name="low", line=dict(color="#bbb")), row=1, col=1)
    fig.add_trace(go.Scatter(y=close, name="close", line=dict(color="royalblue")), row=1, col=1)
    fig.add_trace(go.Scatter(y=out, name="TrueRange", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="True range per bar (TrueRange)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="true range", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
