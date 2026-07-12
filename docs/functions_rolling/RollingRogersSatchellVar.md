---
name: RollingRogersSatchellVar
title: Rolling Rogers-Satchell varariance
implementation_family: rolling
topics:
- volatility
tags:
- rogers-satchell
- range-based
- ohlc
- drift-robust
- var
- rolling
short: Var form of the Rogers-Satchell drift-robust range-based estimator.
inputs: 4
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Smoothing window.
nan_policy: ignore
---

# `RollingRogersSatchellVar`

## Description

The Rogers-Satchell (1991) range-based volatility estimator. Per-bar variance contribution:

$$
\sigma^2_\text{RS}[t] = \ln\tfrac{H}{C}\ \ln\tfrac{H}{O} + \ln\tfrac{L}{C}\ \ln\tfrac{L}{O}
$$

This expression is averaged with a rolling mean over `window_size` bars to form the estimator. The `Vol`
variant returns `sqrt(Var)` (bit-exact via the same internal state).

**4-input, 1-output** on `(open, high, low, close)`. Slightly less efficient (~6x vs
close-to-close) than Garman-Klass but **drift-robust** - works correctly when the underlying
drift is non-zero, a much more realistic assumption for trending markets.


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
    from screamer import RollingRogersSatchellVar

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(np.random.normal(0, 0.4, size=300))
    high = np.maximum(open_, close) + wick
    low  = np.minimum(open_, close) - wick
    out = RollingRogersSatchellVar(window_size=20)(open_, high, low, close)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=close, name="close"), row=1, col=1)
    fig.add_trace(go.Scatter(y=out, name="RollingRogersSatchellVar", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Rolling Rogers-Satchell variance (RollingRogersSatchellVar)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="variance", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
