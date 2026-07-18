---
name: RollingGarmanKlassVol
title: Rolling Garman-Klass volatility
implementation_family: rolling
topics:
- volatility
tags:
- garman-klass
- range-based
- ohlc
- vol
- rolling
short: Vol form of the Garman-Klass range-based volatility estimator (OHLC).
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

# `RollingGarmanKlassVol`

## Description

The Garman-Klass (1980) range-based volatility estimator. Per-bar variance contribution:

$$
\sigma^2_\text{GK}[t] = \tfrac{1}{2}\big(\ln H/L\big)^2 - (2\ln 2 - 1)\big(\ln C/O\big)^2
$$

This expression is averaged with a rolling mean over `window_size` bars to form the estimator. The `Vol`
variant returns `sqrt(Var)` (bit-exact via the same internal state).

**4-input, 1-output** on `(open, high, low, close)`. ~7.4x more statistically efficient
than close-to-close `RollingStd` *under the model's assumptions* (zero drift, no overnight
gaps).


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
    from screamer import RollingGarmanKlassVol

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(np.random.normal(0, 0.4, size=300))
    high = np.maximum(open_, close) + wick
    low  = np.minimum(open_, close) - wick
    out = RollingGarmanKlassVol(window_size=20)(open_, high, low, close)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=close, name="close"), row=1, col=1)
    fig.add_trace(go.Scatter(y=out, name="RollingGarmanKlassVol", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Rolling Garman-Klass volatility (RollingGarmanKlassVol)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="volatility", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
