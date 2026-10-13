---
name: RollingRogersSatchellVol
title: Rolling Rogers-Satchell volatility
implementation_family: rolling
topics:
- volatility
tags:
- rogers-satchell
- range-based
- ohlc
- drift-robust
- vol
- rolling
- realized volatility
- volatility
short: Vol form of the Rogers-Satchell drift-robust range-based estimator.
inputs: 4
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 1
  description: Number of most recent bars over which the per-bar estimate is averaged.
nan_policy: ignore
see_also:
- RollingRogersSatchellVar
- EwRogersSatchellVol
- RollingGarmanKlassVol
- RollingYangZhangVol
- RollingParkinsonVol
references:
- Rogers, L. C. G. and Satchell, S. E. (1991). Estimating variance from high, low and closing
  prices. *Annals of Applied Probability*, 1(4), 504-512. doi:10.1214/aoap/1177005835
---

# `RollingRogersSatchellVol`

## Description

The Rogers-Satchell (1991) range-based volatility estimator. Per-bar variance contribution:

$$
\sigma^2_\text{RS}[t] = \ln\tfrac{H}{C}\ \ln\tfrac{H}{O} + \ln\tfrac{L}{C}\ \ln\tfrac{L}{O}
$$

This expression is averaged with a rolling mean over `window_size` bars to form the estimator. `RollingRogersSatchellVol` returns the square root of `RollingRogersSatchellVar`.

The inputs are `(open, high, low, close)`. The estimator is about 6 times as efficient as close-to-close, slightly less than Garman-Klass, and unlike Garman-Klass it stays unbiased when the price has a non-zero drift, as in trending markets.


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
    from screamer import RollingRogersSatchellVol

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(np.random.normal(0, 0.4, size=300))
    high = np.maximum(open_, close) + wick
    low  = np.minimum(open_, close) - wick
    out = RollingRogersSatchellVol(window_size=20)(open_, high, low, close)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=close, name="close"), row=1, col=1)
    fig.add_trace(go.Scatter(y=out, name="RollingRogersSatchellVol", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Rolling Rogers-Satchell volatility (RollingRogersSatchellVol)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="volatility", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
