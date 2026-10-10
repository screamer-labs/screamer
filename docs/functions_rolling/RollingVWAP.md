---
name: RollingVWAP
title: Rolling VWAP
implementation_family: rolling
topics:
- volume
tags:
- vwap
- volume-weighted
- ohlcv
- typical-price
- volume weighted average price
- rolling vwap
short: Rolling volume-weighted average price (typical-price weighted).
inputs: 4
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of most recent bars in the window.
nan_policy: ignore
see_also:
- RollingMean
- OBV
- AD
---

# `RollingVWAP`

## Description

Rolling volume-weighted average price using the typical price as the weighting basis:

$$
\text{TP}[t] = (\text{high} + \text{low} + \text{close}) / 3
\qquad
\text{VWAP}[t] = \dfrac{\sum_w \text{TP} \cdot \text{volume}}{\sum_w \text{volume}}
$$

The inputs are `(high, low, close, volume)`. Matches `pandas-ta-classic.vwap`. For a session VWAP (cumulative since a reset point), call `reset()` at the session boundary.

The first valid output is at sample index `window_size - 1`. The two sums are kept as running window sums, so each step is O(1).


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
    from screamer import RollingVWAP

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(np.random.normal(0, 0.4, size=300))
    high = np.maximum(open_, close) + wick
    low  = np.minimum(open_, close) - wick
    volume = np.random.uniform(1e5, 5e5, size=300)
    vwap = RollingVWAP(window_size=20)(high, low, close, volume)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=close, name="close"), row=1, col=1)
    fig.add_trace(go.Scatter(y=vwap, name="RollingVWAP", line=dict(color="red")), row=1, col=1)
    fig.add_trace(go.Bar(y=volume, name="volume"), row=2, col=1)
    fig.update_layout(title="Price with rolling VWAP (RollingVWAP)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="volume", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
