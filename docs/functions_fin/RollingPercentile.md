---
name: RollingPercentile
title: Rolling percentile
implementation_family: fin
topics:
- statistics
tags:
- percentile
- position
- pandas
short: Percentile (rank/window) of the current value in the trailing window.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Trailing-window length.
nan_policy: ignore
---

# `RollingPercentile`

## Description

Percentile position of the current value within the trailing window:

$$
\text{percentile}[t] = \text{rank}[t] / w
$$

with the same "average" tie rule as `RollingRank`. Returns values in `[1/w, 1]`. Bit-exact
(0.0) to `pandas.Series.rolling(w).rank(pct=True)`.

1→1. O(W) per step.


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
    from screamer import RollingPercentile

    np.random.seed(0)
    price = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, size=300)))
    pct = RollingPercentile(window_size=50)(price)   # where the latest price sits in the last 50 bars, 0 to 1

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=price, mode="lines", name="price"), row=1, col=1)
    fig.add_trace(go.Scatter(y=pct, mode="lines", name="percentile",
                             line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Percentile of latest price in a 50-bar window (RollingPercentile)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="percentile", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
