---
name: RollingPercentile
title: Rolling percentile
implementation_family: fin
topics:
- statistics
tags:
- percentile
- percentile rank
- position
- pandas
- rank pct
short: Percentile rank (rank / window) of the current value within the trailing window.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 1
  description: Number of most recent values the current value is ranked against.
nan_policy: ignore
see_also:
- RollingRank
- RollingQuantile
- RollingZscore
- CrossRank
---

# `RollingPercentile`

## Description

`RollingPercentile` is the percentile position of the current value within the trailing window:

$$
\text{percentile}[t] = \text{rank}[t] / w
$$

with the same "average" tie rule as [`RollingRank`](RollingRank.md). The output lies in `[1/w, 1]` and matches `pandas.Series.rolling(w).rank(pct=True)`. Each step costs O(window_size).


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
