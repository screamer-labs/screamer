---
name: RollingRank
title: Rolling rank
implementation_family: fin
topics:
- statistics
tags:
- rank
- position
- pandas
short: Rank of the current value within the trailing window (1-based, average tie
  rule).
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

# `RollingRank`

## Description

Where does the current value sit within the trailing window?

$$
\text{rank}[t] = (\text{\#values} < y_t) + 1 + \tfrac{1}{2}(\text{\#ties} - 1)
$$

Pandas's "average" tie-breaking rule (mean rank among tied values). Returns a 1-based rank
in `[1, w]`.

1→1. Circular window buffer + per-step counting sweep; O(W) per step. Bit-exact (0.0) to
`pandas.Series.rolling(w).rank()`.


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
    from screamer import RollingRank

    np.random.seed(0)
    price = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, size=300)))
    rank = RollingRank(window_size=50)(price)   # 1-based rank of the latest price in the last 50 bars

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=price, mode="lines", name="price"), row=1, col=1)
    fig.add_trace(go.Scatter(y=rank, mode="lines", name="rank",
                             line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Rank of latest price in a 50-bar window (RollingRank)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="rank", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
