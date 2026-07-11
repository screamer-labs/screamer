---
name: First
title: First finite value
implementation_family: misc
topics:
- cumulative
tags:
- first
- cumulative
- latch
short: Latch the first finite value seen since reset.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `First`

## Description

`First` latches the first finite value it sees and returns it from then on. NaN inputs are ignored: they return NaN and leave the latched value unchanged. `reset()` clears the latch. As a per-bar reducer it yields the opening value of each bar.

*Parameters*: none.

*NaN handling*: NaN inputs do not update the latch. The output at a NaN index is NaN and subsequent finite samples continue normally.


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
    from screamer import First, Resample

    rng = np.random.default_rng(42)
    n = 100
    x = np.cumsum(rng.normal(0.0, 1.0, size=n))
    idx = np.arange(n, dtype=np.int64)
    opens, bar_idx = Resample(freq=10, agg=First())(x, idx)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=x, mode='lines', name='x[t]',
                             line=dict(color='steelblue')))
    fig.add_trace(go.Scatter(x=bar_idx, y=opens, mode='markers',
                             name='bar open (First)',
                             marker=dict(color='orange', size=8)))
    fig.update_layout(
        title="First: Opening Value of Each Bar",
        xaxis_title="Index",
        yaxis_title="Value",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

`First` keeps a single `double` and a boolean flag. On the first finite input the value is latched; subsequent finite inputs return the same latch. NaN inputs return NaN and leave the latch unchanged. Memory is `O(1)`.
