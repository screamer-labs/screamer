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
- open
- opening value
short: Latch the first finite value seen since reset.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- Last
- Resample
- Hold
---

# `First`

## Description

`First` latches the first finite value it sees and returns it from then on. NaN inputs are ignored: they return NaN and leave the latched value unchanged. `reset()` clears the latch. As a per-bar reducer it yields the opening value of each bar.


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

`First` keeps a single `double` and a boolean flag. On the first finite input the value is latched; subsequent finite inputs return the same latch. Memory is `O(1)`.
