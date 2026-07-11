---
name: Last
title: Last finite value
implementation_family: misc
topics:
- cumulative
tags:
- last
- cumulative
short: Return the most recent finite value seen since reset.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `Last`

## Description

`Last` returns the most recent finite value it has seen. NaN inputs are ignored: they return NaN and leave the retained value unchanged. `reset()` clears it. As a per-bar reducer it yields the closing value of each bar.

*Parameters*: none.

*NaN handling*: NaN inputs do not update the retained value. The output at a NaN index is NaN and subsequent finite samples continue normally.


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
    from screamer import Last, Resample

    rng = np.random.default_rng(42)
    n = 100
    x = np.cumsum(rng.normal(0.0, 1.0, size=n))
    idx = np.arange(n, dtype=np.int64)
    closes, bar_idx = Resample(freq=10, agg=Last())(x, idx)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=x, mode='lines', name='x[t]',
                             line=dict(color='steelblue')))
    fig.add_trace(go.Scatter(x=bar_idx, y=closes, mode='markers',
                             name='bar close (Last)',
                             marker=dict(color='green', size=8)))
    fig.update_layout(
        title="Last: Closing Value of Each Bar",
        xaxis_title="Index",
        yaxis_title="Value",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

`Last` keeps a single `double` initialised to NaN. Each finite input updates the retained value and returns it. NaN inputs return NaN and leave the retained value unchanged. Memory is `O(1)`.
