---
name: Last
title: Last finite value
implementation_family: misc
topics:
- cumulative
tags:
- last
- cumulative
- close
- closing value
- latest
short: Return the most recent finite value seen since reset.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- First
- Ffill
- Resample
- Hold
---

# `Last`

## Description

`Last` returns the most recent finite value it has seen. NaN inputs are ignored: they return NaN and leave the retained value unchanged. `reset()` clears it. As a per-bar reducer it yields the closing value of each bar.


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

`Last` keeps a single `double` initialized to NaN. Each finite input updates the retained value and returns it. Memory is `O(1)`.
