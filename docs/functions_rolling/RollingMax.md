---
name: RollingMax
title: Rolling maximum
implementation_family: rolling
topics:
- statistics
tags:
- max
- rolling
- maximum
- rolling max
- highest high
short: Trailing-window maximum (monotonic deque).
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 1
  description: Number of most recent samples in the window.
nan_policy: ignore
see_also:
- RollingMin
- RollingMinMax
- RollingArgmax
- ExpandingMax
- CumMax
---

# `RollingMax`

## Description

`RollingMax` computes the maximum of the `window_size` most recent values. It is maintained with a monotonic deque, giving O(1) amortized time per step.


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
    from screamer import RollingMax

    np.random.seed(0)  # reproducible example data

    # Generate example data
    N = 300
    window_size = 30
    data = np.cumsum(np.random.normal(size=300))

    # Apply rolling maximum calculation
    rolling_max = RollingMax(window_size)
    results = rolling_max(data)

    # Plotting with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input Data'))
    fig.add_trace(go.Scatter(y=results, mode='lines', name='Rolling Max', line=dict(color='red')))
    fig.update_layout(
        title=f"Rolling Maximum with Window Size = {window_size}",
        xaxis_title="Index",
        yaxis_title="Value",
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)        
    )
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

### Algorithm

`RollingMax` keeps a monotonic deque (the ascending minima algorithm, mirrored for the maximum): candidates are stored in decreasing order, a new sample removes every smaller candidate from the back, and the front is dropped once it leaves the window. The front is the window maximum.

### Complexity

* Time complexity: `O(1)` amortized per step.
* Space complexity: `O(window_size)`.

For timings see [Performance](../performance.md).
