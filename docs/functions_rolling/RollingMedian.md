---
name: RollingMedian
title: Rolling median
implementation_family: rolling
topics:
- statistics
- outliers
tags:
- median
- rolling
- rolling median
- robust
short: Trailing-window median.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of most recent samples in the window.
nan_policy: ignore
see_also:
- RollingQuantile
- RollingMedianAD
- RollingMean
- Hampel
---

# `RollingMedian`

## Description

`RollingMedian` computes the median of the `window_size` most recent values. The window is kept as two balanced sorted halves, so each step costs O(log window_size).


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
    from screamer import RollingMedian

    np.random.seed(0)  # reproducible example data

    # Generate example data
    N = 300
    window_size = 30
    data = np.cumsum(np.random.normal(size=300))

    # Apply rolling maximum calculation
    rolling_median = RollingMedian(window_size)
    results = rolling_median(data)

    # Plotting with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input Data'))
    fig.add_trace(go.Scatter(y=results, mode='lines', name='Rolling Median', line=dict(color='red')))
    fig.update_layout(title=f"Rolling median with Window Size = {window_size}",
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

`RollingMedian` keeps the lower half of the window in one sorted container (`low`) and the upper half in another (`high`). When a new value is added, it is inserted into the appropriate heap based on its value relative to the current median. After insertion, the algorithm rebalances the heaps to ensure that the difference in their sizes is at most one. This setup enables quick median retrieval: if the heaps are equal in size, the median is the average of the largest value in `low` and the smallest in `high`; if they differ, the median is the top of `low`.

### Complexity

* **Time Complexity**: `O(log(window_size))` per new element due to the insertion and deletion operations in the heaps.
* **Space Complexity**: `O(window_size)`, as only elements within the current window are stored.


For timings see [Performance](../performance.md).
