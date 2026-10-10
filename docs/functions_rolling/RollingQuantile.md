---
name: RollingQuantile
title: Rolling quantile
implementation_family: rolling
topics:
- statistics
tags:
- quantile
- percentile
- rolling
- quartile
- rolling quantile
short: Trailing-window quantile (e.g. 0.25 = lower quartile).
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of most recent samples in the window.
- name: quantile
  type: float
  default: 0.5
  min: 0.0
  max: 1.0
  description: Quantile to return; 0.5 gives the median. Values between two order statistics
    are interpolated linearly.
nan_policy: ignore
see_also:
- RollingMedian
- RollingPercentile
- RollingIqr
- RollingRank
---

# `RollingQuantile`

## Description

`RollingQuantile` returns the `quantile` of the `window_size` most recent values. A `quantile` of 0.5 gives the median, and 0.25 and 0.75 give the lower and upper quartiles. When the quantile falls between two order statistics, the result is interpolated linearly between them (the `linear` method of `numpy.quantile` and pandas). The output is `NaN` until the window is full, so the first `window_size - 1` values are `NaN`.


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
    from screamer import RollingQuantile

    np.random.seed(0)  # reproducible example data

    # Generate example data
    data = np.cumsum(np.random.normal(size=300))

    # Rolling 25th and 75th quantiles with a window size of 30
    quantile_25_data = RollingQuantile(window_size=30, quantile=0.25)(data)
    quantile_75_data = RollingQuantile(window_size=30, quantile=0.75)(data)

    # Create subplots with specified row heights and shared x-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[1/2, 1/2],
        vertical_spacing=0.1
    )

    # Add traces for original data and quantiles
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input Data'), row=1, col=1)
    fig.add_trace(go.Scatter(y=quantile_25_data, mode='lines', name='Rolling 25th Percentile', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(y=quantile_75_data, mode='lines', name='Rolling 75th Percentile', line=dict(color='red')), row=2, col=1)

    # Update layout with titles and axis labels
    fig.update_layout(
        title="Rolling Quantile with Window Size 30 (25th and 75th Percentiles)",
        xaxis_title="Index",
        yaxis=dict(title="Input Data"),
        yaxis2=dict(title="Quantile Values"),
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)        
    )

    fig.show()
```

<!-- HELP_END -->

## Implementation Details

### Algorithm

`RollingQuantile` keeps the window in an order statistic tree, a sorted tree that can return its k-th smallest element. Each new value is inserted and the value leaving the window is removed, then the two order statistics around the quantile position are read from the tree.

### Complexity

The Order Statistic Tree provides the following complexities:

* **Time Complexity**: 
  - Insertion and deletion operations are `O(log(window_size))`.
  - Quantile selection is `O(log(window_size))` due to the tree’s ordered structure.
  - The overall time complexity per element is therefore `O(log(window_size))`.

* **Space Complexity**: 
  - `O(window_size)`, as only elements within the current window are stored in the tree at any given time.