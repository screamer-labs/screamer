---
name: RollingMean
title: Rolling mean
implementation_family: rolling
topics:
- smoothing
- statistics
tags:
- sma
- mean
- moving-average
- simple moving average
- rolling mean
- average
short: Trailing-window arithmetic mean (simple moving average).
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 1
  description: Number of most recent samples in the window.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before the window is full: "strict" gives NaN, "expanding" computes
    from the samples seen so far, and "zero" treats the window as pre-filled with zeros.'
nan_policy: ignore
see_also:
- EwMean
- ExpandingMean
- WMA
- RollingSum
- RollingMedian
---

# `RollingMean`

## Description

`RollingMean` computes the arithmetic mean of the `window_size` most recent values (also called a simple moving average).


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
    from screamer import RollingMean

    np.random.seed(0)  # reproducible example data

    # Generate example data
    data = np.cumsum(np.random.normal(size=300))


    # Plotting with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input Data'))
    fig.add_trace(go.Scatter(y=RollingMean(10)(data), mode='lines', name='Rolling Mean 10', line=dict(color='red')))
    fig.add_trace(go.Scatter(y=RollingMean(60)(data), mode='lines', name='Rolling Mean 60', line=dict(color='green')))
    fig.update_layout(title="Rolling mean with window sizes 10 and 60",
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

`RollingMean` keeps a running sum of the window in a circular buffer. Each step adds the new sample to the sum and subtracts the sample that leaves the window.

### Complexity

* **Time Complexity**: `O(1)` per new element.
* **Space Complexity**: `O(window_size)`.


For timings see [Performance](../performance.md).
