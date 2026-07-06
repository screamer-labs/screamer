---
name: RollingStd
title: Rolling standard deviation
implementation_family: rolling
topics:
- statistics
- volatility
tags:
- std
- volatility
- rolling
short: Trailing-window sample standard deviation (ddof=1).
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Trailing-window length.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: Warmup behaviour.
nan_policy: ignore
---

# `RollingStd`

## Description

The `RollingStd` class computes the sample standard deviation of values within a moving window, quantifying the typical spread or dispersion from the mean. It’s useful for tracking variability in the data over a localized window.

*Parameters*: 
- **`window_size`**: Specifies the size of the rolling window.
- **`start_policy`**: Defines how the function handles the initial phase when fewer than `window_size` data points are available. This parameter accepts one of the following three values:
  - `"strict"`: Returns `NaN` for all calculations until `window_size` elements have been processed.
  - `"expanding"`: Adapts the computation by dynamically reducing the window size to include all available data, starting from a single point and growing until `window_size` is reached.
  - `"zero"`: Simulates a full initial window of zeros, effectively pre-filling the data stream with `window_size` zeros before processing the actual input.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. plotly::

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import RollingStd

    data = np.cumsum(np.random.normal(size=300))

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[2/3, 1/3],
        vertical_spacing=0.1
    )

    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input Data'), row=1, col=1)
    fig.add_trace(go.Scatter(y=RollingStd(30)(data), mode='lines', name='Rolling Std Deviation', line=dict(color='blue')), row=2, col=1)

    fig.update_layout(
        title="Rolling Standard Deviation with Window Size 30",
        xaxis_title="Index",
        yaxis=dict(title="Input Data"),
        yaxis2=dict(title="Rolling Std Deviation"),
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)        
    )

    fig.show()
```

<!-- HELP_END -->

## Implementation Details

### Algorithm

`RollingStd` implements cyclic buffers to accumulate windowed statistics.

### Complexity

* **Time Complexity**: `O(log(1))` per new element due to the insertion and deletion operations in the heaps.
* **Space Complexity**: `O(window_size)`, as only elements within the current window are stored.

