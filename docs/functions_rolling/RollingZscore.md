---
name: RollingZscore
title: Rolling z-score
implementation_family: rolling
topics:
- standardization
tags:
- zscore
- standardization
- anomaly
- rolling
- z-score
- standardize
- normalize
short: Latest sample standardized by the trailing-window mean and standard deviation.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
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
- EwZscore
- RollingStd
- RollingMean
- CrossZscore
---

# `RollingZscore`

## Description

`RollingZscore` standardizes each new value against the trailing window:

$$
z[t] = \frac{x[t] - \bar{x}_w[t]}{\text{std}_w[t]}
$$

where `std` is the sample standard deviation (ddof=1) over the `window_size` most recent values.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. plotly::

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import RollingZscore

    np.random.seed(0)  # reproducible example data

    data = np.cumsum(np.random.normal(size=300))

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[2/3, 1/3],
        vertical_spacing=0.1
    )

    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input Data'), row=1, col=1)
    fig.add_trace(go.Scatter(y=RollingZscore(30)(data), mode='lines', name='Rolling Z-score', line=dict(color='red')), row=2, col=1)

    fig.update_layout(
        title="Rolling Z-score with Window Size 30",
        xaxis_title="Index",
        yaxis=dict(title="Input Data"),
        yaxis2=dict(title="Rolling Z-score"),
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)        
    )

    fig.show()
```

<!-- HELP_END -->

## Implementation Details

### Algorithm

`RollingZscore` keeps running sums of the window in circular buffers and updates them as each sample enters and the oldest one leaves.

### Complexity

* **Time Complexity**: `O(1)` per new element.
* **Space Complexity**: `O(window_size)`.

