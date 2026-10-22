---
name: RollingKurt
title: Rolling kurtosis
implementation_family: rolling
topics:
- statistics
tags:
- kurtosis
- moment
- rolling
- excess kurtosis
- fat tails
- tail risk
short: Trailing-window excess kurtosis.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 1
  description: Number of most recent samples in the window. The estimator needs at least 4
    samples, so smaller windows give NaN.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before the window is full: "strict" gives NaN, "expanding" computes
    from the samples seen so far, and "zero" treats the window as pre-filled with zeros.'
see_also:
- RollingSkew
- RollingVar
- EwKurt
- ExpandingKurt
- RollingRealizedKurt
nan_policy: ignore
---

# `RollingKurt`

## Description

`RollingKurt` computes the excess kurtosis of the `window_size` most recent values,
using the bias-corrected Fisher-Pearson estimator (the same formula as
`pandas.Series.rolling(w).kurt()`). A normal distribution has excess kurtosis zero;
positive values indicate heavier tails. The estimator needs at least 4 samples, so the
output is NaN until 4 samples are in the window.

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
    from screamer import RollingKurt

    np.random.seed(0)  # reproducible example data

    # Generate example data
    data = np.cumsum(np.random.normal(size=300))

    # Create subplots with specified row heights and shared x-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[2/3, 1/3],
        vertical_spacing=0.1
    )

    # Add traces for each subplot
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input Data'), row=1, col=1)
    fig.add_trace(go.Scatter(y=RollingKurt(30)(data), mode='lines', name='Rolling Kurtosis', line=dict(color='red')), row=2, col=1)

    # Update layout with titles and axis labels
    fig.update_layout(
        title=f"Rolling Kurtosis with Window Size 30",
        xaxis_title="Index",
        yaxis=dict(title="Input Data"),
        yaxis2=dict(title="Rolling Kurtosis", range=[-2, 4]),
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)        
    )

    fig.show()
```

<!-- HELP_END -->

## Implementation Details

### Algorithm

`RollingKurt` keeps the window in a circular buffer with four running sums, of `d`,
`d²`, `d³` and `d⁴` for `d = x - c`, where the origin `c` is close to the window mean,
and computes the kurtosis from them in closed form (see
[Accuracy at large levels](../conventions.md#accuracy-at-large-levels)).

### Complexity

* **Time complexity**: `O(1)` per new element.
* **Space complexity**: `O(window_size)`.

See [Performance](../performance.md) for benchmarks.
