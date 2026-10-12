---
name: RollingPoly1
title: Rolling 1st-order polynomial fit
implementation_family: rolling
topics:
- trend
- regression
tags:
- regression
- linear
- polynomial
- rolling
- linear regression
- slope
- trend
- savitzky-golay
short: OLS fit y = a + b*t over a trailing window. derivative_order selects value/slope/zero.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of most recent samples in the window.
- name: derivative_order
  type: int
  default: 0
  enum:
  - 0
  - 1
  description: 'What to return from the fitted line: 0 gives the fitted value at the newest
    sample, 1 the slope per sample.'
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
- RollingPoly2
- RollingLinearRegression
- RollingTSF
- ExpandingSlope
references:
- Savitzky, A. and Golay, M. J. E. (1964). Smoothing and differentiation of data by simplified
  least squares procedures. *Analytical Chemistry*, 36(8), 1627-1639. doi:10.1021/ac60214a047
---

# `RollingPoly1`

## Description

`RollingPoly1` fits a straight line by least squares to the `window_size` most recent values and returns either the line's value at the newest sample (`derivative_order=0`) or its slope (`derivative_order=1`). This is a causal Savitzky-Golay filter: the value output smooths the series, and the slope output measures its rate of change per sample.


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
    from screamer import RollingPoly1

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

    # RollingPoly1 for endpoint (derivative_order=0)
    endpoint_data = RollingPoly1(window_size=30, derivative_order=0)(data)

    # RollingPoly1 for slope (derivative_order=1)
    slope_data = RollingPoly1(window_size=30, derivative_order=1)(data)

    # Add traces for input data and results with different derivative orders
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input Data'), row=1, col=1)
    fig.add_trace(go.Scatter(y=endpoint_data, mode='lines', name='Rolling Endpoint (Order 0)', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(y=slope_data, mode='lines', name='Rolling Slope (Order 1)', line=dict(color='green')), row=2, col=1)

    # Update layout with titles and axis labels
    fig.update_layout(
        title="RollingPoly1 with Window Size 30",
        xaxis2_title="Index",
        yaxis=dict(title="Input Data"),
        yaxis2=dict(title="RollingPoly1 Output"),
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)        
    )

    fig.show()
```

<!-- HELP_END -->

## Implementation Details

### Algorithm

`RollingPoly1` keeps running sums over the window, with a circular buffer holding the values that will leave it, and solves the least-squares line from them at each step, so the cost does not depend on `window_size`. The sums are taken about an origin near the window mean, so the fit stays accurate when the values are large next to their spread, as prices are (see [Accuracy at large levels](../conventions.md#accuracy-at-large-levels)).

### Complexity

* **Time Complexity**: O(1) per element.
* **Space Complexity**: O(window_size), since only elements within the current window are stored.

For timings see [Performance](../performance.md).
