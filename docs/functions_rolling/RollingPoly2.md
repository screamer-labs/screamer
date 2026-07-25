---
name: RollingPoly2
title: Rolling 2nd-order polynomial fit
implementation_family: rolling
topics:
- smoothing
- trend
tags:
- regression
- quadratic
- polynomial
- rolling
short: OLS fit y = a + b*t + c*t^2 over a trailing window.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Trailing-window length.
- name: derivative_order
  type: int
  default: 0
  enum:
  - 0
  - 1
  - 2
  description: 0 = value, 1 = first derivative, 2 = second derivative.
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

# `RollingPoly2`

## Description

`RollingPoly2` fits a degree-2 polynomial `y = a*t^2 + b*t + c` to the `window_size` most recent values by ordinary least squares, then returns a quantity evaluated at the trailing endpoint. `derivative_order` selects what is returned: `0` gives the fitted value at the endpoint, `1` gives the first derivative (slope), and `2` gives the second derivative (curvature).


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
    from plotly.subplots import make_subplots
    from screamer import RollingPoly2

    # Generate example data
    data = np.cumsum(np.random.normal(size=300))

    # Create subplots with specified row heights and shared x-axis
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[1/3, 1/3, 1/3],
        vertical_spacing=0.1
    )

    # RollingPoly2 for endpoint (derivative_order=0)
    endpoint_data = RollingPoly2(window_size=30, derivative_order=0)(data)

    # RollingPoly2 for slope (derivative_order=1)
    slope_data = RollingPoly2(window_size=30, derivative_order=1)(data)

    # RollingPoly2 for curvature (derivative_order=2)
    curvature_data = RollingPoly2(window_size=30, derivative_order=2)(data)

    # Add traces for input data and results with different derivative orders
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input Data'), row=1, col=1)
    fig.add_trace(go.Scatter(y=endpoint_data, mode='lines', name='Rolling Endpoint (Order 0)', line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(y=slope_data, mode='lines', name='Rolling Slope (Order 1)', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(y=curvature_data, mode='lines', name='Rolling Curvature (Order 2)', line=dict(color='purple')), row=3, col=1)

    # Update layout with titles and axis labels
    fig.update_layout(
        title="RollingPoly2 with Window Size 30",
        xaxis_title="Index",
        yaxis=dict(title="Input Data and endpoint"),
        yaxis2=dict(title="RollingPoly2 Slope"),
        yaxis3=dict(title="RollingPoly2 Curvature"),
        margin=dict(l=20, r=20, t=120, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)        
    )

    fig.show()
```

<!-- HELP_END -->

## Implementation Details

### Algorithm

`RollingPoly2` uses least-squares regression to fit a quadratic function within each window. The class allows returning either the endpoint value, the slope, or the curvature at the endpoint, based on `derivative_order`. This fitting method effectively captures both linear and non-linear trends, making it suitable for applications needing insights into the rate of change and concavity within the data.

### Complexity

* **Time Complexity**: O(1) per element.
* **Space Complexity**: O(window_size), since only elements within the current window are stored.

### Performance

* `RollingPoly2` is hundreds times faster than `Pandas rolling apply`, and the speed is constant, indepenent of the windows size.