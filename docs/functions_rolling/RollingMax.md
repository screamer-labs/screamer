---
name: RollingMax
title: Rolling maximum
implementation_family: rolling
topics:
- statistics
tags:
- max
- rolling
short: Trailing-window maximum (monotonic deque).
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Trailing-window length.
nan_policy: ignore
---

# `RollingMax`

## Description
The `RollingMax` class computes the maximum value within a moving window of specified size over a sequence of data. 


*Initial values*: The constructor requires a positive integer `window_size` parameter to define the rolling window.  
*NaN handling*: NaN values are not handled natively and should be preprocessed if necessary.


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
    from screamer import RollingMax

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

### ALgorithm

`RollingMax` used the `ascending minima algorithm` using a deque-based that ensures that each new maximum is calculated in `O(1)` constant time while using `O(window_size)` memory. 

### Complexity:

* Time complexity: `O(1)`
* Space complexity: `O(window_size)`

### Performance

* Short streams (n=1.000): 300% faster than `Pandas Rolling max`
* Longer streams (n=1.000.000): 90% faster than `Pandas Rolling max`
