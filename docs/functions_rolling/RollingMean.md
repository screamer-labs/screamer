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
short: Trailing-window arithmetic mean (simple moving average).
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

# `RollingMean`

## Description

`RollingMean` computes the arithmetic mean of the `window_size` most recent values (also called a simple moving average).


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
    from screamer import RollingMean

    # Generate example data
    N = 300
    window_size = 30
    data = np.cumsum(np.random.normal(size=300))


    # Plotting with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input Data'))
    fig.add_trace(go.Scatter(y=RollingMean(10)(data), mode='lines', name='Rolling Mean 10', line=dict(color='red')))
    fig.add_trace(go.Scatter(y=RollingMean(60)(data), mode='lines', name='Rolling Mean 60', line=dict(color='green')))
    fig.update_layout(title=f"Rolling mean with Window Size 10 and 60",
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

`RollingMean` implements a cyclic buffer.

### Complexity

* **Time Complexity**: `O(1))` per new element due to the insertion and deletion operations in the cyclic buffer.
* **Space Complexity**: `O(window_size)`, as only elements within the current window are stored.


### Performance

* Short streams (n=1.000): 450% faster than `Pandas Rolling mean` and 50% faster than a `numpy cumsum` based approach.
* Longer streams (n=1.000.000): 450% faster than `Pandas Rolling mean` and 270% faster than a `numpy cumsum` based approach.
