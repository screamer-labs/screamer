---
name: Clip
title: Clip to bounds
implementation_family: preprocessing
topics:
- arithmetic
- outliers
tags:
- clip
- clamp
- bound
- limit
- winsorize
- cap
- outlier
short: Bound each element below and/or above.
inputs: 1
outputs: 1
parameters:
- name: lower
  type: float|null
  default: null
  description: Smallest allowed value; smaller values are replaced by it. None means no lower
    bound. NaN and -inf also mean no lower bound.
- name: upper
  type: float|null
  default: null
  description: Largest allowed value; larger values are replaced by it. None means no upper
    bound. NaN and inf also mean no upper bound.
nan_policy: ignore
see_also:
- RollingSigmaClip
- CrossWinsorize
- Hampel
- ImpulseClip
---

# `Clip`

## Description

`Clip` limits each value to the range `[lower, upper]`: a value below `lower` is
replaced by `lower`, and a value above `upper` by `upper`. Either bound can be
left as `None` to clip on one side only. Use it to cap extreme values before
they reach a statistic that is sensitive to outliers.

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
    from screamer import Clip

    np.random.seed(0)  # reproducible example data

    # Generate example data
    data = np.random.normal(size=50)

    # Clip data to fall between -1 and 1
    clipped_data = Clip(lower=-1, upper=1)(data)

    # Create plot with input data and clipped data
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines+markers', name='Input Data'))
    fig.add_trace(go.Scatter(y=clipped_data, mode='lines+markers', name='Clipped Data', line=dict(color='red')))
    
    fig.update_layout(
        title="Data Clipping with Bounds (-1, 1)",
        xaxis_title="Index",
        yaxis_title="Value",
        yaxis=dict(range=[-2, 2]),  # Limit y-axis to show the clipping effect clearly
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)        
    )

    fig.show()
```

<!-- HELP_END -->

## Implementation Details

### Algorithm

The `Clip` function performs an element-wise check on each value in the input data:
- If a lower bound (`lower`) is specified, any value below this threshold is replaced with `lower`.
- If an upper bound (`upper`) is specified, any value exceeding this threshold is replaced with `upper`.

This element-wise operation ensures that each data point falls within the specified range, with minimal computational overhead.

### Complexity

* **Time Complexity**: `O(1)`.
* **Space Complexity**: `O(1)`.

### Performance

The `Clip` operation is `O(1)` per sample with no internal state, making it suitable for real-time or streaming data applications.