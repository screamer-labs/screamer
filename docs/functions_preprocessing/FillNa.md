---
name: FillNa
title: Replace NaNs with a constant
implementation_family: preprocessing
topics:
- missing-data
tags:
- nan
- missing data
- fill
- fillna
- impute
- replace nan
- constant fill
short: Replace NaN with a user-specified scalar.
inputs: 1
outputs: 1
parameters:
- name: fill
  type: float
  default: 0.0
  description: Constant that replaces each NaN. Any float, inf and -inf included; NaN leaves each NaN in place.
nan_policy: nan-aware
see_also:
- Ffill
- IsNan
- Where
---

# `FillNa`

## Description

`FillNa` replaces every `NaN` with the constant `fill`, like pandas
`fillna(value)`, so the output contains no `NaN`. Other values pass through
unchanged.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `nan-aware`.** This function is built to process `NaN` inputs, as the Description explains. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import FillNa

    np.random.seed(0)  # reproducible example data

    # Generate example data with NaN values
    data = np.random.normal(size=30)
    data[[3, 6, 10, 15, 21]] = np.nan  # Introduce NaN values

    # Apply forward fill
    filled_data = FillNa(0)(data)

    # Create subplots with specified row heights and shared x-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[1/2, 1/2],
        vertical_spacing=0.1
    )

    # Add traces for original data and filled data
    fig.add_trace(go.Scatter(y=data, mode='lines+markers', name='Original Data'), row=1, col=1)
    fig.add_trace(go.Scatter(y=filled_data, mode='lines+markers', name='Filled Data', line=dict(color='red')), row=2, col=1)
    
    # Update layout with individual y-axis titles
    fig.update_layout(
        title="FillNa(0.0) on Data with NaNs",
        xaxis_title="Index",
        yaxis_title="Original Data",
        yaxis2_title="Filled Data",
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)        
    )

    fig.show()
```

<!-- HELP_END -->

## Implementation Details

### Algorithm for `FillNa`

The `FillNa` function checks each data point for `NaN` values and substitutes any found with the specified `fill` value, providing a straightforward and efficient method to eliminate `NaN`s from the data.

### Complexity


* **Time Complexity**: O(1)
* **Space Complexity**: O(1)


### Performance

`FillNa` is a lightweight operation that processes data efficiently. It is suitable for real-time or streaming data applications where missing values need to be managed with minimal overhead.
