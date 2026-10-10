---
name: Ffill
title: Forward-fill NaNs
implementation_family: preprocessing
topics:
- missing-data
tags:
- nan
- missing data
- fill
- ffill
- forward fill
- pad
- last observation carried forward
- locf
short: Replace NaN with the most recent non-NaN value.
inputs: 1
outputs: 1
parameters: []
nan_policy: nan-aware
see_also:
- FillNa
- Hold
- Lag
---

# `Ffill`

## Description

`Ffill` replaces each `NaN` with the most recent non-`NaN` value (forward fill),
like pandas `ffill()`. Values before the first non-`NaN` sample stay `NaN`,
because there is nothing to carry forward yet.

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
    from screamer import Ffill

    # Generate example data with NaN values
    data = np.random.default_rng(0).normal(size=30)
    data[[3, 6, 10, 15, 21]] = np.nan  # Introduce NaN values

    # Apply forward fill
    ffilled_data = Ffill()(data)

    # Create subplots with specified row heights and shared x-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[1/2, 1/2],
        vertical_spacing=0.1
    )

    # Add traces for original data and forward-filled data
    fig.add_trace(go.Scatter(y=data, mode='lines+markers', name='Original Data'), row=1, col=1)
    fig.add_trace(go.Scatter(y=ffilled_data, mode='lines+markers', name='Forward-Filled Data', line=dict(color='red')), row=2, col=1)
    
    # Update layout with individual y-axis titles
    fig.update_layout(
        title="Forward Fill (Ffill) on Data with NaNs",
        xaxis_title="Index",
        yaxis_title="Original Data",
        yaxis2_title="Forward-Filled Data",
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)        
    )

    fig.show()

```

<!-- HELP_END -->

## Implementation Details

### Algorithm for `Ffill`

The `Ffill` function iterates over the data, replacing each `NaN` value with the most recent valid observation, if available. This ensures that each missing value is filled in a forward direction, but any `NaN` values at the beginning remain unchanged due to the absence of a prior value.

### Complexity


* **Time Complexity**: O(1).
* **Space Complexity**: O(1).

### Performance

 `Ffill` is a lightweight operation that process data efficiently. They are suitable for real-time or streaming data applications where missing values need to be managed with minimal overhead.
