---
name: Abs
title: Absolute value
implementation_family: math
topics:
- arithmetic
tags:
- abs
- magnitude
- absolute value
- fabs
short: Absolute value of each element.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- Sign
- Square
- PosPart
- NegPart
---

# `Abs`

## Description

`Abs` computes the absolute value of each element, so negative values become positive.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import Abs

    np.random.seed(0)  # reproducible example data

    # Generate example data with negative values
    data = np.random.normal(size=100) 
    abs_data = Abs()(data)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[1/2, 1/2],
        vertical_spacing=0.1
    )

    fig.add_trace(go.Scatter(y=data, mode='lines+markers', name='Original Data'), row=1, col=1)
    fig.add_trace(go.Scatter(y=abs_data, mode='lines+markers', name='Absolute Value', line=dict(color='red')), row=2, col=1)

    fig.update_layout(
        title="Absolute Value Transformation (Abs)",
        yaxis=dict(title="Data", range=[-3, 3]),
        yaxis2=dict(title="Output", range=[-3, 3]),
        xaxis_title="Index",
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)        
    )

    fig.show()

```

<!-- HELP_END -->

