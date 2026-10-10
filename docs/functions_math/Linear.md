---
name: Linear
title: Linear (affine)
implementation_family: math
topics:
- arithmetic
- standardization
tags:
- affine
- linear
- scale
- rescale
- offset
short: 'Affine transform: scale * x + shift.'
inputs: 1
outputs: 1
parameters:
- name: scale
  type: float
  default: 1.0
  description: Factor that multiplies each input value.
- name: shift
  type: float
  default: 0.0
  description: Constant added after scaling.
nan_policy: ignore
see_also:
- Linear2
- Add
- Mul
---

# `Linear`

## Description

`Linear` applies an affine transformation to each input value.

*Equation*:

$$
f(x) = \text{scale} \cdot x + \text{shift}
$$


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
    from screamer import Linear

    # Generate example data
    data = np.linspace(-10, 10, 100)

    # Create a Linear transformation with scale=2 and shift=3
    linear = Linear(scale=2, shift=3)
    linear_data = linear(data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Original Data'))
    fig.add_trace(go.Scatter(y=linear_data, mode='lines', name='Linear Transformation (scale=2, shift=3)', line=dict(color='magenta')))

    fig.update_layout(
        title="Linear Transformation",
        yaxis_title="Output",
        xaxis_title="Input",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    fig.show()
```

<!-- HELP_END -->

