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
short: 'Affine transform: scale * x + shift.'
inputs: 1
outputs: 1
parameters:
- name: scale
  type: float
  default: 1.0
  description: Multiplicative coefficient.
- name: shift
  type: float
  default: 0.0
  description: Additive offset.
nan_policy: ignore
---

# `Linear`

## Description

The `Linear` class computes a linear transformation of each input value, defined by the equation  $y = \text{scale} \times x + \text{shift} $. This function is fundamental for adjusting the scale and location of data, often used in data preprocessing or custom activation functions in machine learning models.

*Equation*:

$$
f(x) = \text{scale} \cdot x + \text{shift}
$$

*Parameters*:

- `scale` (double): The scaling factor applied to the input.
- `shift` (double): The value added to the scaled input to shift the output.

*NaN handling*: `NaN` values are not modified by this function.


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

