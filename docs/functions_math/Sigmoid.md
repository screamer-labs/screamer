---
name: Sigmoid
title: Sigmoid
implementation_family: math
topics:
- activations
tags:
- activation
- sigmoid
- logistic
- logistic function
- squash
short: 'Logistic sigmoid: 1 / (1 + exp(-x)).'
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- Tanh
- Softsign
- Erf
---

# `Sigmoid`

## Description

`Sigmoid` applies the logistic function to each element:

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

The output lies in the open interval `(0, 1)` for all finite inputs.

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
    from screamer import Sigmoid

    # Generate example data
    data = np.linspace(-6, 6, 100)
    sigmoid_data = Sigmoid()(data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Original Data'))
    fig.add_trace(go.Scatter(y=sigmoid_data, mode='lines', name='Sigmoid Output', line=dict(color='red')))

    fig.update_layout(
        title="Sigmoid Transformation",
        yaxis_title="Output",
        xaxis_title="Input",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    fig.show()
```

<!-- HELP_END -->

