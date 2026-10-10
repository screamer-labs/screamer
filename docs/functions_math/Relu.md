---
name: Relu
title: ReLU
implementation_family: math
topics:
- activations
tags:
- activation
- relu
- neural
- rectifier
- rectified linear unit
- max(0, x)
short: 'Rectified linear unit: max(0, x).'
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- PosPart
- Elu
- Selu
---

# `Relu`

## Description

`Relu` (rectified linear unit) computes `max(0, x)` for each element: positive values pass unchanged and negative values become zero. It gives the same result as `PosPart`.

*Equation*:

$$
f(x) = \max(0, x)
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
    from screamer import Relu

    # Generate example data with negative and positive values
    data = np.linspace(-3, 3, 100)
    relu_data = Relu()(data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Original Data'))
    fig.add_trace(go.Scatter(y=relu_data, mode='lines', name='ReLU Output', line=dict(color='green')))

    fig.update_layout(
        title="Relu Transformation",
        yaxis_title="Output",
        xaxis_title="Input",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    fig.show()
```

<!-- HELP_END -->

