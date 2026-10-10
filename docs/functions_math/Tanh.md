---
name: Tanh
title: Tanh
implementation_family: math
topics:
- trig
- activations
tags:
- activation
- tanh
- hyperbolic tangent
- squash
short: Hyperbolic tangent.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- Sigmoid
- Softsign
---

# `Tanh`

## Description

`Tanh` applies the hyperbolic tangent to each element, mapping the real line to the range $(-1, 1)$. Its output is centered on zero.

*Equation*:

$$
f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
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
    from screamer import Tanh

    # Generate example data
    data = np.linspace(-3, 3, 100)
    tanh_data = Tanh()(data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Original Data'))
    fig.add_trace(go.Scatter(y=tanh_data, mode='lines', name='Tanh Output', line=dict(color='blue')))

    fig.update_layout(
        title="Tanh Transformation",
        yaxis_title="Output",
        xaxis_title="Input",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    fig.show()
```

<!-- HELP_END -->

