---
name: Elu
title: ELU
implementation_family: math
topics:
- activations
tags:
- activation
- elu
- neural
short: Exponential linear unit.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `Elu`

## Description

The `Elu` (Exponential Linear Unit) class introduces an exponential component for negative inputs, providing smooth and continuous output. It helps mitigate the vanishing gradient problem by ensuring a non-zero gradient for negative inputs.

*Equation*:

$$
f(x) = \begin{cases} 
x, & \text{if } x > 0 \\
\alpha (e^x - 1), & \text{if } x \leq 0
\end{cases}
$$

where $\alpha$ is a constant set to 1.0.

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
    from screamer import Elu

    # Generate example data
    data = np.linspace(-3, 3, 100)
    elu_data = Elu()(data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Original Data'))
    fig.add_trace(go.Scatter(y=elu_data, mode='lines', name='ELU Output', line=dict(color='purple')))

    fig.update_layout(
        title="ELU Transformation",
        yaxis_title="Output",
        xaxis_title="Input",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    fig.show()
```

<!-- HELP_END -->

