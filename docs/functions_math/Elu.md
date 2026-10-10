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
- exponential linear unit
short: Exponential linear unit.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- Selu
- Relu
- Softsign
references:
- Clevert, D.-A., Unterthiner, T. and Hochreiter, S. (2016). Fast and accurate deep network
  learning by exponential linear units (ELUs). *International Conference on Learning Representations*.
---

# `Elu`

## Description

`Elu` (exponential linear unit) is an activation function that passes positive inputs unchanged and maps negative inputs to `exp(x) - 1`, so negative outputs are bounded below by -1.

*Equation*:

$$
f(x) = \begin{cases} 
x, & \text{if } x > 0 \\
\alpha (e^x - 1), & \text{if } x \leq 0
\end{cases}
$$

where $\alpha$ is a constant set to 1.0.

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

