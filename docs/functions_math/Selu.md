---
name: Selu
title: SELU
implementation_family: math
topics:
- activations
tags:
- activation
- selu
- neural
- scaled exponential linear unit
short: Scaled exponential linear unit (self-normalizing networks).
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- Elu
- Relu
references:
- Klambauer, G., Unterthiner, T., Mayr, A. and Hochreiter, S. (2017). Self-normalizing neural
  networks. *Advances in Neural Information Processing Systems*, 30.
---

# `Selu`

## Description

`Selu` (scaled exponential linear unit) is the ELU function scaled by fixed coefficients. These constants make the activations of a network self-normalizing (Klambauer et al., 2017).

*Equation*:

$$
f(x) = \lambda \times \begin{cases} 
x, & \text{if } x > 0 \\
\alpha (e^x - 1), & \text{if } x \leq 0
\end{cases}
$$

where $\lambda \approx 1.0507$ and $\alpha \approx 1.67326$.

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
    from screamer import Selu

    # Generate example data
    data = np.linspace(-3, 3, 100)
    selu_data = Selu()(data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Original Data'))
    fig.add_trace(go.Scatter(y=selu_data, mode='lines', name='SELU Output', line=dict(color='orange')))

    fig.update_layout(
        title="Selu Transformation",
        yaxis_title="Output",
        xaxis_title="Input",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    fig.show()
```

<!-- HELP_END -->

