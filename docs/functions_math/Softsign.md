---
name: Softsign
title: Softsign
implementation_family: math
topics:
- activations
tags:
- activation
- softsign
- squash
short: 'Softsign: x / (1 + |x|).'
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- Tanh
- Sigmoid
---

# `Softsign`

## Description

`Softsign` applies the mapping `x / (1 + |x|)` to each element:

$$
f(x) = \frac{x}{1 + |x|}
$$

The output approaches -1 and +1 as x goes to negative and positive infinity.

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
    from screamer import Softsign

    # Generate example data
    data = np.linspace(-5, 5, 100)
    softsign_data = Softsign()(data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Original Data'))
    fig.add_trace(go.Scatter(y=softsign_data, mode='lines', name='Softsign Output', line=dict(color='brown')))

    fig.update_layout(
        title="Softsign Transformation",
        yaxis_title="Output",
        xaxis_title="Input",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    fig.show()
```

<!-- HELP_END -->

