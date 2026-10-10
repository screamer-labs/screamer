---
name: Power
title: Power
implementation_family: math
topics:
- arithmetic
tags:
- power
- pow
- exponent
- x**p
short: x raised to a fixed exponent p.
inputs: 1
outputs: 1
parameters:
- name: p
  type: float
  default: 2.0
  description: Exponent applied to each element. The default 2 gives the same result as Square,
    which computes x * x directly.
nan_policy: ignore
see_also:
- Square
- Cube
- Sqrt
- Exp
---

# `Power`

## Description

`Power` raises each element to the fixed exponent `p`. A negative input with a non-integer `p` gives `NaN`.


*Equation*:

$$
f(x) = x^p 
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
    from plotly.subplots import make_subplots
    from screamer import Power

    np.random.seed(0)  # reproducible example data

    data = np.random.normal(size=30)
    result = Power(2)(data)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[1/2, 1/2],
        vertical_spacing=0.1
    )

    fig.add_trace(go.Scatter(y=data, mode='lines+markers', name='Original Data'), row=1, col=1)
    fig.add_trace(go.Scatter(y=result, mode='lines+markers', name='Power of 2, square', line=dict(color='red')), row=2, col=1)

    fig.update_layout(
        title="Power 2, square",
        xaxis_title="Index",
        yaxis_title="Original Data",
        yaxis2_title="Square",
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)        
    )

    fig.show()
```

<!-- HELP_END -->

