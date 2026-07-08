---
name: PosPart
title: Positive Part
implementation_family: math
topics:
- arithmetic
tags:
- positive
- part
- relu
short: 'Positive part of x: max(x, 0).'
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `PosPart`

## Description

The `PosPart` class computes the positive part of a value, returning `max(x, 0)`. Together with `NegPart`, it decomposes any real number: `x = PosPart(x) - NegPart(x)`. This is identical to the ReLU activation function and is useful as a building block for signed-quantity aggregations such as buy/sell volume.

*Equation*:

$$
f(x) = \max(x, 0)
$$

*Parameters*: `PosPart` takes no parameters.

*NaN handling*: `NaN` values pass through unchanged.


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
    from screamer import PosPart, NegPart

    data = np.linspace(-3, 3, 100)
    pos = PosPart()(data)
    neg = NegPart()(data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input'))
    fig.add_trace(go.Scatter(y=pos, mode='lines', name='PosPart', line=dict(color='green')))
    fig.add_trace(go.Scatter(y=neg, mode='lines', name='NegPart', line=dict(color='red')))

    fig.update_layout(
        title="PosPart and NegPart Decomposition",
        yaxis_title="Output",
        xaxis_title="Input",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    fig.show()
```

<!-- HELP_END -->
