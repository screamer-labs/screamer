---
name: NegPart
title: Negative Part
implementation_family: math
topics:
- arithmetic
tags:
- negative
- part
- max(-x, 0)
- negative part
short: 'Negative part of x: max(-x, 0).'
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- PosPart
- Abs
---

# `NegPart`

## Description

`NegPart` computes the negative part of each element, `max(-x, 0)`. Together with `PosPart`, it decomposes any real number: `x = PosPart(x) - NegPart(x)`. Use the pair to split a signed quantity into its positive and negative components, for example signed trade volume into buy and sell volume.

*Equation*:

$$
f(x) = \max(-x, 0)
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
