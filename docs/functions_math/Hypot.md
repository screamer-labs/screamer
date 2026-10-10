---
name: Hypot
title: Hypotenuse
implementation_family: math
topics:
- trig
tags:
- hypotenuse
- norm
- polar
- pair
- euclidean norm
- distance
- magnitude
short: Euclidean distance sqrt(x^2 + y^2), numerically stable.
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- Cart2Polar
- Atan2
- Sqrt
---

# `Hypot`

## Description

`Hypot` computes the Euclidean distance `sqrt(x² + y²)` of two inputs. The computation avoids the overflow that squaring a very large `|x|` or `|y|` would cause, and the underflow for very small ones.

It takes two inputs and returns one output. For 2D arrays the inputs are paired column by column.

*Equation*:

$$
y[t] = \sqrt{x_1[t]^2 + x_2[t]^2}
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
    from screamer import Hypot

    rng = np.random.default_rng(0)
    N = 200
    x = np.cumsum(rng.standard_normal(N))
    y = np.cumsum(rng.standard_normal(N))
    r = Hypot()(x, y)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=x, mode='lines', name='x', opacity=0.7))
    fig.add_trace(go.Scatter(y=y, mode='lines', name='y', opacity=0.7))
    fig.add_trace(go.Scatter(y=r, mode='lines', name='Hypot(x, y)', line=dict(color='red')))
    fig.update_layout(
        title="Hypot: Euclidean length sqrt(x^2 + y^2)",
        xaxis_title="Index", yaxis_title="Value",
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Reference

Equivalent to `numpy.hypot`. Also returned as the radial component of `Cart2Polar`.
