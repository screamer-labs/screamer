---
name: Polar2Cart
title: Polar to Cartesian
implementation_family: math
topics:
- trig
tags:
- polar
- conversion
- pair
- cartesian
- sine
- cosine
short: Convert (r, theta) to (x, y).
inputs: 2
outputs: 2
output_names:
- x
- y
parameters: []
nan_policy: ignore
see_also:
- Cart2Polar
- Cos
- Sin
---

# `Polar2Cart`

## Description

Convert 2D polar coordinates `(r, θ)` to Cartesian coordinates `(x, y)`. The inverse of `Cart2Polar`.

It takes two inputs, `r` then `theta`, and returns two outputs, `x` then `y`. For arrays the inputs are paired column by column and the outputs are stacked along a trailing axis of size 2.

*Equation*:

$$
x[t] = r[t] \cos\theta[t], \qquad y[t] = r[t] \sin\theta[t]
$$

A `NaN` in either input gives `NaN` in both outputs.

## Output shapes

Same shape rule as `Cart2Polar`:

| You pass... | You get back... |
|---|---|
| `Polar2Cart()(r, θ)` (two scalars) | tuple `(x, y)` |
| Two 1D arrays of shape `(T,)` | array of shape `(T, 2)` |
| Two 2D arrays of shape `(T, K)` | array of shape `(T, K, 2)` |

`out[..., 0]` is `x`, `out[..., 1]` is `y`.

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
    from screamer import Polar2Cart

    N = 200
    theta = np.linspace(0, 2 * np.pi, N)
    r = np.ones(N)
    out = Polar2Cart()(r, theta)   # shape (N, 2); out[:,0] = x, out[:,1] = y
    cart_x = out[:, 0]
    cart_y = out[:, 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cart_x, y=cart_y, mode='lines', name='unit circle (r=1)'))
    fig.update_layout(
        title="Polar2Cart: unit circle traced by sweeping theta from 0 to 2*pi",
        xaxis_title="x", yaxis_title="y",
        xaxis=dict(scaleanchor="y"),
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Reference

The two outputs are `r * numpy.cos(theta)` and `r * numpy.sin(theta)`, computed in one pass.
