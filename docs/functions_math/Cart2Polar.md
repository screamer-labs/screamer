---
name: Cart2Polar
title: Cartesian to polar
implementation_family: math
topics:
- geometry
tags:
- polar
- conversion
- pair
short: Convert (x, y) to (r, theta).
inputs: 2
outputs: 2
parameters: []
nan_policy: ignore
---

# `Cart2Polar`

## Description

Convert 2D Cartesian coordinates `(x, y)` to polar coordinates `(r, θ)`. The radial component is the Euclidean distance from the origin and the angular component is the signed angle from the positive x-axis, in the range $(-\pi, \pi]$.

This is a **2-input, 2-output** function (`FunctorBase<_, 2, 2>`). Inputs are paired column-by-column; outputs are stacked along a trailing axis of size 2.

*Equation*:

$$
r[t] = \sqrt{x[t]^2 + y[t]^2}, \qquad \theta[t] = \operatorname{atan2}(y[t], x[t])
$$

*Parameters*: `Cart2Polar` takes no parameters.

*NaN handling*: A NaN in either input produces NaN in both outputs.

## Output shapes

| You pass... | You get back... |
|---|---|
| `Cart2Polar()(x, y)` (two scalars) | tuple `(r, θ)` |
| `Cart2Polar()((x, y))` (one pair) | tuple `(r, θ)` |
| `Cart2Polar()([(x1, y1), (x2, y2), ...])` | list of `(r, θ)` tuples |
| Two 1D arrays of shape `(T,)` | array of shape `(T, 2)` |
| Two 2D arrays of shape `(T, K)` | array of shape `(T, K, 2)` |
| Two parallel iterables | list of `(r, θ)` tuples |

The shape rule combines the multi-input pairing (column-by-column) with the multi-output stacking (extra trailing axis of size 2). `out[..., 0]` is the radius, `out[..., 1]` is the angle.


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
    from plotly.subplots import make_subplots
    from screamer import Cart2Polar

    rng = np.random.default_rng(0)
    n = 200
    u = rng.normal(2.0, 1.0, size=n)   # eastward wind
    v = rng.normal(0.5, 1.0, size=n)   # northward wind

    polar = Cart2Polar()(u, v)
    speed = polar[:, 0]
    direction = polar[:, 1]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.1)
    fig.add_trace(go.Scatter(y=speed, mode='lines',
                             name='Speed (r)', line=dict(color='steelblue')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=direction, mode='lines',
                             name='Direction (θ, radians)', line=dict(color='orange')),
                  row=2, col=1)
    fig.update_layout(
        title="Cart2Polar: Wind (u, v) -> (speed, direction)",
        xaxis_title="Index",
        yaxis_title="Speed",
        yaxis2_title="Direction",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.show()
```

<!-- HELP_END -->

## Roundtrip identity

`Cart2Polar` and `Polar2Cart` are inverses of each other:

```python
back = Polar2Cart()(polar[:, 0], polar[:, 1])
np.testing.assert_allclose(back[:, 0], u, atol=1e-12)
np.testing.assert_allclose(back[:, 1], v, atol=1e-12)
```

## Reference

The two outputs are equivalent to `numpy.hypot(x, y)` and `numpy.arctan2(y, x)` respectively, computed in one pass.
