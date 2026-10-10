---
name: Atan2
title: Two-argument arctangent
implementation_family: math
topics:
- trig
tags:
- trigonometry
- atan2
- polar
- pair
- arctan2
- angle
short: Signed angle of (x, y) from the positive x-axis (numpy.arctan2 order).
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- Atan
- Hypot
- Cart2Polar
---

# `Atan2`

## Description

Two-argument inverse tangent: the signed angle of the point `(x, y)` from the positive x-axis, in the range $(-\pi, \pi]$. The argument order matches `numpy.arctan2`: **`y` first, `x` second**.

It takes two inputs and returns one output. For 2D arrays the inputs are paired column by column.

`Atan2` resolves the quadrant because it sees the signs of `x` and `y` separately. `Atan(y / x)` gives the same value for `(x, y)` and `(-x, -y)`, which `Atan2(y, x)` keeps distinct.

*Equation*:

$$
\theta[t] = \operatorname{atan2}(y[t], x[t])
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
    from screamer import Atan2

    N = 200
    t = np.linspace(-np.pi, np.pi, N)
    y = np.sin(t)
    x = np.cos(t)
    angle = Atan2()(y, x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=t, mode='lines', name='Angle t (input)', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(y=angle, mode='lines', name='Atan2(sin t, cos t)', line=dict(color='red')))
    fig.update_layout(
        title="Atan2: recovering the angle from (sin t, cos t)",
        xaxis_title="Index", yaxis_title="Radians",
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Reference

Equivalent to `numpy.arctan2`. Also returned as the angular component of `Cart2Polar`.
