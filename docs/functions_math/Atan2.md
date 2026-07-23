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
short: Signed angle of (x, y) from the positive x-axis (numpy.arctan2 order).
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
---

# `Atan2`

## Description

Two-argument inverse tangent: the signed angle of the point `(x, y)` from the positive x-axis, in the range $(-\pi, \pi]$. The argument order matches `numpy.arctan2`: **`y` first, `x` second**.

This is a **2-input, 1-output** function (`FunctorBase<_, 2, 1>`). Inputs are paired column-by-column for arrays.

Unlike `Atan`, `Atan2` returns the *correct quadrant* because it sees both `x` and `y` separately. `Atan(y / x)` collapses two cases that `Atan2(y, x)` keeps distinct.

*Equation*:

$$
\theta[t] = \operatorname{atan2}(y[t], x[t])
$$

*Parameters*: `Atan2` takes no parameters.

*NaN handling*: A NaN in either input produces a NaN output.


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
