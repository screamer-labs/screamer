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
short: Euclidean distance sqrt(x^2 + y^2), numerically stable.
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
---

# `Hypot`

## Description

Two-argument Euclidean distance: `Hypot(x, y) = sqrt(x*x + y*y)`. Computes `sqrt(x² + y²)` in a numerically stable way that avoids overflow for very large `|x|` or `|y|` and underflow for very small ones.

This is a **2-input, 1-output** function (`FunctorBase<_, 2, 1>`). Inputs are paired column-by-column for arrays.

*Equation*:

$$
y[t] = \sqrt{x_1[t]^2 + x_2[t]^2}
$$

*Parameters*: `Hypot` takes no parameters.

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
