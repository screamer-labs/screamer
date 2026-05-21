---
name: Acos
title: Arccosine
implementation_family: math
topics:
- math
tags:
- trigonometry
- acos
- inverse
short: Inverse cosine of each element (radians, input in [-1, 1]).
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `Acos`

## Description

Inverse cosine of each element. Output is in the range $[0, \pi]$ when the input is in $[-1, 1]$.

*Equation*:

$$
y[i] = \arccos(x[i])
$$

*Parameters*: `Acos` takes no parameters.

*NaN handling*: Inputs outside $[-1, 1]$ produce `NaN`. Existing `NaN` values pass through.


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
    from screamer import Acos

    np.random.seed(0)
    data = np.clip(np.random.normal(size=80) * 0.6, -1.0, 1.0)
    transformed = Acos()(data)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.1)
    fig.add_trace(go.Scatter(y=data, mode="lines+markers",
                             name="input"), row=1, col=1)
    fig.add_trace(go.Scatter(y=transformed, mode="lines+markers",
                             name="Acos",
                             line=dict(color="red")), row=2, col=1)
    fig.update_layout(
        title="Acos transformation",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="input (clipped to [-1, 1])", row=1, col=1)
    fig.update_yaxes(title_text="Acos(input)", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->

## Reference

Equivalent to `numpy.arccos`.
