---
name: Square
title: Square
implementation_family: math
topics:
- arithmetic
tags:
- square
- pow2
short: x squared (faster than Power(2)).
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `Square`

## Description

Square each element. Equivalent to `Power(2)` but skips the `std::pow` logarithm and is therefore faster.

*Equation*:

$$
y[i] = x[i]^2
$$

*Parameters*: `Square` takes no parameters.

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
    from plotly.subplots import make_subplots
    from screamer import Square

    np.random.seed(0)
    data = np.random.normal(size=80) * 1.5
    transformed = Square()(data)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.1)
    fig.add_trace(go.Scatter(y=data, mode="lines+markers",
                             name="input"), row=1, col=1)
    fig.add_trace(go.Scatter(y=transformed, mode="lines+markers",
                             name="Square",
                             line=dict(color="red")), row=2, col=1)
    fig.update_layout(
        title="Square transformation",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="input", row=1, col=1)
    fig.update_yaxes(title_text="Square(input)", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->

## Reference

Equivalent to `numpy.square`.
