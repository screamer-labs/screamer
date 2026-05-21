---
name: Round
title: Round (banker's rounding)
implementation_family: math
topics:
- math
tags:
- round
- rounding
- banker
short: Round each element to the nearest integer (half-to-even).
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `Round`

## Description

Round each element to the nearest integer using *round-half-to-even* (banker's rounding). Halves go to the nearest even integer rather than always away from zero, which matches `numpy.round` and Python's built-in `round`.

*Equation*:

$$
y[i] = \operatorname{round}(x[i])
$$

with halves resolved to even.

*Parameters*: `Round` takes no parameters.

*NaN handling*: `NaN` values pass through unchanged.

*Note*: This is **not** the same as the C++ `std::round` (half-away-from-zero). For example `Round(0.5) == 0`, `Round(1.5) == 2`, `Round(2.5) == 2`.


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
    from screamer import Round

    np.random.seed(0)
    data = np.random.normal(size=80) * 1.5
    transformed = Round()(data)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.1)
    fig.add_trace(go.Scatter(y=data, mode="lines+markers",
                             name="input"), row=1, col=1)
    fig.add_trace(go.Scatter(y=transformed, mode="lines+markers",
                             name="Round",
                             line=dict(color="red")), row=2, col=1)
    fig.update_layout(
        title="Round transformation (banker's rounding)",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="input", row=1, col=1)
    fig.update_yaxes(title_text="Round(input)", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->

## Reference

Equivalent to `numpy.round` and Python's built-in `round`.
