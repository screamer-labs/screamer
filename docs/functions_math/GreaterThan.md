---
name: GreaterThan
title: GreaterThan
implementation_family: math
topics:
- logic
tags:
- logic
- binary
- comparison
short: Returns 1.0 if a > b, else 0.0. NaN in either input yields NaN.
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
---

# `GreaterThan`

## Description

`GreaterThan` compares two aligned input streams element-wise and outputs `1.0` where `a > b` and `0.0` otherwise. The output is a floating-point mask suitable for use with `Where` or `Filter`.

*NaN handling*: if either input is NaN at step `t`, the output is NaN (an unknown comparison cannot yield a definite mask value).

*Parameters*: `GreaterThan` takes no parameters.

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import GreaterThan

    np.random.seed(1)
    x = np.cumsum(np.random.normal(size=200))
    threshold = np.full_like(x, 2.0)        # a constant threshold, as a stream
    above = GreaterThan()(x, threshold)     # 1.0 where x > threshold, else 0.0

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=x, mode="lines", name="x"), row=1, col=1)
    fig.add_trace(go.Scatter(y=threshold, mode="lines", name="threshold = 2.0",
                             line=dict(dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(y=above, mode="lines", name="x > threshold",
                             line=dict(color="red", shape="hv")), row=2, col=1)
    fig.update_layout(title="Threshold crossing (GreaterThan)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="signal", row=1, col=1)
    fig.update_yaxes(title_text="mask (0/1)", range=[-0.1, 1.1], row=2, col=1)
    fig.show()
```
