---
name: Equal
title: Equal
implementation_family: math
topics:
- logic
tags:
- logic
- binary
- comparison
short: Returns 1.0 if a == b, else 0.0. NaN in either input yields NaN.
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
---

# `Equal`

## Description

`Equal` compares two aligned input streams element-wise and outputs `1.0` where `a == b` and `0.0` otherwise.

*NaN handling*: if either input is NaN at step `t`, the output is NaN (NaN is not equal to anything, including itself, by IEEE 754).

*Parameters*: `Equal` takes no parameters.

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import Equal

    np.random.seed(0)
    # a discrete signal that repeatedly revisits whole-number levels
    t = np.linspace(0, 6 * np.pi, 200)
    x = np.round(3 + 2 * np.sin(t) + np.random.normal(0, 0.3, size=200))
    threshold = np.full_like(x, 3.0)        # a constant level, as a stream
    mask = Equal()(x, threshold)            # 1.0 where x == 3.0, else 0.0

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=x, mode="lines", name="x"), row=1, col=1)
    fig.add_trace(go.Scatter(y=threshold, mode="lines", name="level = 3.0",
                             line=dict(dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(y=mask, mode="lines", name="x == level",
                             line=dict(color="red", shape="hv")), row=2, col=1)
    fig.update_layout(title="Exactly on a level (Equal)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="signal", row=1, col=1)
    fig.update_yaxes(title_text="mask (0/1)", range=[-0.1, 1.1], row=2, col=1)
    fig.show()
```
