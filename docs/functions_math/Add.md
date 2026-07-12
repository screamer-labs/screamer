---
name: Add
title: Add
implementation_family: math
topics:
- arithmetic
tags:
- arithmetic
- binary
short: Elementwise sum of two aligned streams (x + y).
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
---

# `Add`

## Description

`Add` computes the elementwise sum of two aligned input streams, `x + y`. It takes two inputs and returns one output; a `NaN` in
either input yields `NaN` at that step.

*Parameters*: `Add` takes no parameters.

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import Add

    np.random.seed(0)
    a = np.cumsum(np.random.normal(size=200))
    b = np.cumsum(np.random.normal(size=200))
    out = Add()(a, b)                       # elementwise a + b

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=a, mode="lines", name="a"), row=1, col=1)
    fig.add_trace(go.Scatter(y=b, mode="lines", name="b"), row=1, col=1)
    fig.add_trace(go.Scatter(y=out, mode="lines", name="a + b",
                             line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Elementwise sum of two signals (Add)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="inputs", row=1, col=1)
    fig.update_yaxes(title_text="a + b", row=2, col=1)
    fig.show()
```


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->
