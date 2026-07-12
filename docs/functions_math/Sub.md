---
name: Sub
title: Sub
implementation_family: math
topics:
- arithmetic
tags:
- arithmetic
- binary
short: Elementwise difference of two aligned streams (x - y).
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
---

# `Sub`

## Description

`Sub` computes the elementwise difference of two aligned input streams, `x - y`. It takes two inputs and returns one output; a `NaN` in
either input yields `NaN` at that step.

*Parameters*: `Sub` takes no parameters.


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
    from screamer import Sub

    np.random.seed(0)
    x = np.cumsum(np.random.normal(size=200))
    y = np.cumsum(np.random.normal(size=200))
    diff = Sub()(x, y)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=x, mode="lines", name="x"), row=1, col=1)
    fig.add_trace(go.Scatter(y=y, mode="lines", name="y"), row=1, col=1)
    fig.add_trace(go.Scatter(y=diff, mode="lines", name="x - y", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Elementwise difference (Sub)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="inputs", row=1, col=1)
    fig.update_yaxes(title_text="x - y", row=2, col=1)
    fig.show()
```
