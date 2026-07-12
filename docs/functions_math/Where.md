---
name: Where
title: Where
implementation_family: math
topics:
- logic
tags:
- logic
- ternary
- conditional
short: Returns a if mask is nonzero, b otherwise. NaN mask yields NaN.
inputs: 3
outputs: 1
parameters: []
nan_policy: nan-aware
---

# `Where`

## Description

`Where` is a conditional element-wise selector over three aligned input streams. Given inputs `mask`, `a`, and `b`:

- output is `a` when `mask` is nonzero (any nonzero value counts as true)
- output is `b` when `mask` is zero

If `mask` is NaN, the output is NaN. If the selected branch (`a` when mask is nonzero, `b` when zero) is NaN, that NaN passes through as the output.

*Parameters*: `Where` takes no parameters.

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import Where, GreaterEqual

    np.random.seed(0)
    a = np.cumsum(np.random.normal(size=200))
    b = np.cumsum(np.random.normal(size=200))
    cond = GreaterEqual()(a, b)             # 1.0 where a >= b, else 0.0
    out = Where()(cond, a, b)               # pick a where cond is nonzero, else b

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=a, mode="lines", name="a"), row=1, col=1)
    fig.add_trace(go.Scatter(y=b, mode="lines", name="b"), row=1, col=1)
    fig.add_trace(go.Scatter(y=out, mode="lines", name="a if a >= b else b",
                             line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Pick the larger of two signals (Where)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="inputs", row=1, col=1)
    fig.update_yaxes(title_text="selected", row=2, col=1)
    fig.show()
```
