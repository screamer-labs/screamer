---
name: CumProd
title: Cumulative product
implementation_family: misc
topics:
- transforms
tags:
- cumulative
- product
short: Running product from t=0.
inputs: 1
outputs: 1
parameters: []
---

# `CumProd`

## Description

The `CumProd` function returns the running product of all samples seen since the start of the stream (or since the last `reset`). It is the streaming equivalent of `numpy.cumprod`. Memory is `O(1)` regardless of how many samples have been processed.

*Equation*:

$$
y[t] = \prod_{i=0}^{t} x[i]
$$

*Parameters*: none.

*NaN handling*: NaN propagates by ordinary IEEE-754 multiplication. A single zero input pins the running product to zero from that point onward.

<!-- HELP_END -->

## Usage Example and Plot

A common application is converting a series of period returns `r[t]` into a wealth path by accumulating gross returns `1 + r[t]`.

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import CumProd

    rng = np.random.default_rng(7)
    returns = rng.normal(0.0005, 0.01, size=200)
    wealth = CumProd()(1.0 + returns)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.1)
    fig.add_trace(go.Scatter(y=returns, mode='lines', name='Period return r[t]'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=wealth, mode='lines',
                             name='Wealth = CumProd(1+r)',
                             line=dict(color='green')), row=2, col=1)
    fig.update_layout(
        title="CumProd: Compounded Wealth from Period Returns",
        xaxis_title="Index",
        yaxis_title="r[t]",
        yaxis2_title="Wealth[t]",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.show()
```

## Implementation Details

`CumProd` keeps a single `double` accumulator initialised to `1.0`. Every input multiplies it in-place, and the running product is returned. There is no warmup. The numpy reference is `numpy.cumprod`.
