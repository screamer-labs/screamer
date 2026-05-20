---
name: CumSum
title: Cumulative sum
implementation_family: misc
topics:
- transforms
tags:
- cumulative
- sum
short: Running sum from t=0.
inputs: 1
outputs: 1
parameters: []
---

# `CumSum`

## Description

The `CumSum` function returns the running sum of all samples seen since the start of the stream (or since the last `reset`). It is the streaming equivalent of `numpy.cumsum`. Memory is `O(1)` regardless of how many samples have been processed.

*Equation*:

$$
y[t] = \sum_{i=0}^{t} x[i]
$$

*Parameters*: none.

*NaN handling*: NaN propagates by ordinary IEEE-754 addition. Once a NaN enters the input, every subsequent output is NaN. This matches `numpy.cumsum`, not `pandas.Series.cumsum(skipna=True)`.

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import CumSum

    data = np.random.default_rng(0).normal(0.05, 1.0, size=100)
    cumulative = CumSum()(data)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.1)
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Daily return'), row=1, col=1)
    fig.add_trace(go.Scatter(y=cumulative, mode='lines',
                             name='Cumulative return',
                             line=dict(color='green')), row=2, col=1)
    fig.update_layout(
        title="CumSum: Running Total of a Random Walk Increment",
        xaxis_title="Index",
        yaxis_title="x[t]",
        yaxis2_title="CumSum(x)[t]",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

`CumSum` keeps a single `double` accumulator. Every input is added in-place, and the running total is returned. There is no warmup. The numpy reference is `numpy.cumsum`.
