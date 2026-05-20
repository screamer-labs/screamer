---
name: LogReturn
title: Log return
implementation_family: fin
topics:
- transforms
tags:
- return
- log-return
- returns
short: log(x[t] / x[t-k]) — log return at lag k.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 1
  min: 1
  description: Lag for the log return.
---

# `LogReturn`

## Description

The `LogReturn` function computes the natural logarithm of the ratio between an element and the element `delay` positions before it. This function is a key tool in financial analysis for modeling continuously compounded returns, as it provides a symmetric measure for gains and losses.

*Equation*:

$$
y[i] = \ln \left(\frac{x[i]}{x[i - \text{delay}]}\right)
$$

*Parameters*:

- `delay` (int): The number of steps backward to use for calculating the log return. Must be non-negative.

*NaN handling*: When `delay` exceeds the available data points at the start of the sequence or if `x[i - \text{delay}]` is zero or negative (to avoid invalid operations in the logarithm), the output is set to `NaN`.

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import LogReturn

    data = np.exp(0.1*np.cumsum(np.random.normal(size=50)))
    log_return_data = LogReturn(3)(data)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[1/2, 1/2], vertical_spacing=0.1)

    fig.add_trace(go.Scatter(y=data, mode='lines+markers', name='Original Data'), row=1, col=1)
    fig.add_trace(go.Scatter(y=log_return_data, mode='lines+markers', name='Log Return (3-step)', line=dict(color='purple')), row=2, col=1)

    fig.update_layout(
        title="Log Return Computation (3-step Delay)",
        xaxis_title="Index",
        yaxis_title="Original Data",
        yaxis2_title="Log Return",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.show()
```

<!-- HELP_END -->

## Implementation Details

`LogReturn` performs element-wise computation of the natural logarithm of the return ratio using array operations. To ensure the validity of results, it checks for non-positive denominators and sets the output to `NaN` where such conditions are detected. The function is optimized for speed and handles large data sets efficiently.