---
name: LogReturn
title: Log return
implementation_family: fin
topics:
- returns
tags:
- return
- log return
- log-return
- returns
- continuously compounded return
- log returns
short: Log return log(x[t] / x[t-k]) at lag k.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 1
  min: 1
  description: 'Lag k: the output compares each value with the value k steps earlier.'
nan_policy: propagate
see_also:
- Return
- ROCP
- Diff
- Log
- CumSum
---

# `LogReturn`

## Description

The `LogReturn` function computes the natural logarithm of the ratio between an element and the element `window_size` positions before it.

*Equation*:

$$
y[i] = \ln \left(\frac{x[i]}{x[i - k]}\right)
$$

where `k` is `window_size`.

The first `k` outputs are `NaN` because `x[i - k]` does not exist yet. The logarithm follows IEEE arithmetic: a negative ratio gives `NaN`, and a zero in the numerator or denominator gives `-inf` or `inf`.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `propagate`.** A `NaN` input is stored in the lookback like any other value. The output is `NaN` wherever the formula references it and recovers once it leaves the lookback. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import LogReturn

    np.random.seed(0)  # reproducible example data

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