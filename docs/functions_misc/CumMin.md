---
name: CumMin
title: Cumulative minimum
implementation_family: misc
topics:
- transforms
- statistics
tags:
- cumulative
- min
short: Running minimum from t=0.
inputs: 1
outputs: 1
parameters: []
---

# `CumMin`

## Description

The `CumMin` function returns the running minimum of all samples seen since the start of the stream (or since the last `reset`). The output is monotonically non-increasing while inputs are finite. It is the streaming equivalent of `numpy.minimum.accumulate`. Memory is `O(1)` regardless of how many samples have been processed.

This is an *expanding* (cumulative-from-zero) reduction, not a sliding window. For a fixed-window trough see `RollingMin`.

*Equation*:

$$
y[t] = \min_{0 \le i \le t} x[i]
$$

*Parameters*: none.

*NaN handling*: Once an input is NaN, every subsequent output is NaN. This matches `numpy.minimum.accumulate`.

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import CumMin

    rng = np.random.default_rng(3)
    n = 300
    returns = rng.normal(0.0005, 0.012, size=n)
    price = 100.0 * np.cumprod(1.0 + returns)
    worst_return = CumMin()(returns)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[1/3, 1/3, 1/3], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=price, mode='lines', name='Price',
                             line=dict(color='steelblue')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=returns, mode='lines', name='Period return',
                             line=dict(color='gray')),
                  row=2, col=1)
    fig.add_trace(go.Scatter(y=worst_return, mode='lines',
                             name='CumMin(return) = worst so far',
                             line=dict(color='red', dash='dash')),
                  row=3, col=1)
    fig.update_layout(
        title="CumMin: Worst Single-Period Return Seen So Far",
        xaxis_title="Index",
        yaxis_title="Price",
        yaxis2_title="Return",
        yaxis3_title="Worst return",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

`CumMin` keeps a single `double` initialised to `+infinity`. Each input is compared and the smaller value retained. There is no warmup. The numpy reference is `numpy.minimum.accumulate`.
