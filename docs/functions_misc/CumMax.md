---
name: CumMax
title: Cumulative maximum
implementation_family: misc
topics:
- transforms
- statistics
tags:
- cumulative
- max
short: Running maximum from t=0.
inputs: 1
outputs: 1
parameters: []
---

# `CumMax`

## Description

The `CumMax` function returns the running maximum of all samples seen since the start of the stream (or since the last `reset`). The output is monotonically non-decreasing while inputs are finite. It is the streaming equivalent of `numpy.maximum.accumulate`. Memory is `O(1)` regardless of how many samples have been processed.

This is an *expanding* (cumulative-from-zero) reduction, not a sliding window. For a fixed-window peak see `RollingMax`.

*Equation*:

$$
y[t] = \max_{0 \le i \le t} x[i]
$$

*Parameters*: none.

*NaN handling*: Once an input is NaN, every subsequent output is NaN. This matches `numpy.maximum.accumulate`.

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import CumMax

    rng = np.random.default_rng(2)
    x = np.cumsum(rng.normal(0.0, 1.0, size=300))
    peak = CumMax()(x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=x, mode='lines',
                             name='x[t]', line=dict(color='steelblue')))
    fig.add_trace(go.Scatter(y=peak, mode='lines',
                             name='CumMax(x)[t]',
                             line=dict(color='green', dash='dash')))
    fig.update_layout(
        title="CumMax: High-Water Mark of a Random Walk",
        xaxis_title="Index",
        yaxis_title="Value",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

`CumMax` keeps a single `double` initialised to `-infinity`. Each input is compared and the larger value retained. There is no warmup. The numpy reference is `numpy.maximum.accumulate`.
