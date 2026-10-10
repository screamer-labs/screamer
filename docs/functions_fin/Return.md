---
name: Return
title: Simple return
implementation_family: fin
topics:
- returns
tags:
- return
- returns
- simple return
- pct_change
- pct-change
- percent change
- percentage change
short: Simple return (x[t] - x[t-k]) / x[t-k] at lag k.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 1
  min: 1
  description: 'Lag k: the output compares each value with the value k steps earlier (1 compares
    consecutive values).'
nan_policy: propagate
see_also:
- LogReturn
- ROCP
- ROC
- ROCR
- Diff
---

# `Return`

## Description

The `Return` function computes the simple return between an element and the element `window_size` positions before it.

*Equation*:

$$
y[i] = \frac{x[i] - x[i - k]}{x[i - k]}
$$

where `k` is `window_size`.

The first `k` outputs are `NaN` because `x[i - k]` does not exist yet. The division follows IEEE arithmetic, so `x[i - k] == 0` gives `inf` (or `NaN` when `x[i]` is also zero). The result matches `pandas.Series.pct_change(k)`.


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
    from screamer import Return

    np.random.seed(0)  # reproducible example data

    data = np.exp(0.1*np.cumsum(np.random.normal(size=50)))
    return_data = Return(5)(data)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[1/2, 1/2], vertical_spacing=0.1)

    fig.add_trace(go.Scatter(y=data, mode='lines+markers', name='Original Data'), row=1, col=1)
    fig.add_trace(go.Scatter(y=return_data, mode='lines+markers', name='Return (5-step)', line=dict(color='blue')), row=2, col=1)

    fig.update_layout(
        title="Simple Return Computation (5-step Delay)",
        xaxis_title="Index",
        yaxis_title="Original Data",
        yaxis2_title="Simple Return",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.show()
```

<!-- HELP_END -->
