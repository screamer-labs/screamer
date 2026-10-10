---
name: CumProd
title: Cumulative product
implementation_family: misc
topics:
- cumulative
tags:
- cumulative
- product
- cumprod
- running product
- compounding
- wealth
short: Running product from t=0.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- CumSum
- ExpandingProd
- Return
- LogReturn
---

# `CumProd`

## Description

The `CumProd` function returns the running product of all samples seen since the start of the stream (or since the last `reset`). Memory is `O(1)` regardless of how many samples have been processed.

*Equation*:

$$
y[t] = \prod_{i=0}^{t} x[i]
$$

A single zero input pins the running product to zero from that point onward. Skipping `NaN` inputs matches `pandas.Series.cumprod(skipna=True)`; `numpy.cumprod` would propagate the `NaN` instead.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

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

<!-- HELP_END -->

## Implementation Details

`CumProd` keeps a single `double` accumulator initialized to `1.0`. Every finite input multiplies it in-place, and the running product is returned. There is no warmup.
