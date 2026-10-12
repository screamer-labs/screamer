---
name: ExpandingSlope
title: Expanding slope
implementation_family: expanding
topics:
- trend
tags:
- slope
- regression
- expanding
- linear trend
- ols
- trend
short: Running OLS slope of the series against time over the whole history.
inputs: 1
outputs: 1
parameters: []
see_also:
- RollingPoly1
- ExpandingMean
- RollingLinearRegression
- Detrend
nan_policy: ignore
---

# `ExpandingSlope`

## Description

`ExpandingSlope` returns the ordinary-least-squares slope of the samples seen so far against the integer time axis `i = 0, 1, ..., n-1`. It is the whole-history counterpart of [`RollingPoly1`](../functions_rolling/RollingPoly1.md) with `derivative_order=1`. The output is `NaN` until two samples have been seen. Memory is `O(1)`.

$$
y[t] = \frac{n\sum i\,x[i] - (\sum i)(\sum x[i])}{n\sum i^2 - (\sum i)^2}, \quad i=0..t,\; n=t+1
$$

The sums are taken about an origin that follows the mean, so the slope stays accurate when the values are large next to their spread (see [Accuracy at large levels](../conventions.md#accuracy-at-large-levels)).

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
    from screamer import ExpandingSlope

    rng = np.random.default_rng(0)
    N = 300
    data = np.cumsum(rng.standard_normal(N))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input', line=dict(color='steelblue', width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=ExpandingSlope()(data), mode='lines', name='ExpandingSlope',
                             line=dict(color='darkorchid', width=2)), row=2, col=1)
    fig.update_layout(
        title="Expanding OLS slope over a random walk",
        yaxis=dict(title='Input'), yaxis2=dict(title='Slope (units/step)'),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.show()
```

<!-- HELP_END -->
