---
name: ExpandingVar
title: Expanding variance
implementation_family: expanding
topics:
- statistics
- volatility
tags:
- variance
- expanding
- running variance
- cumulative
short: Running sample variance (ddof=1) over the whole history.
inputs: 1
outputs: 1
parameters: []
see_also:
- ExpandingStd
- ExpandingMean
- RollingVar
- EwVar
nan_policy: ignore
---

# `ExpandingVar`

## Description

`ExpandingVar` returns the sample variance (`ddof=1`) of every sample seen since the start of the stream, or since the last `reset`. It uses the same `ddof=1` convention as [`RollingVar`](../functions_rolling/RollingVar.md) and `pandas.Series.expanding().var()`. The output is `NaN` until two samples have been seen. Memory is `O(1)`.

$$
y[t] = \frac{1}{n-1}\sum_{i=0}^{t}(x[i]-\bar{x})^2, \quad n=t+1
$$

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
    from screamer import ExpandingVar

    rng = np.random.default_rng(0)
    N = 300
    data = np.cumsum(rng.standard_normal(N))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input', line=dict(color='steelblue', width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=ExpandingVar()(data), mode='lines', name='ExpandingVar',
                             line=dict(color='darkorange', width=2)), row=2, col=1)
    fig.update_layout(
        title="Expanding variance over a random walk",
        yaxis=dict(title='Input'), yaxis2=dict(title='Variance'),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.show()
```

<!-- HELP_END -->
