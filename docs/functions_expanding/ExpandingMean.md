---
name: ExpandingMean
title: Expanding mean
implementation_family: expanding
topics:
- statistics
tags:
- mean
- expanding
- cumulative
- running mean
- cumulative average
short: Running mean over the whole history since the last reset.
inputs: 1
outputs: 1
parameters: []
see_also:
- RollingMean
- EwMean
- ExpandingSum
- ExpandingStd
nan_policy: ignore
---

# `ExpandingMean`

## Description

`ExpandingMean` returns the arithmetic mean of every sample seen since the start of the stream, or since the last `reset`. It is the whole-history counterpart of [`RollingMean`](../functions_rolling/RollingMean.md) and matches `pandas.Series.expanding().mean()`. Memory is `O(1)`.

$$
y[t] = \frac{1}{n} \sum_{i=0}^{t} x[i], \quad n = t + 1
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
    from screamer import ExpandingMean

    rng = np.random.default_rng(0)
    N = 300
    data = np.cumsum(rng.standard_normal(N))

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input', line=dict(color='steelblue', width=1)))
    fig.add_trace(go.Scatter(y=ExpandingMean()(data), mode='lines', name='ExpandingMean', line=dict(color='crimson', width=2)))
    fig.update_layout(
        title="Expanding mean over a random walk",
        xaxis_title="Index", yaxis_title="Value",
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->
