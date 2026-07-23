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
short: Running mean over the whole history since the last reset.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `ExpandingMean`

## Description

The `ExpandingMean` function returns the arithmetic mean of every sample seen since the start of the stream (or since the last `reset`). It is the whole-history analogue of `RollingMean` and matches `pandas.Series.expanding().mean()`. Memory is `O(1)`.

*Equation*:

$$
y[t] = \frac{1}{t+1} \sum_{i=0}^{t} x[i]
$$

*Parameters*: none.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in the input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
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
