---
name: ExpandingMin
title: Expanding minimum
implementation_family: expanding
topics:
- cumulative
tags:
- min
- expanding
- cumulative
short: Running minimum from t=0.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `ExpandingMin`

## Description

The `ExpandingMin` function returns the running minimum of every sample seen since the last `reset`. It is an alias of `CumMin` exposed under the expanding family and matches `pandas.Series.expanding().min()`.

*Equation*:

$$
y[t] = \min_{0 \le i \le t} x[i]
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
    from screamer import ExpandingMin

    rng = np.random.default_rng(0)
    N = 300
    data = np.cumsum(rng.standard_normal(N))

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input', line=dict(color='steelblue', width=1)))
    fig.add_trace(go.Scatter(y=ExpandingMin()(data), mode='lines', name='ExpandingMin',
                             line=dict(color='crimson', width=2)))
    fig.update_layout(
        title="Expanding minimum: monotone lower envelope",
        xaxis_title="Index", yaxis_title="Value",
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->
