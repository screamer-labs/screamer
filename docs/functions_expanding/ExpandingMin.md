---
name: ExpandingMin
title: Expanding minimum
implementation_family: expanding
topics:
- cumulative
tags:
- min
- minimum
- expanding
- cumulative
- running min
- cummin
short: Running minimum since the start of the stream.
inputs: 1
outputs: 1
parameters: []
see_also:
- CumMin
- ExpandingMax
- RollingMin
nan_policy: ignore
---

# `ExpandingMin`

## Description

`ExpandingMin` returns the running minimum of every sample seen since the start of the stream, or since the last `reset`. It computes the same result as [`CumMin`](../functions_misc/CumMin.md), under the name of the expanding family, and matches `pandas.Series.expanding().min()`.

$$
y[t] = \min_{0 \le i \le t} x[i]
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
