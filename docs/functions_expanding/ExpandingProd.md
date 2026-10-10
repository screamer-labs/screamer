---
name: ExpandingProd
title: Expanding product
implementation_family: expanding
topics:
- cumulative
tags:
- product
- expanding
- cumulative
- running product
- cumprod
- compounding
short: Running product since the start of the stream.
inputs: 1
outputs: 1
parameters: []
see_also:
- CumProd
- ExpandingSum
- CumSum
nan_policy: ignore
---

# `ExpandingProd`

## Description

`ExpandingProd` returns the running product of every sample seen since the start of the stream, or since the last `reset`. It computes the same result as [`CumProd`](../functions_misc/CumProd.md), under the name of the expanding family, and matches `numpy.cumprod` on input without `NaN`.

$$
y[t] = \prod_{i=0}^{t} x[i]
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
    from screamer import ExpandingProd

    rng = np.random.default_rng(0)
    N = 300
    data = 1.0 + 0.01 * rng.standard_normal(N)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Gross returns', line=dict(color='steelblue', width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=ExpandingProd()(data), mode='lines', name='ExpandingProd',
                             line=dict(color='darkorange', width=2)), row=2, col=1)
    fig.update_layout(
        title="Expanding product of gross returns (cumulative growth curve)",
        yaxis=dict(title='Gross return'), yaxis2=dict(title='Cumulative growth'),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.show()
```

<!-- HELP_END -->
