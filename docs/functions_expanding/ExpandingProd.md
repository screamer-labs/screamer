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
short: Running product from t=0.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `ExpandingProd`

## Description

The `ExpandingProd` function returns the running product of every sample seen since the last `reset`. It is an alias of `CumProd` exposed under the expanding family and matches `numpy.cumprod` (skipping NaN).

*Equation*:

$$
y[t] = \prod_{i=0}^{t} x[i]
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
