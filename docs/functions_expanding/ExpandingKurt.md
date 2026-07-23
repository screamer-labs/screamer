---
name: ExpandingKurt
title: Expanding kurtosis
implementation_family: expanding
topics:
- statistics
tags:
- kurtosis
- expanding
short: Running bias-corrected excess kurtosis (Fisher) over the whole history.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `ExpandingKurt`

## Description

The `ExpandingKurt` function returns the bias-corrected Fisher excess kurtosis of every sample seen since the last `reset`, using the same estimator as `RollingKurt` and `pandas.Series.expanding().kurt()`. Undefined (`NaN`) until at least four samples have been seen. Memory is `O(1)`.

*Equation*:

$$
y[t] = \frac{n(n+1)}{(n-1)(n-2)(n-3)}\sum_{i=0}^{t}\left(\frac{x[i]-\bar{x}}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}, \quad n=t+1
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
    from screamer import ExpandingKurt

    rng = np.random.default_rng(0)
    N = 300
    data = np.cumsum(rng.standard_normal(N))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input', line=dict(color='steelblue', width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=ExpandingKurt()(data), mode='lines', name='ExpandingKurt',
                             line=dict(color='seagreen', width=2)), row=2, col=1)
    fig.update_layout(
        title="Expanding kurtosis over a random walk",
        yaxis=dict(title='Input'), yaxis2=dict(title='Excess kurtosis'),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.show()
```

<!-- HELP_END -->
