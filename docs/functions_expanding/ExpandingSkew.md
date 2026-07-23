---
name: ExpandingSkew
title: Expanding skewness
implementation_family: expanding
topics:
- statistics
tags:
- skew
- expanding
short: Running bias-corrected sample skewness (G1) over the whole history.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `ExpandingSkew`

## Description

The `ExpandingSkew` function returns the adjusted Fisher-Pearson standardized moment coefficient (G1) of every sample seen since the last `reset`. It uses the same bias-corrected estimator as `RollingSkew` and `pandas.Series.expanding().skew()`. Undefined (`NaN`) until at least three samples have been seen. Memory is `O(1)`.

*Equation*:

$$
y[t] = \frac{n}{(n-1)(n-2)}\sum_{i=0}^{t}\left(\frac{x[i]-\bar{x}}{s}\right)^3, \quad n=t+1
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
    from screamer import ExpandingSkew

    rng = np.random.default_rng(0)
    N = 300
    data = np.cumsum(rng.standard_normal(N))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input', line=dict(color='steelblue', width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=ExpandingSkew()(data), mode='lines', name='ExpandingSkew',
                             line=dict(color='purple', width=2)), row=2, col=1)
    fig.update_layout(
        title="Expanding skewness over a random walk",
        yaxis=dict(title='Input'), yaxis2=dict(title='Skewness'),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.show()
```

<!-- HELP_END -->
