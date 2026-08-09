---
name: EwResidualStd
title: Exponentially-weighted residual std
implementation_family: ew
topics:
- regression
tags:
- ew
- spread
- residual
- pair
short: EW std of the spread x - EwBeta(x,y) * y.
inputs: 2
outputs: 1
parameters:
- name: com
  type: float
  default: null
  description: Center of mass (alpha = 1 / (1 + com)). Exclusive with span/halflife/alpha.
- name: span
  type: float
  default: 20.0
  description: Span (alpha = 2 / (span + 1)). Default smoothing parameter. Exclusive
    with com/halflife/alpha.
- name: halflife
  type: float
  default: null
  description: Halflife (alpha = 1 - 0.5^(1/halflife)). Exclusive with com/span/alpha.
- name: alpha
  type: float
  default: null
  description: Smoothing parameter directly. Exclusive with com/span/halflife.
nan_policy: ignore
---

# `EwResidualStd`

## Description

`EwResidualStd` is the exponentially weighted standard deviation of the spread `x - EwBeta(x, y) * y`, the EW analog of [`RollingResidualStd`](../functions_fin/RollingResidualStd.md). It is the natural scale for normalising an [`EwSpread`](EwSpread.md): the mean-reversion z-score of a spread is `(spread - ewmean(spread)) / EwResidualStd`.

$$
\text{EwResidualStd}(x, y) = \operatorname{Std}_{ew}\!\big(x - \beta_{ew}\, y\big)
$$

This is a **2-input, 1-output** function (`FunctorBase<_, 2, 1>`).

## Argument-order convention

The **first argument is the target**, the **second is the regressor**, matching [`EwBeta`](EwBeta.md) and [`EwSpread`](EwSpread.md).

## Parameters

Specify exactly one of the following to set the smoothing factor `alpha`:

- **`com`**: Center of mass. `alpha = 1 / (1 + com)`
- **`span`**: Span. `alpha = 2 / (span + 1)`
- **`halflife`**: Half-life. `alpha = 1 - exp(-log(2) / halflife)`
- **`alpha`**: Directly sets the smoothing factor, `0 < alpha < 1`

The output is `NaN` while the spread is still warming up.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import EwSpread, EwResidualStd

    rng = np.random.default_rng(0)
    N = 400
    common = np.cumsum(rng.standard_normal(N))
    a = common + 0.3 * np.cumsum(rng.standard_normal(N))
    b = 1.2 * common + 0.3 * np.cumsum(rng.standard_normal(N))

    spread = EwSpread(span=60)(a, b)
    resid_std = EwResidualStd(span=60)(a, b)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=spread, name='EwSpread(60)'), row=1, col=1)
    fig.add_trace(go.Scatter(y=resid_std, name='EwResidualStd(60)',
                             line=dict(color='crimson')), row=2, col=1)
    fig.update_layout(
        title='EW spread and its residual scale (span=60)',
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.update_yaxes(title_text='spread', row=1, col=1)
    fig.update_yaxes(title_text='std', row=2, col=1)
    fig.show()
```

<!-- HELP_END -->

## Reference

Composes [`EwSpread`](EwSpread.md) and [`EwStd`](EwStd.md); `O(1)` per step. The EW
counterpart of `RollingResidualStd`.
