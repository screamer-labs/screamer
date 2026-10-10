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
- residual volatility
- pair
- pairs trading
- hedge ratio
short: EW std of the spread x - EwBeta(x, y) * y.
inputs: 2
outputs: 1
parameters:
- name: com
  type: float
  default: null
  description: Center of mass c, giving alpha = 1 / (1 + c). Must be positive. Give exactly
    one of com, span, halflife and alpha.
- name: span
  type: float
  default: 20.0
  description: Span s in samples, giving alpha = 2 / (s + 1). Must be greater than 1. Give
    exactly one of com, span, halflife and alpha. The default shown is an example value; the
    constructor has no default.
- name: halflife
  type: float
  default: null
  description: Half-life h in samples, giving alpha = 1 - exp(-ln(2) / h). Must be positive.
    Give exactly one of com, span, halflife and alpha.
- name: alpha
  type: float
  default: null
  description: Smoothing factor, with 0 < alpha < 1; a larger alpha gives recent samples more
    weight. Give exactly one of com, span, halflife and alpha.
see_also:
- EwSpread
- EwBeta
- EwStd
- RollingResidualStd
nan_policy: ignore
---

# `EwResidualStd`

## Description

`EwResidualStd` computes the exponentially weighted standard deviation of the spread `x - EwBeta(x, y) * y`, the output of [`EwSpread`](EwSpread.md). It is the exponentially weighted counterpart of [`RollingResidualStd`](../functions_fin/RollingResidualStd.md), and the scale for normalizing a spread: the mean-reversion z-score of a spread is `(spread - EwMean(spread)) / EwResidualStd`.

$$
\text{EwResidualStd}(x, y) = \operatorname{Std}_{ew}\!\big(x - \beta_{ew}\, y\big)
$$

The first argument is the target and the second is the regressor, as in [`EwBeta`](EwBeta.md) and `EwSpread`. The smoothing factor `alpha` is set by exactly one of `com`, `span`, `halflife` or `alpha`, and the same value is used for the hedge ratio and for the standard deviation.

The output is `NaN` until the spread has two defined values.

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
