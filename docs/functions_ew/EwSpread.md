---
name: EwSpread
title: Exponentially-weighted spread
implementation_family: ew
topics:
- regression
tags:
- ew
- spread
- hedge
- hedge ratio
- pair
- pairs trading
- residual
- mean reversion
short: EW hedge-adjusted residual x - EwBeta(x, y) * y.
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
- EwBeta
- EwResidualStd
- EwZscore
- RollingSpread
- KalmanRegression
nan_policy: ignore
---

# `EwSpread`

## Description

`EwSpread` computes the hedge-adjusted residual of the first input `x` against the second input `y`, using the exponentially weighted hedge ratio. It is the exponentially weighted counterpart of [`RollingSpread`](../functions_fin/RollingSpread.md). At each step it takes the slope $\beta_{ew}[t]$ = `EwBeta(x, y)` and returns:

$$
\text{spread}[t] = x[t] - \beta_{ew}[t]\, y[t],
\qquad
\beta_{ew}[t] = \frac{\operatorname{Cov}_{ew}(x, y)}{\operatorname{Var}_{ew}(y)}
$$

The first argument is the target and the second is the hedge instrument, as in [`EwBeta`](EwBeta.md). `EwSpread(price_a, price_b)` measures how far `a` is from its hedge against `b`, the input to a pairs-trading mean-reversion signal. To turn it into such a signal, apply [`EwZscore`](EwZscore.md) to the spread, or divide by [`EwResidualStd`](EwResidualStd.md). The smoothing factor `alpha` is set by exactly one of `com`, `span`, `halflife` or `alpha`.

The output is `NaN` while `EwBeta` is undefined, that is when `y` is constant over the effective window, which includes the first sample.

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
    from screamer import EwSpread

    rng = np.random.default_rng(0)
    N = 400
    common = np.cumsum(rng.standard_normal(N))
    a = common + 0.3 * np.cumsum(rng.standard_normal(N))
    b = 1.2 * common + 0.3 * np.cumsum(rng.standard_normal(N))
    a[200:240] += 5.0   # a transient mispricing in a

    spread = EwSpread(span=60)(a, b)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(y=a, name='a'), row=1, col=1)
    fig.add_trace(go.Scatter(y=b, name='b'), row=1, col=1)
    fig.add_trace(go.Scatter(y=spread, name='EwSpread(span=60)',
                             line=dict(color='crimson')), row=2, col=1)
    fig.update_layout(
        title='EW hedge-adjusted spread of a against b (span=60)',
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.update_yaxes(title_text='price', row=1, col=1)
    fig.update_yaxes(title_text='spread', row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
