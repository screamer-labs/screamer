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
- pair
short: x - EwBeta(x,y) * y - EW hedge-adjusted residual.
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

# `EwSpread`

## Description

`EwSpread` is the exponentially weighted hedge-adjusted residual of `x` against `y`, the EW analog of [`RollingSpread`](../functions_fin/RollingSpread.md). At each step it takes the EW regression slope `beta_ew = EwBeta(x, y)` and returns `x[t] - beta_ew[t] * y[t]`.

$$
\beta_{ew}[t] = \frac{\operatorname{Cov}_{ew}(x, y)}{\operatorname{Var}_{ew}(y)},
\qquad
\text{spread}[t] = x[t] - \beta_{ew}[t]\, y[t]
$$

This is a **2-input, 1-output** function (`FunctorBase<_, 2, 1>`).

## Argument-order convention

The **first argument is the target**, the **second is the hedge/regressor**, matching [`EwBeta`](EwBeta.md) and `RollingSpread`. `EwSpread(price_a, price_b)` measures the deviation of `a` from its exponentially weighted hedge against `b`, the building block for pairs-trading mean reversion.

## Parameters

Specify exactly one of the following to set the smoothing factor `alpha`:

- **`com`**: Center of mass. `alpha = 1 / (1 + com)`
- **`span`**: Span. `alpha = 2 / (span + 1)`
- **`halflife`**: Half-life. `alpha = 1 - exp(-log(2) / halflife)`
- **`alpha`**: Directly sets the smoothing factor, `0 < alpha < 1`

The output is `NaN` while `EwBeta` is undefined, that is when the regressor has zero variance over the effective window.

## Identity check

By definition `EwSpread(x, y) == x - EwBeta(x, y) * y`, verified to ~1e-12 in the test suite.


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

## Reference

Composes [`EwBeta`](EwBeta.md); `O(1)` per step. The EW counterpart of the rolling
`x - RollingBeta(x, y) * y`. Apply [`EwZscore`](EwZscore.md) to the spread for a
standard mean-reversion signal, or normalise with [`EwResidualStd`](EwResidualStd.md).
