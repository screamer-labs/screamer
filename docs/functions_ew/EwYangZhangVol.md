---
name: EwYangZhangVol
title: EW Yang-Zhang volatility
implementation_family: ew
topics:
- volatility
tags:
- yang-zhang
- range-based
- ohlc
- drift-robust
- gap-robust
- overnight gap
- vol
- volatility
- realized volatility
- ew
short: Exponentially weighted Yang-Zhang range-based volatility estimator (OHLC).
inputs: 4
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
- EwYangZhangVar
- RollingYangZhangVol
- EwRogersSatchellVol
- EwGarmanKlassVol
references:
- Yang, D. and Zhang, Q. (2000). Drift-independent volatility estimation based on high, low,
  open, and close prices. *Journal of Business*, 73(3), 477-491. doi:10.1086/209650
nan_policy: ignore
---

# `EwYangZhangVol`

## Description

`EwYangZhangVol` is the square root of [`EwYangZhangVar`](EwYangZhangVar.md), the exponentially weighted Yang-Zhang (2000) range-based variance estimate. It takes four inputs, `(open, high, low, close)`, and returns a volatility in units of log price per bar:

$$
\sigma_{YZ} = \sqrt{\sigma^2_{YZ}}
$$

The estimator accounts for both drift and overnight gaps. [`EwYangZhangVar`](EwYangZhangVar.md) gives the full definition and its components. The smoothing factor `alpha` is set by exactly one of `com`, `span`, `halflife` or `alpha`.

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
    from screamer import EwYangZhangVol

    rng = np.random.default_rng(1)
    N = 500
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, N)))
    o = c * np.exp(rng.normal(0, 0.004, N))
    hi = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.006, N)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.006, N)))
    vol = EwYangZhangVol(span=60)(o, hi, lo, c)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=vol, name='EwYangZhangVol(span=60)',
                             line=dict(color='crimson')))
    fig.update_layout(
        title='EW Yang-Zhang volatility from OHLC bars',
        yaxis=dict(title='per-bar volatility'),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.show()
```

<!-- HELP_END -->
