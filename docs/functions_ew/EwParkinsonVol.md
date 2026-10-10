---
name: EwParkinsonVol
title: Exponentially-weighted Parkinson volatility
implementation_family: ew
topics:
- volatility
tags:
- parkinson
- range-based
- hl
- high low
- vol
- volatility
- realized volatility
- ew
short: Exponentially weighted Parkinson range-based volatility estimator (high, low).
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
- EwParkinsonVar
- RollingParkinsonVol
- EwGarmanKlassVol
- EwYangZhangVol
references:
- Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of
  return. *Journal of Business*, 53(1), 61-65. doi:10.1086/296071
nan_policy: ignore
---

# `EwParkinsonVol`

## Description

`EwParkinsonVol` is the square root of [`EwParkinsonVar`](EwParkinsonVar.md), the exponentially weighted Parkinson (1980) range-based variance estimate. It takes two inputs, `(high, low)`. The per-bar variance contribution is

$$
\sigma^2_\text{Parkinson}[t] = \frac{1}{4 \ln 2}\ \big(\ln H/L\big)^2
$$

and the output is the square root of its exponentially weighted mean, a volatility in units of log price per bar. The smoothing factor `alpha` is set by exactly one of `com`, `span`, `halflife` or `alpha`.

Under zero drift and no overnight gaps, the variance estimator is about five times as efficient as the close-to-close estimator (Parkinson 1980). It ignores the open and close, so it is biased when prices trend or gap between bars.

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
    from screamer import EwParkinsonVol

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(np.random.normal(0, 0.4, size=300))
    high = np.maximum(open_, close) + wick
    low  = np.minimum(open_, close) - wick
    vol = EwParkinsonVol(span=20)(high, low)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=high, name="high"), row=1, col=1)
    fig.add_trace(go.Scatter(y=low, name="low"), row=1, col=1)
    fig.add_trace(go.Scatter(y=vol, name="EwParkinsonVol", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="EW Parkinson volatility from high-low range (EwParkinsonVol)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="volatility", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
