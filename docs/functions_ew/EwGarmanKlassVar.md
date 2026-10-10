---
name: EwGarmanKlassVar
title: EW Garman-Klass variance
implementation_family: ew
topics:
- volatility
tags:
- garman-klass
- range-based
- ohlc
- var
- variance
- realized volatility
- ew
short: Exponentially weighted Garman-Klass range-based variance estimator (OHLC).
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
- EwGarmanKlassVol
- RollingGarmanKlassVar
- EwParkinsonVar
- EwRogersSatchellVar
- EwYangZhangVar
references:
- Garman, M. B. and Klass, M. J. (1980). On the estimation of security price volatilities
  from historical data. *Journal of Business*, 53(1), 67-78. doi:10.1086/296072
nan_policy: ignore
---

# `EwGarmanKlassVar`

## Description

`EwGarmanKlassVar` is the exponentially weighted Garman-Klass (1980) range-based variance estimator. It takes four inputs, `(open, high, low, close)`, and averages the per-bar variance contribution

$$
\sigma^2_\text{GK}[t] = \tfrac{1}{2}\big(\ln H/L\big)^2 - (2\ln 2 - 1)\big(\ln C/O\big)^2
$$

with an exponentially weighted mean. The smoothing factor `alpha` is set by exactly one of `com`, `span`, `halflife` or `alpha`. [`EwGarmanKlassVol`](EwGarmanKlassVol.md) returns the square root.

This is the paper's simplified estimator, written $\sigma_5$ in Garman and Klass, and the form usually meant by "Garman-Klass". The paper also gives a higher-order approximation of the same quantity ($\sigma_4$), which differs in the fourth significant figure, and variants with an overnight term ($\sigma_1$, $\sigma_3$, $\sigma_6$). For data with overnight gaps or drift, use [`EwYangZhangVar`](EwYangZhangVar.md) instead.

Under zero drift and no overnight gaps, the estimator is about 7.4 times as efficient as the close-to-close variance estimator (Garman and Klass 1980).

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
    from screamer import EwGarmanKlassVar

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(np.random.normal(0, 0.4, size=300))
    high = np.maximum(open_, close) + wick
    low  = np.minimum(open_, close) - wick
    var = EwGarmanKlassVar(span=20)(open_, high, low, close)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=close, name="close"), row=1, col=1)
    fig.add_trace(go.Scatter(y=var, name="EwGarmanKlassVar", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="EW Garman-Klass variance from OHLC bars (EwGarmanKlassVar)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="variance", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
