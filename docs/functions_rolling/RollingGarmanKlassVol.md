---
name: RollingGarmanKlassVol
title: Rolling Garman-Klass volatility
implementation_family: rolling
topics:
- volatility
tags:
- garman-klass
- range-based
- ohlc
- vol
- rolling
- garman klass volatility
- realized volatility
- ohlc volatility
short: Vol form of the Garman-Klass range-based volatility estimator (OHLC).
inputs: 4
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 1
  description: Number of most recent bars whose per-bar Garman-Klass variances are averaged.
see_also:
- RollingGarmanKlassVar
- RollingParkinsonVar
- RollingRogersSatchellVar
- RollingYangZhangVar
- EwGarmanKlassVar
references:
- Garman, M. B. and Klass, M. J. (1980). On the estimation of security price volatilities
  from historical data. *Journal of Business*, 53(1), 67-78. doi:10.1086/296072
nan_policy: ignore
---

# `RollingGarmanKlassVol`

## Description

`RollingGarmanKlassVol` is the square root of [`RollingGarmanKlassVar`](RollingGarmanKlassVar.md): the Garman-Klass (1980) range-based volatility estimator over a trailing window.
Each bar contributes

$$
\sigma^2_\text{GK}[t] = \tfrac{1}{2}\big(\ln H/L\big)^2 - (2\ln 2 - 1)\big(\ln C/O\big)^2
$$

and the estimator is the rolling mean of this quantity over `window_size` bars.

This is the paper's simplified estimator, written $\sigma_5$ in Garman and Klass's
sequence and the form usually meant by "Garman-Klass". The paper also gives a
higher-order approximation of the same quantity ($\sigma_4$), and variants with an
overnight term driven by the fraction of the day the market is closed ($\sigma_1$,
$\sigma_3$, $\sigma_6$). screamer ships only $\sigma_5$: $\sigma_4$ differs in the
fourth significant figure, and for overnight gaps
[`RollingYangZhangVar`](RollingYangZhangVar.md) handles both the gaps and drift.

Inputs are `(open, high, low, close)`. Under the model's assumptions (zero drift, no
overnight gaps) the estimator is about 7.4 times as efficient as the close-to-close
variance from [`RollingStd`](RollingStd.md).

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
    from screamer import RollingGarmanKlassVol

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(np.random.normal(0, 0.4, size=300))
    high = np.maximum(open_, close) + wick
    low  = np.minimum(open_, close) - wick
    out = RollingGarmanKlassVol(window_size=20)(open_, high, low, close)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=close, name="close"), row=1, col=1)
    fig.add_trace(go.Scatter(y=out, name="RollingGarmanKlassVol", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Rolling Garman-Klass volatility (RollingGarmanKlassVol)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="volatility", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
