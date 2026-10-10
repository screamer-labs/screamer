---
name: RollingParkinsonVar
title: Rolling Parkinson variance
implementation_family: rolling
topics:
- volatility
tags:
- parkinson
- range-based
- hl
- var
- rolling
- realized volatility
- high-low volatility
- variance
short: Var form of the Parkinson range-based volatility estimator (uses high & low).
inputs: 2
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of most recent bars over which the per-bar estimate is averaged.
nan_policy: ignore
see_also:
- RollingParkinsonVol
- EwParkinsonVar
- RollingGarmanKlassVar
- RollingRogersSatchellVar
- RollingYangZhangVar
references:
- Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of
  return. *Journal of Business*, 53(1), 61-65. doi:10.1086/296071
---

# `RollingParkinsonVar`

## Description

The Parkinson (1980) range-based volatility estimator. Per-bar variance contribution:

$$
\sigma^2_\text{Parkinson}[t] = \frac{1}{4 \ln 2}\ \big(\ln H/L\big)^2
$$

This expression is averaged with a rolling mean over `window_size` bars to form the estimator. `RollingParkinsonVol` returns the square root of `RollingParkinsonVar`.

The inputs are `(high, low)`. Under the model's assumptions (zero drift, no overnight gaps) the estimator is about 5 times as efficient as the close-to-close `RollingStd`.


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
    from screamer import RollingParkinsonVar

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(np.random.normal(0, 0.4, size=300))
    high = np.maximum(open_, close) + wick
    low  = np.minimum(open_, close) - wick
    out = RollingParkinsonVar(window_size=20)(high, low)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=high, name="high"), row=1, col=1)
    fig.add_trace(go.Scatter(y=low, name="low"), row=1, col=1)
    fig.add_trace(go.Scatter(y=out, name="RollingParkinsonVar", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Rolling Parkinson variance (RollingParkinsonVar)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="variance", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
