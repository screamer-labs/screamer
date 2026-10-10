---
name: RollingYangZhangVol
title: Rolling Yang-Zhang volatility
implementation_family: rolling
topics:
- volatility
tags:
- yang-zhang
- range-based
- ohlc
- drift-robust
- gap-aware
- vol
- rolling
- realized volatility
- volatility
- overnight gap
short: Vol form of the Yang-Zhang estimator (drift + gap robust).
inputs: 4
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of bars n in each variance component; the first estimate needs n + 1
    bars.
nan_policy: ignore
see_also:
- RollingYangZhangVar
- EwYangZhangVol
- RollingRogersSatchellVol
- RollingGarmanKlassVol
- RollingParkinsonVol
references:
- Yang, D. and Zhang, Q. (2000). Drift-independent volatility estimation based on high, low,
  open, and close prices. *Journal of Business*, 73(3), 477-491. doi:10.1086/209650
---

# `RollingYangZhangVol`

## Description

The Yang-Zhang (2000) estimator combines three variance components:

$$
\begin{aligned}
\sigma^2_o    &= \text{sample variance of overnight log returns } \ln(O_t / C_{t-1}) \\
\sigma^2_c    &= \text{sample variance of open-to-close log returns } \ln(C_t / O_t) \\
\sigma^2_{RS} &= \text{mean of per-bar Rogers-Satchell estimates} \\
k             &= \dfrac{0.34}{1.34 + (n+1)/(n-1)} \\
\sigma^2_{YZ} &= \sigma^2_o + k\ \cdot\ \sigma^2_c + (1-k)\ \cdot\ \sigma^2_{RS}
\end{aligned}
$$

where `n = window_size`. The estimator is unbiased under both a non-zero drift and overnight gaps, and Yang and Zhang report an efficiency of up to about 14 times close-to-close.

The inputs are `(open, high, low, close)`. The first valid output is at sample index `window_size`, because `n` overnight returns need `n + 1` bars. `RollingYangZhangVol` returns the square root of `RollingYangZhangVar`. The exponentially weighted form, `EwYangZhangVol`, replaces the window size in `k` with an effective sample count derived from the decay.


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
    from screamer import RollingYangZhangVol

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(np.random.normal(0, 0.4, size=300))
    high = np.maximum(open_, close) + wick
    low  = np.minimum(open_, close) - wick
    out = RollingYangZhangVol(window_size=20)(open_, high, low, close)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=close, name="close"), row=1, col=1)
    fig.add_trace(go.Scatter(y=out, name="RollingYangZhangVol", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Rolling Yang-Zhang volatility (RollingYangZhangVol)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="volatility", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
