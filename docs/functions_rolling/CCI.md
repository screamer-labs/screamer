---
name: CCI
title: Commodity Channel Index (CCI)
implementation_family: rolling
topics:
- momentum
tags:
- cci
- lambert
- oscillator
- talib
- hlc
- commodity channel index
short: Commodity Channel Index over typical price.
inputs: 3
outputs: 1
parameters:
- name: window_size
  type: int
  default: 14
  min: 2
  description: Number of most recent typical prices in the mean and mean-absolute-deviation
    window. Lambert used 20; TA-Lib defaults to 14.
see_also:
- RollingMad
- RollingZscore
- WilliamsR
- Stoch
references:
- 'Lambert, D. R. (1980). Commodity Channel Index: tool for trading cyclic trends. *Commodities*,
  October 1980.'
nan_policy: ignore
---

# `CCI`

## Description

`CCI` (Commodity Channel Index, Lambert 1980) measures how far the current typical
price is from its rolling mean, normalized by the mean absolute deviation of the same
window:

$$
\begin{aligned}
\text{TP}[t]      &= (\text{high} + \text{low} + \text{close}) / 3 \\
\overline{\text{TP}} &= \text{SMA}(\text{TP},\ n) \\
\text{MAD}        &= \text{mean}\big(\ |\text{TP} - \overline{\text{TP}}|\ \big) \quad \text{over the same window} \\
\text{CCI}[t]     &= \frac{\text{TP}[t] - \overline{\text{TP}}[t]}{0.015 \cdot \text{MAD}[t]}
\end{aligned}
$$

The constant 0.015 is Lambert's convention: it places roughly 70 to 80 percent of CCI
readings in `[-100, +100]` for normally distributed input.

Inputs are `(high, low, close)`. The output is NaN for the first `window_size - 1`
samples. When the MAD is 0 (a perfectly flat window) the output is `0`.

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
    from screamer import CCI

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(np.random.normal(0, 0.4, size=300))
    high = np.maximum(open_, close) + wick
    low  = np.minimum(open_, close) - wick
    out = CCI(20)(high, low, close)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=high, name="high", line=dict(color="#888")), row=1, col=1)
    fig.add_trace(go.Scatter(y=low, name="low", line=dict(color="#bbb")), row=1, col=1)
    fig.add_trace(go.Scatter(y=close, name="close", line=dict(color="royalblue")), row=1, col=1)
    fig.add_trace(go.Scatter(y=out, name="CCI(20)", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Commodity channel index (CCI)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="CCI", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

One circular buffer of typical prices with a rolling sum gives the mean in `O(1)`; the
MAD is a sweep over the window, `O(window_size)` per step. [`RollingMad`](RollingMad.md)
has the same cost for the same reason: there is no closed-form `O(1)` update for the
MAD when the mean shifts every step.

## Validation

Matches `talib.CCI(high, low, close, timeperiod)` after warmup, to about 1e-11 in
`tests/test_third_party_alignment.py`.
