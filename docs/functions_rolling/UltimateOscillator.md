---
name: UltimateOscillator
title: Ultimate Oscillator
implementation_family: rolling
topics:
- momentum
tags:
- ultimate-oscillator
- williams
- oscillator
- talib
- hlc
- ultimate oscillator
- ULTOSC
short: Three-period weighted oscillator (Williams, 1976).
inputs: 3
outputs: 1
parameters:
- name: period1
  type: int
  default: 7
  min: 2
  description: Shortest of the three averaging periods, in bars; weighted 4.
- name: period2
  type: int
  default: 14
  min: 2
  description: Middle averaging period, in bars; weighted 2.
- name: period3
  type: int
  default: 28
  min: 2
  description: Longest averaging period, in bars; weighted 1. Output starts once this window
    is full.
nan_policy: ignore
see_also:
- Stoch
- WilliamsR
- RollingRSI
- TrueRange
references:
- Williams, L. (1985). The ultimate oscillator. *Technical Analysis of Stocks & Commodities*,
  3(4), 140-141.
---

# `UltimateOscillator`

## Description

`UltimateOscillator` (Larry Williams, 1976) combines the ratio of buying pressure to true range over three periods into one weighted oscillator, so that short, medium and long momentum enter a single value.

$$
\begin{aligned}
\text{BP}[t]   &= \text{close} - \min(\text{low},\ \text{close}_{t-1}) \\
\text{TR}[t]   &= \max(\text{high},\ \text{close}_{t-1}) - \min(\text{low},\ \text{close}_{t-1}) \\
\text{avg}_k   &= \frac{\sum \text{BP} \text{ over } \text{period}_k}{\sum \text{TR} \text{ over } \text{period}_k} \\
\text{UO}[t]   &= 100 \cdot \frac{4 \cdot \text{avg}_1 + 2 \cdot \text{avg}_2 + \text{avg}_3}{7}
\end{aligned}
$$

The 4 / 2 / 1 weighting puts the heaviest emphasis on the shortest period.

The inputs are `(high, low, close)`. The output lies in `[0, 100]` and is NaN until sample index `max(period1, period2, period3)`, following TA-Lib.


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
    from screamer import UltimateOscillator

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(np.random.normal(0, 0.4, size=300))
    high = np.maximum(open_, close) + wick
    low  = np.minimum(open_, close) - wick
    out = UltimateOscillator()(high, low, close)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=high, name="high", line=dict(color="#888")), row=1, col=1)
    fig.add_trace(go.Scatter(y=low, name="low", line=dict(color="#bbb")), row=1, col=1)
    fig.add_trace(go.Scatter(y=close, name="close", line=dict(color="royalblue")), row=1, col=1)
    fig.add_trace(go.Scatter(y=out, name="UO(7,14,28)", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Ultimate oscillator (UltimateOscillator)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="UO (0-100)", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

`UltimateOscillator` keeps the previous close and six rolling sums, one for BP and one for TR at each of the three periods. Each rolling sum is O(1) per step, so the total cost per step is O(1).

## Compatibility

Bit-exact match to `talib.ULTOSC(high, low, close, timeperiod1, timeperiod2, timeperiod3)` post-warmup (verified to ~1e-14 in `tests/test_third_party_alignment.py`).
