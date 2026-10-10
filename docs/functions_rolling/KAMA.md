---
name: KAMA
title: Kaufman Adaptive MA (KAMA)
implementation_family: rolling
topics:
- smoothing
- trend
tags:
- kama
- kaufman
- adaptive
- moving-average
- kaufman adaptive moving average
- adaptive moving average
- efficiency ratio
short: Adaptive MA whose smoothing constant responds to the efficiency ratio.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 10
  min: 2
  description: 'Efficiency-ratio lookback n in samples: the net move and the path length are
    measured over the last n steps.'
- name: fast
  type: int
  default: 2
  min: 2
  description: Period of the fastest smoothing constant, alpha_fast = 2 / (fast + 1), used
    when the efficiency ratio is 1. TA-Lib fixes it at 2.
- name: slow
  type: int
  default: 30
  min: 2
  description: Period of the slowest smoothing constant, alpha_slow = 2 / (slow + 1), used
    when the efficiency ratio is 0. TA-Lib fixes it at 30.
see_also:
- EwMean
- DEMA
- HullMA
- MAMA
references:
- 'Kaufman, P. J. (1995). *Smarter Trading: Improving Performance in Changing Markets*. McGraw-Hill.'
nan_policy: ignore
---

# `KAMA`

## Description

`KAMA` (Kaufman's Adaptive Moving Average) smooths the input with a smoothing constant
that changes every step with the efficiency of the recent price path. When the input
moves in one direction (most of its travel is net displacement), the constant
approaches the fast EMA's alpha. When the input is noisy and reverses often (much
travel, little net displacement), it approaches the slow EMA's alpha. The average
therefore follows trends quickly and moves little in sideways, noisy stretches.

$$
\begin{aligned}
\text{direction}[t]  &= \big| x[t] - x[t-n] \big| \\
\text{volatility}[t] &= \sum_{i=1}^{n} \big| x[t-i+1] - x[t-i] \big| \\
\text{ER}[t]         &= \text{direction} / \text{volatility} \quad \in [0, 1] \\
\alpha_\text{fast}   &= 2 / (\text{fast} + 1) \\
\alpha_\text{slow}   &= 2 / (\text{slow} + 1) \\
\text{SC}[t]         &= \big( \text{ER} \cdot (\alpha_\text{fast} - \alpha_\text{slow}) + \alpha_\text{slow} \big)^{2} \\
\text{KAMA}[t]       &= \text{KAMA}[t-1] + \text{SC}[t] \cdot (x[t] - \text{KAMA}[t-1])
\end{aligned}
$$

By the triangle inequality `direction ≤ volatility`, so `ER ∈ [0, 1]` without
clamping.

The first valid output is at sample index `window_size`. The recurrence is seeded with
`KAMA[n-1] = x[n-1]`, as in TA-Lib, and earlier samples return NaN. Each step costs
`O(1)` time and `O(window_size)` memory.

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
    from screamer import KAMA, EwMean

    rng = np.random.default_rng(0)
    n = 300
    # Quiet drift followed by a noisy regime then a clean run-up.
    quiet = np.cumsum(rng.normal(0.02, 0.3, n // 3))
    noisy = quiet[-1] + np.cumsum(rng.normal(0.0, 1.5, n // 3))
    runup = noisy[-1] + np.cumsum(rng.normal(0.10, 0.3, n - 2 * (n // 3)))
    price = np.concatenate([quiet, noisy, runup])

    kama = KAMA(window_size=10)(price)
    ema = EwMean(span=10)(price)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=price, mode='lines', name='Price',
                             line=dict(color='steelblue')))
    fig.add_trace(go.Scatter(y=ema, mode='lines', name='EMA(span=10)',
                             line=dict(color='gray', dash='dot')))
    fig.add_trace(go.Scatter(y=kama, mode='lines', name='KAMA(n=10)',
                             line=dict(color='red')))
    fig.update_layout(
        title="KAMA tracks trending segments closely and flattens during chop",
        xaxis_title="Index",
        yaxis_title="Value",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Validation

Matches TA-Lib's `KAMA` and pandas-ta-classic's `kama` to about 1e-15 after warmup, in
`tests/test_third_party_alignment.py`.
