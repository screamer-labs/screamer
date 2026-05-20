---
name: KAMA
title: Kaufman Adaptive MA (KAMA)
implementation_family: rolling
topics:
- trend
- smoothing
tags:
- kama
- kaufman
- adaptive
- moving-average
short: Adaptive MA whose smoothing constant responds to the efficiency ratio.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 10
  min: 2
  description: Efficiency-ratio lookback.
- name: fast
  type: int
  default: 2
  min: 2
  description: Fast EMA period.
- name: slow
  type: int
  default: 30
  min: 2
  description: Slow EMA period.
---

# `KAMA`

## Description

`KAMA` (Kaufman's Adaptive Moving Average, Perry Kaufman 1998) smooths the input with a per-step smoothing constant that adapts to the **efficiency** of the recent price action. When the input moves monotonically (most of its travel is net displacement), the smoothing constant approaches the fast EMA's α. When the input is noisy and round-trips a lot (much travel, little net displacement), it approaches the slow EMA's α. The result tracks trending markets responsively while ignoring chop.

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

By the triangle inequality `direction ≤ volatility`, so `ER ∈ [0, 1]` always; no clamping is needed.

## Parameters

- `window_size` (int, ≥ 2): the efficiency-ratio lookback `n`.
- `fast` (int, optional): "fast" smoothing period. Default `2` (TA-Lib's hard-coded value); `α_fast = 2/3`.
- `slow` (int, optional): "slow" smoothing period. Default `30` (TA-Lib's hard-coded value); `α_slow ≈ 0.0645`.

*Warmup*: first valid output at sample index `window_size`. The recurrence is seeded with `KAMA[n-1] = x[n-1]`, matching TA-Lib exactly. Earlier samples return `NaN`.

*NaN handling*: NaN inputs poison subsequent outputs (the rolling sum of absolute deltas absorbs the NaN). Preprocess if necessary.

## Implementation Details

### Composition

Built from existing screamer primitives:

- `detail::DelayBuffer(n)` for `x[t-n]`.
- `detail::RollingSum(n)` for `Σ |Δx|` over the period (the volatility denominator).
- One `prev_x_` scalar for the one-step difference.
- One running KAMA scalar.

### Complexity

* Time complexity: `O(1)` per step.
* Space complexity: `O(window_size)`.

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

## Reference

Equivalent to TA-Lib's `KAMA` and pandas-ta-classic's `kama` to ~1e-15 (post-warmup). Cross-validated in `tests/test_third_party_alignment.py`.
