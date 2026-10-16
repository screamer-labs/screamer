---
name: TripleBarrierOHLC
title: Triple-barrier label on OHLC bars
implementation_family: fin
topics:
- supervised
tags:
- triple barrier
- label
- labeling
- target
- classification
- meta-labeling
- take profit
- stop loss
- vertical barrier
- machine learning
- ohlc
- bars
short: 'A three-state label over a window of OHLC bars: +1 when the upper barrier is reached
  first, -1 when the lower barrier is reached first, 0 when neither is reached within `horizon`
  bars; barriers are tested on each bar''s open, high, and low.'
inputs: 4
outputs: 3
parameters:
- name: horizon
  type: int
  default: 20
  min: 1
  description: Bars after each entry in which a barrier can be reached. It is also the vertical
    barrier, and the delay of the output.
- name: upper
  type: float
  default: 0.0
  min: 0.0
  description: Distance of the upper barrier above the entry close, in `barrier_unit`. 0 turns
    it off.
- name: lower
  type: float
  default: 0.0
  min: 0.0
  description: Distance of the lower barrier below the entry close, in `barrier_unit`. 0 turns
    it off. Below 1 with barrier_unit="fraction".
- name: barrier_unit
  type: str
  default: fraction
  description: 'Unit of `upper` and `lower`: "fraction" of the entry close (0.02 = 2%), "price"
    (an absolute amount), or "vol" (multiples of the volatility expected over the horizon).'
- name: vol_span
  type: float
  default: 100.0
  description: Span of the exponentially weighted standard deviation of one-step log returns
    of the close that sets the "vol" barriers. Must be greater than 1. Used only with barrier_unit="vol".
nan_policy: nan-aware
see_also:
- TripleBarrier
- forecast_pairs
- BacktestOHLCTarget
- LogReturn
output_names:
- label
- return
- holding
references:
- 'Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, chapter 3. Wiley.'
---

# `TripleBarrierOHLC`

## Description

`TripleBarrierOHLC` is a three-state label over a window of OHLC bars. Every
bar `e` is an entry at its close `p_e`. Over the next `horizon` bars the
label is `+1` when the upper barrier is reached first, `-1` when the lower
barrier is reached first, and `0` when neither is reached (Lopez de Prado
2018). The label describes the price path only; no position or side is
involved. Inputs are `(open, high, low, close)`.

It returns three columns:

0. `label`: `+1`, `-1`, or `0`.
1. `return`: `exit / p_e - 1`, the simple return from the entry close to the
   exit price.
2. `holding`: the bars from entry to exit, 1 to `horizon`.

**Stamping.** The output at step `t` is the entry at `t - horizon`, the first
bar at which its outcome is known. The first `horizon` outputs are `NaN`.
Pair it with the features of its entry bar with
[`forecast_pairs(X, y, count=horizon)`](../functions_supervised/forecast_pairs.md).

**Barriers.** `upper` and `lower` are distances above and below the entry
close, in `barrier_unit`: `"fraction"` (`p_e * (1 + upper)`), `"price"`
(`p_e + upper`), or `"vol"` (`p_e * (1 + upper * s)`, with `s` the
`EwStd(span=vol_span)` of the one-step log returns of the close up to and
including `e`, times `sqrt(horizon)`); the lower barrier mirrors each. A
barrier set to `0` is off. See [`TripleBarrier`](TripleBarrier.md) for the
units in full.

**Bars.** Each bar after the entry is tested in this order, with the rules
of the backtest engines' risk exits:

1. An open at or beyond a barrier exits at the open (a gap through it).
2. When the high reaches the upper barrier and the low reaches the lower one,
   the order of the two within the bar is unknown, and the entry's three
   outputs are `NaN`.
3. A high at or above the upper barrier gives `+1`, a low at or below the
   lower barrier gives `-1`. The exit price is the barrier level, kept inside
   the bar's range.
4. When no barrier is reached by bar `e + horizon`, the label is `0` and the
   exit is that bar's close.

On every bar that is not ambiguous, the exit bar and price agree with
[`BacktestOHLCTarget`](BacktestOHLCTarget.md) holding a position from the
entry close with `stop_loss` and `take_profit` at the same levels.

**Missing prices.** A `NaN` or non-positive close at entry, or a `NaN` or
non-positive open, high, low, or close inside an entry's window, makes that
entry's outputs `NaN`; other entries are unaffected.

**Meta-labeling.** With a rule that gives a `side` (`+1` long, `-1` short) at
each bar, and symmetric barriers, the trade worked when `label == side`, and
its return is `side * return`.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `nan-aware`.** This function is built to process `NaN` inputs, as the Description explains. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import TripleBarrierOHLC

    rng = np.random.default_rng(5)
    n, h = 120, 8
    close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.standard_normal(n) * 0.002)
    high = np.maximum(open_, close) * np.exp(np.abs(rng.standard_normal(n)) * 0.004)
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.standard_normal(n)) * 0.004)

    out = TripleBarrierOHLC(horizon=h, upper=1.5, lower=1.5, barrier_unit='vol',
                            vol_span=20)(open_, high, low, close)
    label = out[h:, 0]                       # row e: the label of the entry at bar e
    bars = np.arange(n - h)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(open=open_, high=high, low=low, close=close,
                                 name='bars', showlegend=False))
    for value, color, name in [(1, 'seagreen', 'upper first (+1)'),
                               (-1, 'crimson', 'lower first (-1)'),
                               (0, 'steelblue', 'neither (0)')]:
        at = bars[label == value]
        fig.add_trace(go.Scatter(x=at, y=close[at], mode='markers', name=name,
                                 marker=dict(color=color, size=6)))
    fig.update_layout(title='TripleBarrierOHLC: barriers at 1.5 expected deviations over 8 bars',
                      yaxis=dict(title='price'), xaxis_rangeslider_visible=False,
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

<!-- HELP_END -->
