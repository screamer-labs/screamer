---
name: TrendMode
title: Trend-vs-cycle classifier
implementation_family: signal
topics:
- cycles
tags:
- ehlers
- hilbert
- cycle
- analytic-signal
- regime
- trend detection
- trend vs cycle
short: Ehlers' trend mode, 1.0 while the series trends and 0.0 while it cycles.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- DominantCycle
- CyclePhase
- InstantaneousTrendline
references:
- 'Ehlers, J. F. (2001). *Rocket Science for Traders: Digital Signal Processing Applications*.
  Wiley.'
---

# `TrendMode`

## Description

`TrendMode` classifies each sample of a series as trending (`1.0`) or
cycling (`0.0`) with John Ehlers' rule from *Rocket Science for Traders*
("Identifying Market Modes"), as TA-Lib implements it in `HT_TRENDMODE`. It
combines three measurements of the dominant cycle: the sinewave indicator
(`CycleSine`), the rate of its phase, and the instantaneous trendline
(`InstantaneousTrendline`).

Each sample starts as trend, then:

1. A crossing of `sine` and `leadsine` marks a cycle turn. That sample reads
   cycle, and so does every sample until half a dominant period has passed
   since the last crossing.
2. A phase that advances between 0.67 and 1.5 times `360 / period` degrees per
   sample (the dominant cycle rotating at its own rate) reads cycle.
3. A smoothed price at least 1.5% away from the trendline,
   `|smooth - trendline| >= 0.015 |trendline|`, reads trend whatever rules
   1 and 2 say.

Rule 3 compares the price with the trendline in relative terms, so it is
meant for price series, which stay well away from zero. On a series that
oscillates around zero, such as returns or a detrended signal, the trendline
is close to zero and every sample reads trend. For the same reason a cycle
whose swing exceeds about 1.5% of the price reads trend near its peaks and
troughs, where the price runs furthest from the trendline: a cycle of
amplitude 3 on a level of 100 reads cycle on only about half the samples.
There are no settings: the thresholds are Ehlers'.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import TrendMode

    rng = np.random.default_rng(20)
    c = 300
    cycle_part = 100 + np.sin(2 * np.pi * np.arange(c) / 20)    # 300 samples of a 20-sample cycle around 100
    ramp_part = cycle_part[-1] + 0.1 * np.arange(1, c + 1)        # then 300 samples of a straight ramp
    x = np.concatenate([cycle_part, ramp_part]) + 0.01 * rng.standard_normal(2 * c)

    mode = TrendMode()(x)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=x, name='cycle, then ramp', line=dict(color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=mode, name='TrendMode()',
                             line=dict(color='darkorange', shape='hv')), row=2, col=1)
    fig.update_layout(title='TrendMode: cycle (0) in the cycle, trend (1) on the ramp',
                      yaxis=dict(title='input'), yaxis2=dict(title='mode', tickvals=[0, 1]),
                      xaxis2=dict(title='sample'),
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
