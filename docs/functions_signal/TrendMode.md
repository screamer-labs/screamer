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
short: Classifies each sample as trending (1.0) or cycling (0.0) from the dominant-cycle phase
  advance.
inputs: 1
outputs: 1
parameters:
- name: phase_rate_frac
  type: float
  default: 0.5
  min: 0.0
  description: Threshold, as a fraction of the expected per-sample phase advance for the current
    dominant cycle, below which a sample is classified as trending.
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

`TrendMode` classifies each sample of a series as trending or cycling, using
the same instantaneous phase and dominant-cycle period as `CyclePhase` and
`DominantCycle`. While a series oscillates at a stable dominant cycle, the
phase advances by close to `360 / period` degrees per sample. When the phase
advance stalls well below that rate, the dominant cycle has effectively
stopped rotating, which this function reads as a trending market.

The rule: at each sample, compute the phase change from the previous sample
(unwrapped to `[-180, 180]` degrees) and compare its magnitude to
`phase_rate_frac * (360 / period)`, the expected per-sample advance scaled
by `phase_rate_frac`. If the phase change is smaller, the output is `1.0`
(trending); otherwise it is `0.0` (cycling).

This threshold rule is screamer's own way to turn the phase and period
estimates into a binary trend/cycle flag; it does not reproduce a specific
published formula. Lowering `phase_rate_frac` makes the classifier stricter
about calling a trend, and raising it makes trend mode more frequent.

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
    cycle_part = np.sin(2 * np.pi * np.arange(c) / 20)          # 300 samples of a 20-sample cycle
    ramp_part = cycle_part[-1] + 0.1 * np.arange(1, c + 1)        # then 300 samples of a straight ramp
    x = np.concatenate([cycle_part, ramp_part]) + 0.01 * rng.standard_normal(2 * c)

    mode = TrendMode(phase_rate_frac=0.5)(x)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=x, name='cycle, then ramp', line=dict(color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=mode, name='TrendMode(phase_rate_frac=0.5)',
                             line=dict(color='darkorange', shape='hv')), row=2, col=1)
    fig.update_layout(title='TrendMode: no trend flags in the cycle, frequent flags in the ramp',
                      yaxis=dict(title='input'), yaxis2=dict(title='mode', tickvals=[0, 1]),
                      xaxis2=dict(title='sample'),
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
