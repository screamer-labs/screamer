---
name: RollingSeasonalZscore
title: Rolling Seasonal Z-score
implementation_family: rolling
topics:
- standardization
tags:
- seasonal
- seasonality
- intraday
- time of day
- day of week
- profile
- slot
short: (x - mean) / std over the same slot of the last window_size cycles, x included.
inputs: 1
outputs: 1
parameters:
- name: period
  type: int
  default: 24
  min: 1
  description: Steps per cycle, for example 24 for hourly data with a daily cycle.
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of cycles in the window.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before a slot holds window_size values: "strict" gives NaN,
    "expanding" uses the values the slot holds (two are needed for a standard
    deviation), and "zero" starts each slot with window_size zeros.'
nan_policy: nan-aware
see_also:
- RollingSeasonalMean
- RollingSeasonalStd
- RollingZscore
---

# `RollingSeasonalZscore`

## Description

`RollingSeasonalZscore` standardizes each value against the same slot of the last
`window_size` cycles:

$$
z_t = \frac{x_t - \text{mean}_t}{\text{std}_t}
$$

The mean and the sample standard deviation (ddof 1) are taken over the window
that includes `x_t`, as in [`RollingZscore`](RollingZscore.md). With `period=1`
it equals `RollingZscore(window_size)`. On hourly volume it flags an hour that
is unusual for that hour of the day. A slot window whose values are all equal
has standard deviation 0 and gives `NaN` (0 / 0), as in `RollingZscore`.

Step `t` belongs to slot `t mod period`, counted in steps from the first input,
so the input must be on a regular clock: one value per hour for `period=24`,
for example. On hourly volume the current hour of the day is compared with the
same hour on the last `window_size` days.

A `NaN` step advances the slot clock and enters no window, and its output is
`NaN`: a missing hour is treated as missing, and the hours after it stay in
their own slots. With `period=2, window_size=2` and `start_policy="expanding"`,
the input `[1, 10, NaN, 20, 5, 30]` gives `[NaN, NaN, NaN, 0.71, 0.71, 0.71]`, where the `NaN` at index 2 is the missing step.

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
    from screamer import RollingSeasonalZscore

    rng = np.random.default_rng(1)
    hours = np.arange(24 * 30)
    # hourly volume with a daily profile: busy around hour 14
    profile = 100 + 80 * np.exp(-0.5 * ((hours % 24 - 14) / 3.0) ** 2)
    volume = profile * rng.lognormal(0.0, 0.3, len(hours))
    z = RollingSeasonalZscore(period=24, window_size=10)(volume)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=z, name='seasonal z-score', line=dict(color='navy')))
    fig.add_hline(y=2, line=dict(color='crimson', dash='dash'))
    fig.update_layout(title='RollingSeasonalZscore: how unusual a value is for its hour of the day',
                      xaxis_title='hour', yaxis_title='z-score',
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

<!-- HELP_END -->
