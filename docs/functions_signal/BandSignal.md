---
name: BandSignal
title: Entry and exit bands on a z-score
implementation_family: signal
topics:
- signals
- pairs
tags:
- pairs
- mean reversion
- hysteresis
- entry
- exit
- stop
- z-score
- signal
- stop loss
- threshold
- z-score bands
short: 'Three-state mean-reversion position from a z-score: short above +entry, long below
  -entry, flat again inside the exit level, with an optional stop and lockout.'
inputs: 1
outputs: 1
parameters:
- name: entry
  type: float
  default: 2.0
  description: Entry level. A flat position goes short when the z-score reaches +entry and
    long when it reaches -entry.
- name: exit
  type: float
  default: 0.0
  description: Exit level, in (-entry, entry). A short exits when the z-score falls to exit,
    a long when it rises to -exit.
- name: stop
  type: float
  default: .inf
  description: Stop level, above entry (inf = no stop). A short exits when the z-score reaches
    +stop, a long at -stop, and that side cannot re-enter until the z-score returns inside
    the exit level.
nan_policy: ignore
see_also:
- SchmittTrigger
- KalmanRegression
- RollingZscore
- Hold
---

# `BandSignal`

## Description

`BandSignal(entry, exit, stop)(z)` turns a z-score into a position of `-1`, `0`,
or `+1` in a mean-reverting spread. The position is short (`-1`) when the
spread is high, because it is expected to fall, and long (`+1`) when it is low:

- Flat: go short when `z >= entry`, long when `z <= -entry`.
- Short: exit when `z <= exit`. Long: exit when `z >= -exit`.
- Stop: a short exits when `z >= stop` and a long when `z <= -stop`. That side
  is then locked until the z-score returns inside the exit level, so a spread
  that keeps diverging is not entered again at once. No entry is made beyond
  the stop.

The gap between the entry and exit levels is hysteresis: a z-score that hovers
near one level does not switch the position on and off. A negative `exit`
holds a position until the spread overshoots its mean. If the z-score jumps
from beyond one entry level to beyond the other in one step, the position
exits and enters the other side on that step.

Multiply the position by the leg sizes to get the targets of a pairs trade; see
the [portfolio backtest](../functions_fin/BacktestPortfolioTarget.md).

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
    from screamer import BandSignal

    rng = np.random.default_rng(2)
    z = np.zeros(600)
    for t in range(1, 600):
        z[t] = 0.95 * z[t - 1] + rng.standard_normal() * 0.45
    z[400:440] += np.linspace(0, 5, 40)                   # a divergence that hits the stop

    pos = BandSignal(entry=2.0, exit=0.0, stop=3.5)(z)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=z, name='z-score', line=dict(color='gray')))
    fig.add_trace(go.Scatter(y=pos, name='position', line=dict(shape='hv', color='crimson')))
    for level, dash in [(2, 'dash'), (-2, 'dash'), (3.5, 'dot'), (-3.5, 'dot')]:
        fig.add_hline(y=level, line_dash=dash, line_color='steelblue')
    fig.update_layout(title='BandSignal(entry=2, exit=0, stop=3.5)', xaxis_title='Step',
                      margin=dict(l=20, r=20, t=80, b=20))
    fig.show()
```

<!-- HELP_END -->
