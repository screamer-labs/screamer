---
name: MinusDM
title: Minus directional movement (-DM)
implementation_family: rolling
topics:
- trend
tags:
- wilder
- dmi
- directional-movement
- talib
- hl
short: Wilder smoothed negative directional movement.
inputs: 2
outputs: 1
parameters:
- name: window_size
  type: int
  default: 14
  min: 2
  description: Wilder smoothing period.
nan_policy: ignore
---

# `MinusDM`

## Description

`MinusDM` is Wilder's negative directional movement, `-DM` (Wilder, 1978),
Wilder-smoothed over `window_size` bars:

$$
-\text{DM}[t] = \max(L_{t-1} - L,\ 0)\quad\text{if}\quad L_{t-1} - L > H - H_{t-1},\ \text{else}\ 0
$$

$$
-\text{DM}_{\text{smoothed}} = \text{Wilder}(-\text{DM},\ w)
$$

**2-input, 1-output** (`FunctorBase<_, 2, 1>`) on `(high, low)`. The value
depends on high and low only; it does not use close. `MinusDM` shares its
smoothing engine with `ADX` and `MinusDI`, which divide the same quantity by
the smoothed true range to form `-DI`. Use `MinusDM` on its own when the raw
smoothed directional movement is needed rather than the normalized indicator.

### Parameters

**`window_size`** *(int, default 14)*: The Wilder smoothing period. Because
`MinusDM` shares its warmup timeline with the rest of the DMI family (which
also smooths the true range), the first valid output is at sample index
`window_size`, one bar later than a plain Wilder sum of `-DM` alone would
give.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import PlusDM, MinusDM

    rng = np.random.default_rng(0)
    N = 300
    close = 100 + np.cumsum(rng.standard_normal(N))
    high = close + np.abs(rng.standard_normal(N))
    low = close - np.abs(rng.standard_normal(N))

    plus_dm = PlusDM(window_size=14)(high, low)
    minus_dm = MinusDM(window_size=14)(high, low)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=close, name='Close', line=dict(color='steelblue')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=plus_dm, name='+DM', line=dict(color='green', dash='dot')), row=2, col=1)
    fig.add_trace(go.Scatter(y=minus_dm, name='-DM', line=dict(color='orange')), row=2, col=1)
    fig.update_layout(
        title='MinusDM: smoothed negative directional movement',
        yaxis=dict(title='Price'), yaxis2=dict(title='DM'),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Reference

Matches `talib.MINUS_DM` from the first index where both are finite; see
`tests/test_dmi.py`. `talib.MINUS_DM` emits its first value one bar earlier
than `MinusDM`, which follows the shared DMI warmup timeline.
