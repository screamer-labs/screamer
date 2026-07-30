---
name: MinusDI
title: Minus directional indicator (-DI)
implementation_family: rolling
topics:
- trend
tags:
- wilder
- dmi
- directional-movement
- talib
- hlc
short: Wilder negative directional indicator.
inputs: 3
outputs: 1
parameters:
- name: window_size
  type: int
  default: 14
  min: 2
  description: Wilder smoothing period.
nan_policy: ignore
---

# `MinusDI`

## Description

`MinusDI` is Wilder's negative directional indicator, `-DI` (Wilder, 1978). It
is 100 times the Wilder-smoothed negative directional movement divided by the
smoothed true range, so it measures the share of recent range attributable to
downward movement:

$$
\begin{aligned}
\text{TR}[t]  &= \max(H - L,\ |H - C_{t-1}|,\ |L - C_{t-1}|) \\
-\text{DM}[t] &= \max(L_{t-1} - L,\ 0)\quad\text{if}\quad L_{t-1} - L > H - H_{t-1},\ \text{else}\ 0 \\
-\text{DI}    &= 100 \cdot \text{Wilder}(-\text{DM},\ w) \big/ \text{Wilder}(\text{TR},\ w)
\end{aligned}
$$

**3-input, 1-output** (`FunctorBase<_, 3, 1>`) on `(high, low, close)`.
`MinusDI` shares its smoothing engine with `ADX`, which returns `+DI`, `-DI`,
and `ADX` together; use `MinusDI` on its own when only the negative
directional indicator is needed.

### Parameters

**`window_size`** *(int, default 14)*: The Wilder smoothing period. The first
valid output is at sample index `window_size`.

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
    from screamer import PlusDI, MinusDI

    rng = np.random.default_rng(0)
    N = 300
    close = 100 + np.cumsum(rng.standard_normal(N))
    high = close + np.abs(rng.standard_normal(N))
    low = close - np.abs(rng.standard_normal(N))

    plus_di = PlusDI(window_size=14)(high, low, close)
    minus_di = MinusDI(window_size=14)(high, low, close)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=close, name='Close', line=dict(color='steelblue')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=plus_di, name='+DI', line=dict(color='green', dash='dot')), row=2, col=1)
    fig.add_trace(go.Scatter(y=minus_di, name='-DI', line=dict(color='orange')), row=2, col=1)
    fig.update_layout(
        title='MinusDI: strength of downward movement',
        yaxis=dict(title='Price'), yaxis2=dict(title='DI (0-100)'),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Reference

Matches `talib.MINUS_DI`; verified in `tests/test_dmi.py::TestPlusMinusDI`.
