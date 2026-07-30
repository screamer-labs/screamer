---
name: PlusDI
title: Plus directional indicator (+DI)
implementation_family: rolling
topics:
- trend
tags:
- wilder
- dmi
- directional-movement
- talib
- hlc
short: Wilder positive directional indicator.
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

# `PlusDI`

## Description

`PlusDI` is Wilder's positive directional indicator, `+DI` (Wilder, 1978). It is
100 times the Wilder-smoothed positive directional movement divided by the
smoothed true range, so it measures the share of recent range attributable to
upward movement:

$$
\begin{aligned}
\text{TR}[t]  &= \max(H - L,\ |H - C_{t-1}|,\ |L - C_{t-1}|) \\
+\text{DM}[t] &= \max(H - H_{t-1},\ 0)\quad\text{if}\quad H - H_{t-1} > L_{t-1} - L,\ \text{else}\ 0 \\
+\text{DI}    &= 100 \cdot \text{Wilder}(+\text{DM},\ w) \big/ \text{Wilder}(\text{TR},\ w)
\end{aligned}
$$

**3-input, 1-output** (`FunctorBase<_, 3, 1>`) on `(high, low, close)`. `PlusDI`
shares its smoothing engine with `ADX`, which returns `+DI`, `-DI`, and `ADX`
together; use `PlusDI` on its own when only the positive directional
indicator is needed.

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
    fig.add_trace(go.Scatter(y=plus_di, name='+DI', line=dict(color='green')), row=2, col=1)
    fig.add_trace(go.Scatter(y=minus_di, name='-DI', line=dict(color='orange', dash='dot')), row=2, col=1)
    fig.update_layout(
        title='PlusDI: strength of upward movement',
        yaxis=dict(title='Price'), yaxis2=dict(title='DI (0-100)'),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Reference

Matches `talib.PLUS_DI` bit-exactly (verified to ~1e-14 in `tests/test_third_party_alignment.py`).
