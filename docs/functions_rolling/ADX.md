---
name: ADX
title: Average Directional Index (ADX)
implementation_family: rolling
topics:
- trend
- momentum
tags:
- adx
- wilder
- directional
- talib
- hlc
- average directional index
- trend strength
- dmi
short: Wilder's ADX with +DI and -DI (3 inputs -> 3 outputs).
inputs: 3
outputs: 3
parameters:
- name: window_size
  type: int
  default: 14
  min: 2
  description: 'Wilder smoothing period: the smoother seeds on the first window_size values,
    then each step keeps (window_size - 1) / window_size of the previous value. 14 is Wilder''s
    choice.'
output_names:
- plus_di
- minus_di
- adx
see_also:
- PlusDI
- MinusDI
- DX
- ADXR
- ATR
references:
- Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
nan_policy: ignore
---

# `ADX`

## Description

`ADX` (Average Directional Index, Wilder 1978) measures the strength of a trend, not
its direction. It returns three columns per step, `+DI`, `-DI` and `ADX`:

$$
\begin{aligned}
\text{TR}[t]  &= \max(H - L,\ |H - C_{t-1}|,\ |L - C_{t-1}|) \\
+\text{DM}[t] &= \max(H - H_{t-1},\ 0)\quad\text{if}\quad H - H_{t-1} > L_{t-1} - L,\ \text{else}\ 0 \\
-\text{DM}[t] &= \max(L_{t-1} - L,\ 0)\quad\text{if}\quad L_{t-1} - L > H - H_{t-1},\ \text{else}\ 0 \\
\text{ATR}   &= \text{Wilder}(\text{TR},\ w) \\
+\text{DI}   &= 100 \cdot \text{Wilder}(+\text{DM},\ w) / \text{ATR} \\
-\text{DI}   &= 100 \cdot \text{Wilder}(-\text{DM},\ w) / \text{ATR} \\
\text{DX}    &= 100 \cdot |+\text{DI} - -\text{DI}| / (+\text{DI} + -\text{DI}) \\
\text{ADX}   &= \text{Wilder}(\text{DX},\ w)
\end{aligned}
$$

Inputs are `(high, low, close)`. Use [`PlusDI`](PlusDI.md), [`MinusDI`](MinusDI.md) or
[`DX`](DX.md) when only one of the intermediate series is needed.

## Warmup

| Output | First valid sample |
|---|---|
| `+DI` | `window_size` |
| `-DI` | `window_size` |
| `ADX` | `2 * window_size - 1` (two Wilder smoothers in sequence) |

For the default `window_size=14`, `+DI` and `-DI` start at sample 14 and `ADX` at
sample 27. The outputs match TA-Lib's `PLUS_DI`, `MINUS_DI` and `ADX` exactly.

## Convention note

TA-Lib seeds the Wilder smoother for the DI, DM and TR series differently from its
ATR smoother: it accumulates `w-1` values during warmup, then applies the recurrence
at the `w`-th value (sum form). The ADX smoother itself uses the mean of the first `w`
values as its seed (average form). `ADX` implements both seeds to match TA-Lib, so its
internal true-range average is seeded differently from [`ATR`](ATR.md).

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
    from plotly.subplots import make_subplots
    from screamer import ADX

    rng = np.random.default_rng(0)
    N = 300
    close = 100 + np.cumsum(rng.standard_normal(N))
    high = close + np.abs(rng.standard_normal(N))
    low = close - np.abs(rng.standard_normal(N))

    out = ADX(window_size=14)(high, low, close)
    plus_di  = out[:, 0]
    minus_di = out[:, 1]
    adx      = out[:, 2]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=close, name='Close', line=dict(color='steelblue')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=adx, name='ADX', line=dict(color='crimson')), row=2, col=1)
    fig.add_trace(go.Scatter(y=plus_di, name='+DI', line=dict(color='green', dash='dot')), row=2, col=1)
    fig.add_trace(go.Scatter(y=minus_di, name='-DI', line=dict(color='orange', dash='dot')), row=2, col=1)
    fig.update_layout(
        title='ADX: trend strength over synthetic OHLC data',
        yaxis=dict(title='Price'), yaxis2=dict(title='ADX / DI (0-100)'),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Validation

Matches `talib.PLUS_DI`, `talib.MINUS_DI` and `talib.ADX` to about 1e-14 in
`tests/test_third_party_alignment.py`.
