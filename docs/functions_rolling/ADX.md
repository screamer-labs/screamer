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
short: Wilder's ADX with +DI and -DI (3 inputs -> 3 outputs).
inputs: 3
outputs: 3
parameters:
- name: window_size
  type: int
  default: 14
  min: 2
  description: Period (Wilder's default).
nan_policy: ignore
---

# `ADX`

## Description

`ADX` (Average Directional Index, J. Welles Wilder Jr. 1978) measures **trend strength**, not direction. It is the canonical filter for "are we trending or chopping?" Returns the triple `(+DI, -DI, ADX)` per step:

$$
\begin{aligned}
\text{TR}[t]  &= \max(H - L,\ |H - C_{t-1}|,\ |L - C_{t-1}|) \\
+\text{DM}[t] &= H - C_{t-1}\quad\text{if}\quad H - C_{t-1} > C_{t-1} - L > 0,\ \text{else}\ 0 \\
-\text{DM}[t] &= C_{t-1} - L\quad\text{if}\quad C_{t-1} - L > H - C_{t-1} > 0,\ \text{else}\ 0 \\
\text{ATR}   &= \text{Wilder}(\text{TR},\ w) \\
+\text{DI}   &= 100 \cdot \text{Wilder}(+\text{DM},\ w) / \text{ATR} \\
-\text{DI}   &= 100 \cdot \text{Wilder}(-\text{DM},\ w) / \text{ATR} \\
\text{DX}    &= 100 \cdot |+\text{DI} - -\text{DI}| / (+\text{DI} + -\text{DI}) \\
\text{ADX}   &= \text{Wilder}(\text{DX},\ w)
\end{aligned}
$$

**3-input, 3-output** (`FunctorBase<_, 3, 3>`) on `(high, low, close)`. Outputs are `(out[..., 0]=+DI, out[..., 1]=-DI, out[..., 2]=ADX)`.

## Parameters and warmup

- `window_size` (int, default `14`, the Wilder convention).

| Output | First valid sample |
|---|---|
| `+DI` | `window_size` |
| `-DI` | `window_size` |
| `ADX` | `2 * window_size - 1` (double-Wilder warmup) |

For the default `window_size=14`, `+DI`/`-DI` start at sample 14 and `ADX` at sample 27. Matches TA-Lib's `PLUS_DI` / `MINUS_DI` / `ADX` bit-exactly.

## Convention note

TA-Lib's Wilder smoother for the DI/DM/TR triplet uses a slightly different seed than its ATR smoother: accumulate `w-1` values during warmup, then apply the recurrence at the `w`-th value (sum form). The ADX smoother itself uses the standard SMA-of-`w`-values seed (average form). `screamer.ADX` implements both conventions inline to match TA-Lib exactly; it does **not** share state with the existing `ATR` class.

## Usage


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

## Reference

Matches `talib.PLUS_DI`, `talib.MINUS_DI`, `talib.ADX` bit-exactly (verified to ~1e-14 in `tests/test_third_party_alignment.py`).
