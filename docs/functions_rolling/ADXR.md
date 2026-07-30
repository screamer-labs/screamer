---
name: ADXR
title: Average Directional Index Rating (ADXR)
implementation_family: rolling
topics:
- trend
tags:
- adxr
- wilder
- dmi
- directional-movement
- talib
- hlc
short: Wilder's ADXR, the mean of ADX now and ADX one window ago.
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

# `ADXR`

## Description

`ADXR` is Wilder's average directional index rating (Wilder, 1978), the mean
of the current `ADX` and the `ADX` value `window_size - 1` bars earlier:

$$
\text{ADXR}[t] = \frac{\text{ADX}[t] + \text{ADX}[t - (\text{window\_size} - 1)]}{2}
$$

**3-input, 1-output** (`FunctorBase<_, 3, 1>`) on `(high, low, close)`. `ADXR`
shares its smoothing engine with `ADX`, `PlusDI`, `MinusDI`, and `DX`. Averaging
`ADX` against its own value one window back damps the swings of `ADX` itself,
giving a slower read on whether trend strength is rising or falling.

### Parameters

**`window_size`** *(int, default 14)*: The Wilder smoothing period, also the
lag used for the second `ADX` term. Because `ADX` itself is only valid from
sample `2 * window_size - 1` onward, the first valid `ADXR` output is at
sample `3 * window_size - 2`.

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
    from screamer import ADXR, ADX

    rng = np.random.default_rng(0)
    N = 300
    close = 100 + np.cumsum(rng.standard_normal(N))
    high = close + np.abs(rng.standard_normal(N))
    low = close - np.abs(rng.standard_normal(N))

    adxr = ADXR(window_size=14)(high, low, close)
    adx = ADX(window_size=14)(high, low, close)[:, 2]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=close, name='Close', line=dict(color='steelblue')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=adx, name='ADX', line=dict(color='crimson', dash='dot')), row=2, col=1)
    fig.add_trace(go.Scatter(y=adxr, name='ADXR', line=dict(color='crimson')), row=2, col=1)
    fig.update_layout(
        title='ADXR: ADX smoothed against its own value one window back',
        yaxis=dict(title='Price'), yaxis2=dict(title='ADX / ADXR (0-100)'),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Reference

Matches `talib.ADXR` on the finite overlap; see `tests/test_dmi.py`.
