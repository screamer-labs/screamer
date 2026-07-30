---
name: DX
title: Directional index (DX)
implementation_family: rolling
topics:
- trend
tags:
- wilder
- dmi
- directional-movement
- talib
- hlc
short: Wilder directional index, the pre-average input to ADX.
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

# `DX`

## Description

`DX` is Wilder's directional index (Wilder, 1978), 100 times the absolute
difference of `+DI` and `-DI` divided by their sum:

$$
\text{DX} = 100 \cdot \frac{|+\text{DI} - -\text{DI}|}{+\text{DI} + -\text{DI}}
$$

**3-input, 1-output** (`FunctorBase<_, 3, 1>`) on `(high, low, close)`. `DX`
shares its smoothing engine with `ADX`, `PlusDI`, and `MinusDI`; `ADX` is the
Wilder moving average of `DX` itself. Use `DX` on its own when the raw,
unsmoothed directional index is needed, for example to build a different
average over it than `ADX` applies.

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
    from screamer import DX, ADX

    rng = np.random.default_rng(0)
    N = 300
    close = 100 + np.cumsum(rng.standard_normal(N))
    high = close + np.abs(rng.standard_normal(N))
    low = close - np.abs(rng.standard_normal(N))

    dx = DX(window_size=14)(high, low, close)
    adx = ADX(window_size=14)(high, low, close)[:, 2]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=close, name='Close', line=dict(color='steelblue')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=dx, name='DX', line=dict(color='crimson')), row=2, col=1)
    fig.add_trace(go.Scatter(y=adx, name='ADX', line=dict(color='crimson', dash='dot')), row=2, col=1)
    fig.update_layout(
        title='DX: directional index and its Wilder average',
        yaxis=dict(title='Price'), yaxis2=dict(title='DX / ADX (0-100)'),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Reference

Matches `talib.DX` on the finite overlap; see `tests/test_dmi.py`.
