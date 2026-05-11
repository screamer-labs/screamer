---
name: Linear2
title: Linear (two-input affine)
implementation_family: math
topics:
- math
- regression
tags:
- affine
- linear
- pair
short: 'Two-input affine combination: a*x + b*y + c.'
inputs: 2
outputs: 1
parameters:
- name: a
  type: float
  default: 1.0
  description: Coefficient on the first input.
- name: b
  type: float
  default: 1.0
  description: Coefficient on the second input.
- name: c
  type: float
  default: 0.0
  description: Additive offset.
---

# `Linear2`

## Description

Two-input affine combination:

$$
f(x, y) = a \cdot x + b \cdot y + c
$$

Stateless 2→1 function (`FunctorBase<_, 2, 1>`). Inputs are paired column-by-column for arrays.

The class is small but composes nicely with the existing element-wise transforms (`Sign`, `Relu`, `Sigmoid`, ...) to build common idioms in a single chain:

| Expression | Compact form | Meaning |
|---|---|---|
| `Linear2(1, -1, 0)(x, y)` | $x - y$ | signed difference |
| `Sign() o Linear2(1, -1, 0)` | $\text{sign}(x - y)$ | is `x` greater than `y`? (`+1` / `0` / `−1`) |
| `Relu() o Linear2(1, -1, 0)` | $\max(x - y, 0)$ | positive excess |
| `Linear2(0.7, 0.3, 0)` | $0.7x + 0.3y$ | weighted blend |
| `Sigmoid() o Linear2(a, b, c)` | $\sigma(ax + by + c)$ | logistic mix |

*Parameters*:

- `a` (float): coefficient on the first input.
- `b` (float): coefficient on the second input.
- `c` (float, optional): additive constant. Defaults to `0.0`.

*NaN handling*: a NaN in either input produces a NaN output (arithmetic propagation).

<!-- HELP_END -->

## Usage Example

```python
import numpy as np
from screamer import Linear2, Sign, Relu

x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([2.0, 2.0, 2.0, 2.0])

Linear2(1, -1, 0)(x, y)        # array([-1., 0., 1., 2.])
Sign()(Linear2(1, -1, 0)(x, y))     # array([-1., 0., 1., 1.])
Relu()(Linear2(1, -1, 0)(x, y))     # array([0., 0., 1., 2.])

# Two parallel 2D arrays (column-by-column pairing)
X = np.random.randn(100, 4)
Y = np.random.randn(100, 4)
Linear2(0.5, 0.5)(X, Y).shape       # (100, 4)
```

## Visual example: positive excess of a price over its trend

`Relu(Linear2(1, -1)(price, trend))` returns `max(price - trend, 0)`: zero when the price is at or below the trend, otherwise the gap. A natural way to highlight regimes where the price is *above* its smoothed trendline.

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import Linear2, Relu, EwMean

    rng = np.random.default_rng(0)
    n = 400
    # A drifting price with occasional jumps above and below.
    noise = rng.normal(0.0, 0.5, n)
    drift = np.linspace(0, 6, n)
    bumps = 1.5 * np.sin(np.linspace(0, 4 * np.pi, n))
    price = drift + bumps + noise

    trend = EwMean(span=30)(price)
    excess = Relu()(Linear2(1, -1)(price, trend))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=price, mode='lines', name='Price',
                             line=dict(color='steelblue')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=trend, mode='lines', name='EW trend (span=30)',
                             line=dict(color='gray', dash='dash')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=excess, mode='lines',
                             name='Relu(price - trend)',
                             fill='tozeroy',
                             line=dict(color='red')),
                  row=2, col=1)
    fig.update_layout(
        title="Positive excess: Relu(Linear2(1, -1)(price, trend))",
        xaxis_title="Index",
        yaxis_title="Price",
        yaxis2_title="Excess",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

## Reference

There is no direct numpy / pandas / TA-Lib counterpart -- it is a primitive intended for composition. The single-input sibling is `Linear(scale, shift)`.
