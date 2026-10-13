---
name: Linear2
title: Linear (two-input affine)
implementation_family: math
topics:
- arithmetic
tags:
- affine
- linear
- pair
- weighted sum
- blend
- difference
short: 'Two-input affine combination: a*x + b*y + c.'
inputs: 2
outputs: 1
parameters:
- name: a
  type: float
  default: 1.0
  description: Coefficient that multiplies the first input. inf and -inf are taken, as numpy takes them.
- name: b
  type: float
  default: 1.0
  description: Coefficient that multiplies the second input. inf and -inf are taken, as numpy takes them.
- name: c
  type: float
  default: 0.0
  description: Constant added to the weighted sum. inf and -inf are taken, as numpy takes them.
nan_policy: ignore
see_also:
- Linear
- Add
- Sub
---

# `Linear2`

## Description

Two-input affine combination:

$$
f(x, y) = a \cdot x + b \cdot y + c
$$

It takes two inputs and returns one output. For 2D arrays the inputs are paired column by column.

Composed with element-wise transforms such as `Sign`, `Relu` and `Sigmoid`, it expresses common idioms in one chain:

| Expression | Compact form | Meaning |
|---|---|---|
| `Linear2(1, -1, 0)(x, y)` | $x - y$ | signed difference |
| `Sign() o Linear2(1, -1, 0)` | $\text{sign}(x - y)$ | is `x` greater than `y`? (`+1` / `0` / `−1`) |
| `Relu() o Linear2(1, -1, 0)` | $\max(x - y, 0)$ | positive excess |
| `Linear2(0.7, 0.3, 0)` | $0.7x + 0.3y$ | weighted blend |
| `Sigmoid() o Linear2(a, b, c)` | $\sigma(ax + by + c)$ | logistic mix |


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

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

<!-- HELP_END -->

## Visual example: positive excess of a price over its trend

`Relu(Linear2(1, -1)(price, trend))` returns `max(price - trend, 0)`: zero when the price is at or below the trend, otherwise the gap.

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

It has no direct numpy, pandas or TA-Lib counterpart and is meant as a building block for composition. The single-input sibling is `Linear(scale, shift)`.
