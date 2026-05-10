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

## Reference

There is no direct numpy / pandas / TA-Lib counterpart -- it is a primitive intended for composition. The single-input sibling is `Linear(scale, shift)`.
