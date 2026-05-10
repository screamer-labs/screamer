# `Hypot`

## Description

Two-argument Euclidean distance: `Hypot(x, y) = sqrt(x*x + y*y)`. Computes `sqrt(x² + y²)` in a numerically stable way that avoids overflow for very large `|x|` or `|y|` and underflow for very small ones.

This is a **2-input, 1-output** function (`FunctorBase<_, 2, 1>`). Inputs are paired column-by-column for arrays.

*Equation*:

$$
y[t] = \sqrt{x_1[t]^2 + x_2[t]^2}
$$

*Parameters*: `Hypot` takes no parameters.

*NaN handling*: A NaN in either input produces a NaN output.

## Usage Example

```python
import numpy as np
from screamer import Hypot

# Scalar pair
Hypot()(3.0, 4.0)              # 5.0

# Two parallel 1D arrays
ux = np.random.randn(100)
uy = np.random.randn(100)
speed = Hypot()(ux, uy)        # shape (100,)

# Two parallel 2D arrays (column-by-column pairing)
UX = np.random.randn(100, 4)
UY = np.random.randn(100, 4)
Hypot()(UX, UY).shape          # (100, 4)
```

## Reference

Equivalent to `numpy.hypot`. Also returned as the radial component of `Cart2Polar`.
