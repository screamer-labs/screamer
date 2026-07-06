---
name: Polar2Cart
title: Polar to Cartesian
implementation_family: math
topics:
- trig
tags:
- polar
- conversion
- pair
short: Convert (r, theta) to (x, y).
inputs: 2
outputs: 2
parameters: []
nan_policy: ignore
---

# `Polar2Cart`

## Description

Convert 2D polar coordinates `(r, θ)` to Cartesian coordinates `(x, y)`. The inverse of `Cart2Polar`.

This is a **2-input, 2-output** function (`FunctorBase<_, 2, 2>`). Inputs are paired column-by-column; outputs are stacked along a trailing axis of size 2.

*Equation*:

$$
x[t] = r[t] \cos\theta[t], \qquad y[t] = r[t] \sin\theta[t]
$$

*Parameters*: `Polar2Cart` takes no parameters.

*NaN handling*: A NaN in either input produces NaN in both outputs.

## Output shapes

Same shape rule as `Cart2Polar`:

| You pass... | You get back... |
|---|---|
| `Polar2Cart()(r, θ)` (two scalars) | tuple `(x, y)` |
| Two 1D arrays of shape `(T,)` | array of shape `(T, 2)` |
| Two 2D arrays of shape `(T, K)` | array of shape `(T, K, 2)` |

`out[..., 0]` is `x`, `out[..., 1]` is `y`.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```python
import numpy as np
from screamer import Polar2Cart, Cart2Polar

# Roundtrip with Cart2Polar (inverse pair)
rng = np.random.default_rng(0)
x = rng.standard_normal(100)
y = rng.standard_normal(100)

polar = Cart2Polar()(x, y)              # shape (100, 2)
back = Polar2Cart()(polar[:, 0], polar[:, 1])
np.testing.assert_allclose(back[:, 0], x, atol=1e-12)
np.testing.assert_allclose(back[:, 1], y, atol=1e-12)
```

<!-- HELP_END -->

## Reference

The two outputs are `r * numpy.cos(theta)` and `r * numpy.sin(theta)`, computed in one pass.
