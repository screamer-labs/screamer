---
name: Atan2
title: Two-argument arctangent
implementation_family: math
topics:
- trig
tags:
- trigonometry
- atan2
- polar
- pair
short: Signed angle of (x, y) from the positive x-axis (numpy.arctan2 order).
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
---

# `Atan2`

## Description

Two-argument inverse tangent: the signed angle of the point `(x, y)` from the positive x-axis, in the range $(-\pi, \pi]$. The argument order matches `numpy.arctan2`: **`y` first, `x` second**.

This is a **2-input, 1-output** function (`FunctorBase<_, 2, 1>`). Inputs are paired column-by-column for arrays.

Unlike `Atan`, `Atan2` returns the *correct quadrant* because it sees both `x` and `y` separately. `Atan(y / x)` collapses two cases that `Atan2(y, x)` keeps distinct.

*Equation*:

$$
\theta[t] = \operatorname{atan2}(y[t], x[t])
$$

*Parameters*: `Atan2` takes no parameters.

*NaN handling*: A NaN in either input produces a NaN output.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```python
import numpy as np
from screamer import Atan2

Atan2()(1.0,  0.0)             # +pi/2  (point on +y axis)
Atan2()(-1.0, 0.0)             # -pi/2  (point on -y axis)
Atan2()(0.0, -1.0)             # +pi    (point on -x axis)

# Two parallel arrays - wind direction from u/v components
ux = np.random.randn(100)
uy = np.random.randn(100)
direction = Atan2()(uy, ux)
```

<!-- HELP_END -->

## Reference

Equivalent to `numpy.arctan2`. Also returned as the angular component of `Cart2Polar`.
