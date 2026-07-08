---
name: ExpandingSlope
title: Expanding slope
implementation_family: expanding
topics:
- trend
tags:
- slope
- regression
- expanding
short: Running OLS slope of the series against time over the whole history.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `ExpandingSlope`

## Description

The `ExpandingSlope` function returns the ordinary-least-squares slope of the samples seen so far against an implicit integer time axis `x = 0, 1, ..., n-1`. It is the whole-history analogue of `RollingPoly1` with `derivative_order=1`. Undefined (`NaN`) until at least two samples have been seen. Memory is `O(1)`.

*Equation*:

$$
y[t] = \frac{n\sum i\,x[i] - (\sum i)(\sum x[i])}{n\sum i^2 - (\sum i)^2}, \quad i=0..t,\; n=t+1
$$

*Parameters*: none.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in the input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```python
import numpy as np
from screamer import ExpandingSlope

x = np.arange(1.0, 11.0)
y = ExpandingSlope()(x)
```

<!-- HELP_END -->
