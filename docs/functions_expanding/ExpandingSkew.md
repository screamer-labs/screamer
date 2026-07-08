---
name: ExpandingSkew
title: Expanding skewness
implementation_family: expanding
topics:
- statistics
tags:
- skew
- expanding
short: Running bias-corrected sample skewness (G1) over the whole history.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `ExpandingSkew`

## Description

The `ExpandingSkew` function returns the adjusted Fisher-Pearson standardized moment coefficient (G1) of every sample seen since the last `reset`. It uses the same bias-corrected estimator as `RollingSkew` and `pandas.Series.expanding().skew()`. Undefined (`NaN`) until at least three samples have been seen. Memory is `O(1)`.

*Equation*:

$$
y[t] = \frac{n}{(n-1)(n-2)}\sum_{i=0}^{t}\left(\frac{x[i]-\bar{x}}{s}\right)^3, \quad n=t+1
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
from screamer import ExpandingSkew

x = np.arange(1.0, 11.0)
y = ExpandingSkew()(x)
```

<!-- HELP_END -->
