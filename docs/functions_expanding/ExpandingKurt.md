---
name: ExpandingKurt
title: Expanding kurtosis
implementation_family: expanding
topics:
- statistics
tags:
- kurtosis
- expanding
short: Running bias-corrected excess kurtosis (Fisher) over the whole history.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `ExpandingKurt`

## Description

The `ExpandingKurt` function returns the bias-corrected Fisher excess kurtosis of every sample seen since the last `reset`, using the same estimator as `RollingKurt` and `pandas.Series.expanding().kurt()`. Undefined (`NaN`) until at least four samples have been seen. Memory is `O(1)`.

*Equation*:

$$
y[t] = \frac{n(n+1)}{(n-1)(n-2)(n-3)}\sum_{i=0}^{t}\left(\frac{x[i]-\bar{x}}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}, \quad n=t+1
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
from screamer import ExpandingKurt

x = np.arange(1.0, 11.0)
y = ExpandingKurt()(x)
```

<!-- HELP_END -->
