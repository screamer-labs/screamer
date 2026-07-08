---
name: ExpandingStd
title: Expanding standard deviation
implementation_family: expanding
topics:
- statistics
- volatility
tags:
- std
- expanding
short: Running sample standard deviation (ddof=1) over the whole history.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `ExpandingStd`

## Description

The `ExpandingStd` function returns the sample standard deviation (`ddof=1`) of every sample seen since the last `reset`; it is the square root of `ExpandingVar` and matches `pandas.Series.expanding().std()`. Undefined (`NaN`) until at least two samples have been seen. Memory is `O(1)`.

*Equation*:

$$
y[t] = \sqrt{\frac{1}{n-1}\sum_{i=0}^{t}(x[i]-\bar{x})^2}, \quad n=t+1
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
from screamer import ExpandingStd

x = np.arange(1.0, 11.0)
y = ExpandingStd()(x)
```

<!-- HELP_END -->
