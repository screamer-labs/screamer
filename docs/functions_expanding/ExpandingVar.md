---
name: ExpandingVar
title: Expanding variance
implementation_family: expanding
topics:
- statistics
- volatility
tags:
- variance
- expanding
short: Running sample variance (ddof=1) over the whole history.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `ExpandingVar`

## Description

The `ExpandingVar` function returns the sample variance (delta degrees of freedom `ddof=1`) of every sample seen since the last `reset`. It shares the ddof=1 convention with `RollingVar` and `pandas.Series.expanding().var()`. Undefined (`NaN`) until at least two samples have been seen. Memory is `O(1)`.

*Equation*:

$$
y[t] = \frac{1}{n-1}\sum_{i=0}^{t}(x[i]-\bar{x})^2, \quad n=t+1
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
from screamer import ExpandingVar

x = np.arange(1.0, 11.0)
y = ExpandingVar()(x)
```

<!-- HELP_END -->
