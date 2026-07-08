---
name: ExpandingSum
title: Expanding sum
implementation_family: expanding
topics:
- cumulative
tags:
- sum
- expanding
- cumulative
short: Running sum from t=0.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `ExpandingSum`

## Description

The `ExpandingSum` function returns the running sum of every sample seen since the last `reset`. It is an alias of `CumSum` exposed under the expanding family and matches `pandas.Series.expanding().sum()`.

*Equation*:

$$
y[t] = \sum_{i=0}^{t} x[i]
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
from screamer import ExpandingSum

x = np.arange(1.0, 11.0)
y = ExpandingSum()(x)
```

<!-- HELP_END -->
