---
name: ExpandingMin
title: Expanding minimum
implementation_family: expanding
topics:
- cumulative
tags:
- min
- expanding
- cumulative
short: Running minimum from t=0.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `ExpandingMin`

## Description

The `ExpandingMin` function returns the running minimum of every sample seen since the last `reset`. It is an alias of `CumMin` exposed under the expanding family and matches `pandas.Series.expanding().min()`.

*Equation*:

$$
y[t] = \min_{0 \le i \le t} x[i]
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
from screamer import ExpandingMin

x = np.arange(1.0, 11.0)
y = ExpandingMin()(x)
```

<!-- HELP_END -->
