---
name: ExpandingMax
title: Expanding maximum
implementation_family: expanding
topics:
- cumulative
tags:
- max
- expanding
- cumulative
short: Running maximum from t=0.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `ExpandingMax`

## Description

The `ExpandingMax` function returns the running maximum of every sample seen since the last `reset`. It is an alias of `CumMax` exposed under the expanding family and matches `pandas.Series.expanding().max()`.

*Equation*:

$$
y[t] = \max_{0 \le i \le t} x[i]
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
from screamer import ExpandingMax

x = np.arange(1.0, 11.0)
y = ExpandingMax()(x)
```

<!-- HELP_END -->
