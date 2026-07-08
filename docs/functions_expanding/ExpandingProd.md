---
name: ExpandingProd
title: Expanding product
implementation_family: expanding
topics:
- cumulative
tags:
- product
- expanding
- cumulative
short: Running product from t=0.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `ExpandingProd`

## Description

The `ExpandingProd` function returns the running product of every sample seen since the last `reset`. It is an alias of `CumProd` exposed under the expanding family and matches `numpy.cumprod` (skipping NaN).

*Equation*:

$$
y[t] = \prod_{i=0}^{t} x[i]
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
from screamer import ExpandingProd

x = np.arange(1.0, 11.0)
y = ExpandingProd()(x)
```

<!-- HELP_END -->
