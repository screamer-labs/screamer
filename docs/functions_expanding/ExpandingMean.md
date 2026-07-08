---
name: ExpandingMean
title: Expanding mean
implementation_family: expanding
topics:
- statistics
tags:
- mean
- expanding
- cumulative
short: Running mean over the whole history since the last reset.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `ExpandingMean`

## Description

The `ExpandingMean` function returns the arithmetic mean of every sample seen since the start of the stream (or since the last `reset`). It is the whole-history analogue of `RollingMean` and matches `pandas.Series.expanding().mean()`. Memory is `O(1)`.

*Equation*:

$$
y[t] = \frac{1}{t+1} \sum_{i=0}^{t} x[i]
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
from screamer import ExpandingMean

x = np.arange(1.0, 11.0)
y = ExpandingMean()(x)
```

<!-- HELP_END -->
