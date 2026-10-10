---
name: IsNan
title: IsNan
implementation_family: math
topics:
- logic
- missing-data
tags:
- logic
- unary
- nan-aware
- isnan
- missing
- nan mask
short: Returns 1.0 if the input is NaN, else 0.0. Does not propagate NaN.
inputs: 1
outputs: 1
parameters: []
nan_policy: nan-aware
see_also:
- IsFinite
- FillNa
- Ffill
- Where
---

# `IsNan`

## Description

`IsNan` classifies each input element: outputs `1.0` when the input is NaN, and `0.0` for any finite or infinite value.

A `NaN` input gives `1.0`, so the output is never `NaN`. Use it to build a mask of missing positions for `Where` or the other logic operators.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `nan-aware`.** This function is built to process `NaN` inputs, as the Description explains. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import IsNan

    x = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    print(IsNan()(x))     # -> [0. 1. 0. 1. 0.]  (1.0 where the input is NaN)
```
