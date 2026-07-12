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
short: Returns 1.0 if the input is NaN, else 0.0. Does not propagate NaN.
inputs: 1
outputs: 1
parameters: []
nan_policy: nan-aware
---

# `IsNan`

## Description

`IsNan` classifies each input element: outputs `1.0` when the input is NaN, and `0.0` for any finite or infinite value.

Unlike most screamer functions, `IsNan` does **not** propagate NaN; instead, it converts NaN into a definite `1.0` flag. This makes it useful for building masks that mark missing data positions before passing them to `Where` or other logic operators.

*Parameters*: `IsNan` takes no parameters.

## Examples

### Usage example

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import IsNan

    x = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    print(IsNan()(x))     # -> [0. 1. 0. 1. 0.]  (1.0 where the input is NaN)
```
