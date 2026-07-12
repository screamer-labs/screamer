---
name: And
title: And
implementation_family: math
topics:
- logic
tags:
- logic
- binary
short: Returns 1.0 if both inputs are nonzero, else 0.0. NaN in either input yields NaN.
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
---

# `And`

## Description

`And` applies a logical AND to two aligned input streams: outputs `1.0` when both inputs are nonzero, and `0.0` otherwise. Any nonzero floating-point value (not just `1.0`) counts as true.

*NaN handling*: if either input is NaN at step `t`, the output is NaN.

*Parameters*: `And` takes no parameters.

## Examples

### Usage example

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import And

    m1 = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    m2 = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    print(And()(m1, m2))     # -> [1. 0. 0. 1. 0. 1.]  (1.0 only where both masks are nonzero)
```
