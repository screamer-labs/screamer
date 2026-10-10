---
name: Or
title: Or
implementation_family: math
topics:
- logic
tags:
- logic
- binary
- logical or
- boolean
short: Returns 1.0 if either input is nonzero, else 0.0. NaN in either input yields NaN.
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- And
- Not
- Where
---

# `Or`

## Description

`Or` applies a logical OR to two aligned input streams: outputs `1.0` when at least one input is nonzero, and `0.0` otherwise. Any nonzero floating-point value (not just `1.0`) counts as true.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import Or

    m1 = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    m2 = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    print(Or()(m1, m2))     # -> [1. 0. 1. 1. 1. 1.]  (1.0 where at least one mask is nonzero)
```
