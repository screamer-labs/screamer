---
name: Not
title: Not
implementation_family: math
topics:
- logic
tags:
- logic
- unary
- logical not
- negation
- boolean
short: Returns 0.0 if input is nonzero, 1.0 if zero. NaN propagates.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- And
- Or
- IsNan
---

# `Not`

## Description

`Not` applies a logical NOT to a single input stream: outputs `0.0` when the input is nonzero, and `1.0` when the input is zero. Any nonzero floating-point value (not just `1.0`) counts as true.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import Not

    m = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    print(Not()(m))     # -> [0. 1. 0. 0. 1. 0.]  (flips the mask: 1.0 where the input is zero)
```
