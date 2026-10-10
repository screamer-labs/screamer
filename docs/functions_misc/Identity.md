---
name: Identity
title: Identity
implementation_family: misc
topics:
- arithmetic
tags:
- identity
- noop
- passthrough
- pass-through
- no-op
short: Pass-through (y = x).
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- Lag
- Linear
---

# `Identity`

## Description

The `Identity` function returns each input unchanged. It is a no-op transform whose only purpose is to act as a placeholder in pipelines: a slot that has the same call signature as any other transform but does nothing to the data.

*Equation*:

$$
y[t] = x[t]
$$

Inputs pass through bit-for-bit, including `NaN` and infinities.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import Identity

    x = np.array([1.0, 2.0, 3.0])
    print(Identity()(x))
```

<!-- HELP_END -->

## Notes

`Identity` computes the same values as `lambda x: x`. As a screamer functor it can be used wherever screamer expects one.
