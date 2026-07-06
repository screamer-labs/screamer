---
name: Identity
title: Identity
implementation_family: misc
topics:
- arithmetic
tags:
- identity
- noop
short: Pass-through (y = x).
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `Identity`

## Description

The `Identity` function returns each input unchanged. It is a no-op transform whose only purpose is to act as a placeholder in pipelines: a slot that has the same call signature as any other transform but does nothing to the data.

*Equation*:

$$
y[t] = x[t]
$$

*Parameters*: `Identity` takes no parameters.

*NaN handling*: Inputs pass through bit-for-bit, including `NaN` and infinities.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```python
from screamer import Identity, RollingMean

# A pipeline slot that can later be swapped for a real transform.
preproc = Identity()
smoother = RollingMean(10)

stream = (smoother(preproc(v)) for v in source)
```

<!-- HELP_END -->

## Reference

Conceptually equivalent to the identity function `lambda x: x`. It exists as a class so it can be slotted in anywhere a `ScreamerBase` is expected.
