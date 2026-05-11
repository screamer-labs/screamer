---
name: Identity
title: Identity
implementation_family: misc
topics:
- transforms
tags:
- identity
- noop
short: Pass-through (y = x).
inputs: 1
outputs: 1
parameters: []
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

<!-- HELP_END -->

## Usage Example

```python
from screamer import Identity, RollingMean

# A pipeline slot that can later be swapped for a real transform.
preproc = Identity()
smoother = RollingMean(10)

stream = (smoother(preproc(v)) for v in source)
```

## Reference

Conceptually equivalent to the identity function `lambda x: x`. It exists as a class so it can be slotted in anywhere a `ScreamerBase` is expected.
