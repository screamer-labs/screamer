---
name: Add
title: Add
implementation_family: math
topics:
- arithmetic
tags:
- arithmetic
- binary
short: Elementwise sum of two aligned streams (x + y).
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
---

# `Add`

## Description

`Add` computes the elementwise sum of two aligned input streams, `x + y`. It takes two inputs and returns one output; a `NaN` in
either input yields `NaN` at that step.

*Parameters*: `Add` takes no parameters.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->
