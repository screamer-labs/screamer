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
