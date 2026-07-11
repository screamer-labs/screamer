---
name: Or
title: Or
implementation_family: math
topics:
- logic
tags:
- logic
- binary
short: Returns 1.0 if either input is nonzero, else 0.0. NaN in either input yields NaN.
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
---

# `Or`

## Description

`Or` applies a logical OR to two aligned input streams: outputs `1.0` when at least one input is nonzero, and `0.0` otherwise. Any nonzero floating-point value (not just `1.0`) counts as true.

*NaN handling*: if either input is NaN at step `t`, the output is NaN.

*Parameters*: `Or` takes no parameters.
