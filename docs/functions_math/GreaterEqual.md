---
name: GreaterEqual
title: GreaterEqual
implementation_family: math
topics:
- logic
tags:
- logic
- binary
- comparison
short: Returns 1.0 if a >= b, else 0.0. NaN in either input yields NaN.
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
---

# `GreaterEqual`

## Description

`GreaterEqual` compares two aligned input streams element-wise and outputs `1.0` where `a >= b` and `0.0` otherwise.

*NaN handling*: if either input is NaN at step `t`, the output is NaN.

*Parameters*: `GreaterEqual` takes no parameters.
