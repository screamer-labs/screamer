---
name: Equal
title: Equal
implementation_family: math
topics:
- logic
tags:
- logic
- binary
- comparison
short: Returns 1.0 if a == b, else 0.0. NaN in either input yields NaN.
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
---

# `Equal`

## Description

`Equal` compares two aligned input streams element-wise and outputs `1.0` where `a == b` and `0.0` otherwise.

*NaN handling*: if either input is NaN at step `t`, the output is NaN (NaN is not equal to anything, including itself, by IEEE 754).

*Parameters*: `Equal` takes no parameters.
