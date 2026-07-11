---
name: IsFinite
title: IsFinite
implementation_family: math
topics:
- logic
- missing-data
tags:
- logic
- unary
- nan-aware
short: Returns 1.0 for finite values, 0.0 for NaN or inf. Does not propagate NaN.
inputs: 1
outputs: 1
parameters: []
nan_policy: nan-aware
---

# `IsFinite`

## Description

`IsFinite` classifies each input element: outputs `1.0` when the input is a finite number (not NaN and not infinite), and `0.0` for NaN, positive infinity, or negative infinity.

Unlike most screamer functions, `IsFinite` does **not** propagate NaN; it converts NaN into a definite `0.0` flag. Use it to build validity masks that distinguish well-defined observations from missing or overflow values.

*Parameters*: `IsFinite` takes no parameters.
