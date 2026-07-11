---
name: Where
title: Where
implementation_family: math
topics:
- logic
tags:
- logic
- ternary
- conditional
short: Returns a if mask is nonzero, b otherwise. NaN mask yields NaN.
inputs: 3
outputs: 1
parameters: []
nan_policy: ignore
---

# `Where`

## Description

`Where` is a conditional element-wise selector over three aligned input streams. Given inputs `mask`, `a`, and `b`:

- output is `a` when `mask` is nonzero (any nonzero value counts as true)
- output is `b` when `mask` is zero

If `mask` is NaN, the output is NaN. If the selected branch (`a` when mask is nonzero, `b` when zero) is NaN, that NaN passes through as the output.

*Parameters*: `Where` takes no parameters.
