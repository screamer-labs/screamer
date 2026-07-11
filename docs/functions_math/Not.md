---
name: Not
title: Not
implementation_family: math
topics:
- logic
tags:
- logic
- unary
short: Returns 0.0 if input is nonzero, 1.0 if zero. NaN propagates.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `Not`

## Description

`Not` applies a logical NOT to a single input stream: outputs `0.0` when the input is nonzero, and `1.0` when the input is zero. Any nonzero floating-point value (not just `1.0`) counts as true.

*NaN handling*: NaN input yields NaN output.

*Parameters*: `Not` takes no parameters.
