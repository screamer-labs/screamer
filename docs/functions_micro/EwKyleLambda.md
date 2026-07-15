---
name: EwKyleLambda
title: Kyle's Lambda (exponentially weighted)
implementation_family: micro
topics:
- microstructure
- regression
tags:
- price impact
- illiquidity
- market impact
- kyle
- lambda
- microstructure
- ew
short: Exponentially-weighted price-impact slope of return on signed order flow.
inputs: 2
outputs: 1
parameters:
- name: span
  type: float
  default: 20.0
  description: EW span (alpha = 2 / (span + 1)). Controls the effective lookback.
nan_policy: ignore
see_also:
- EwBeta
- RollingKyleLambda
---

# `EwKyleLambda`

## Description

`EwKyleLambda` estimates Kyle's lambda (the price-impact / illiquidity
coefficient) using exponential weighting rather than a fixed rolling window.
Recent observations receive higher weight, so the estimate adapts faster to
changing liquidity conditions than `RollingKyleLambda`.

`EwKyleLambda(span)(signed_flow, return_)` returns the EW regression slope
of `return_` on `signed_flow`. It is a documented specialization of `EwBeta`:
internally it calls `EwBeta(span=span)(return_, signed_flow)`.

Kyle's lambda (Kyle 1985) is the slope of price change on signed order flow:
a high value signals an illiquid market with large price impact per unit of
net flow, a low value signals a liquid one.

*Parameters*:

- **`span`** (`float`, default `20.0`): EW span, where the decay factor is
  `alpha = 2 / (span + 1)`. Larger spans place more weight on older data and
  produce smoother, slower-adapting estimates.

*Return value*: the price-impact coefficient lambda at each time step. The
first output is `NaN` (EwBeta warmup). Subsequent `NaN` values occur when
`signed_flow` has zero variance over the effective EW window.

Because the implementation delegates entirely to `EwBeta`, causality and the
`nan_policy: ignore` contract are inherited from the C++ engine.

**Reference**: Kyle, A. S. (1985). "Continuous Auctions and Insider Trading."
*Econometrica*, 53(6), 1315-1335.

<!-- HELP_END -->
