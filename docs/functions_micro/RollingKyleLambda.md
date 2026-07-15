---
name: RollingKyleLambda
title: Kyle's Lambda (rolling)
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
short: Rolling-window price-impact slope of return on signed order flow (Kyle 1985).
inputs: 2
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Window length in observations.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: Warmup behaviour.
nan_policy: ignore
see_also:
- RollingBeta
- RollingLinearRegression
- AmihudIlliquidity
---

# `RollingKyleLambda`

## Description

Kyle's lambda (Kyle 1985) is the slope of price change on signed order flow.
It quantifies how much a unit of net buying or selling pressure moves the
price: a high lambda means the market is illiquid (large price impact per
unit of flow), a low lambda means it is liquid.

`RollingKyleLambda(window_size, start_policy)(signed_flow, return_)` returns
the trailing-window OLS slope of `return_` on `signed_flow` over the most
recent `window_size` observations. It is a documented specialization of
`RollingBeta`: internally it calls `RollingBeta(window_size, start_policy)(return_, signed_flow)`.

The two inputs should be co-sampled and aligned in time. A common pipeline is:

1. Compute `signed_flow` with `SignedVolume` (or `OFI`) on tick data.
2. Compute `return_` as a log-return over the same bar.
3. Feed both to `RollingKyleLambda` to obtain a rolling price-impact estimate.

Because the implementation delegates entirely to `RollingBeta`, causality,
warmup behavior (`start_policy`), and the `nan_policy: ignore` contract are
all inherited from the C++ engine.

*Parameters*:

- **`window_size`** (`int`, >= 2): size of the trailing window.
- **`start_policy`** (`str`, default `"strict"`): controls warmup behavior.
  `"strict"` emits `NaN` until `window_size` observations have been seen.
  `"expanding"` uses however many observations are available. `"zero"` fills
  the warmup period with zero.

*Return value*: the price-impact coefficient lambda at each time step. `NaN`
during warmup (under `strict`) or when `signed_flow` has zero variance within
the window.

**Reference**: Kyle, A. S. (1985). "Continuous Auctions and Insider Trading."
*Econometrica*, 53(6), 1315-1335.

<!-- HELP_END -->
