---
name: LeeReadySign
title: Lee-Ready Trade Sign
implementation_family: micro
topics:
- microstructure
tags:
- trade sign
- lee ready
- classification
- flow
- microstructure
short: "Trade sign by the Lee-Ready (1991) rule: quote test with tick-rule fallback."
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- TickRuleSign
- SignedVolume
---

# `LeeReadySign`

## Description

`LeeReadySign` classifies each trade as buyer-initiated (`+1`) or
seller-initiated (`-1`) using the Lee-Ready (1991) algorithm. The rule has
two steps.

1. **Quote test**: if the trade price is above the mid-quote, it is a buy
   (`+1`); if it is below, it is a sell (`-1`).
2. **Tick-rule fallback**: if the trade price equals the mid-quote exactly,
   the sign of the most recent price change is used instead (the tick rule).
   An up-tick is a buy; a down-tick is a sell; an unchanged tick carries the
   previous sign.

`LeeReadySign()(price, mid)` returns the signed array. It composes
`TickRuleSign` for the fallback: the tick-rule state is advanced on every
price (not only at-mid samples) so the result is the same whether the
operator is driven by a whole array or one sample at a time (batch == stream).

A missing price or mid (`NaN`) yields `NaN` (nan_policy: ignore).

*Return value*: an array of `+1.0` and `-1.0` trade signs (or `NaN` where
input is missing). The first sample is `NaN` when the tick-rule fallback is
needed at index 0 (no prior price to compute a tick direction).

**Reference**: Lee, C. M. C., and Ready, M. J. (1991). "Inferring Trade
Direction from Intraday Data." *Journal of Finance*, 46(2), 733-746.

<!-- HELP_END -->
