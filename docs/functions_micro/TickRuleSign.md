---
name: TickRuleSign
title: Tick Rule Sign
implementation_family: micro
topics:
- microstructure
tags:
- trade sign
- tick rule
- lee ready
- classification
- flow
- microstructure
short: Trade sign by the tick rule (+1 up-tick, -1 down-tick, carry on unchanged).
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `TickRuleSign`

## Description

`TickRuleSign` classifies each trade as buyer-initiated (`+1`) or
seller-initiated (`-1`) by the tick rule: a trade above the previous price is a
buy, below is a sell, and an unchanged price carries the previous sign. It is the
simplest trade-sign classifier and needs only the price series. References: the
tick rule, and Lee, Ready (1991), "Inferring Trade Direction from Intraday Data".

The output stays `NaN` until the first price change: the initial bar is always
`NaN` (no prior price), and if subsequent prices are all unchanged there is no
directional tick yet and no sign to carry forward.
