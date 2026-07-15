---
name: OFI
title: Order-Flow Imbalance
implementation_family: micro
topics:
- microstructure
tags:
- order flow
- imbalance
- ofi
- flow
- microstructure
short: Normalized signed order flow, (buy - sell) / (buy + sell).
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
---

# `OFI`

## Description

`OFI` is the order-flow imbalance: the net of buy-aggressor and sell-aggressor
volume, normalized by their total. It lives in `[-1, 1]`, is positive when buyers
lift the offer more than sellers hit the bid, and is `0` on an empty bucket.

Order-flow imbalance is the standard short-horizon driver of price: over a bar it
explains a large share of the price change (see `RollingKyleLambda`). Feed
`OFI(buy_volume, sell_volume)` as a signal, or pair it with returns to estimate
price impact. References: Cont, Kukanov, Stoikov (2014), "The price impact of
order book events".
