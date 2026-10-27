# Read order flow and liquidity

Order flow is the stream of trades and the side that started each one.
These pages sign trades as buys or sells, measure the imbalance between the
two, and measure how far the price moves per unit traded. The examples use
six hours of Deribit BTC perpetual trades, and daily bars of 20 coins for
illiquidity.

<!-- tasks:start -->

1. [Sign trades as buys or sells](sign-trades.md)
2. [Measure order-flow imbalance](imbalance.md)
3. [Measure price impact](price-impact.md)
4. [Measure illiquidity from daily bars](illiquidity.md)

```{toctree}
:hidden:

/guide/order-flow/sign-trades
/guide/order-flow/imbalance
/guide/order-flow/price-impact
/guide/order-flow/illiquidity
/notebooks/11-microstructure-order-flow
/notebooks/12-microstructure-price-impact
/notebooks/13-microstructure-toxicity-and-book
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| Build bars with buy and sell volume | `Resample` |
| Sign trades as buys or sells | `TickRuleSign`, `LeeReadySign` |
| Signed volume per trade | `SignedVolume` |
| Order-flow imbalance | `OFI`, `RollingOrderImbalance` |
| Price impact | `RollingKyleLambda`, `EwKyleLambda`, `AmihudIlliquidity` |
