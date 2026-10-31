# Read order flow and liquidity

Order flow is the stream of trades and the side that started each one.
These pages sign trades as buys or sells, measure the imbalance between the
two and how far the price moves per unit traded, estimate the bid-ask spread
and the cost of fills, and model how trades cluster in time. The examples use
six hours of Deribit BTC perpetual trades, daily bars of 20 coins for
illiquidity, and a small synthetic order book for the micro-price.

<!-- tasks:start -->

1. [Sign trades as buys or sells](sign-trades.md)
2. [Measure order-flow imbalance](imbalance.md)
3. [Measure price impact](price-impact.md)
4. [Measure illiquidity from daily bars](illiquidity.md)
5. [Estimate the bid-ask spread from OHLC bars (EDGE, Corwin-Schultz, Abdi-Ranaldo, Roll)](spread-from-bars.md)
6. [Measure the effective and realized spread of your fills](fill-quality.md)
7. [Measure order-flow toxicity (VPIN)](vpin.md)
8. [Compute the micro-price and queue imbalance from quotes](micro-price.md)
9. [Model clustered trade arrivals (Hawkes intensity)](hawkes.md)

```{toctree}
:hidden:

/guide/order-flow/sign-trades
/guide/order-flow/imbalance
/guide/order-flow/price-impact
/guide/order-flow/illiquidity
/guide/order-flow/spread-from-bars
/guide/order-flow/fill-quality
/guide/order-flow/vpin
/guide/order-flow/micro-price
/guide/order-flow/hawkes
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
| Bid-ask spread from bars or prices | `EdgeSpread`, `CorwinSchultzSpread`, `AbdiRanaldoSpread`, `RollSpread` |
| Cost of fills against the mid | `EffectiveSpread`, `RealizedSpread` |
| Order-flow toxicity | `VPIN` |
| Book pressure and fair value | `QueueImbalance`, `MicroPrice` |
| Clustered trade arrivals | `HawkesIntensity` |
