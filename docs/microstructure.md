# Market microstructure

Screamer ships a set of operators that read the trade and order-flow tape: who is
trading, how their flow moves price, and how liquid the market is. They turn a raw
stream of trades into the signals that short-horizon models are built on. Like
every screamer operator they are causal and run in the C++ core, so they give
identical results in batch and live streaming.

This page maps out what is available and shows a typical use. For each operator's
exact signature and formula, follow its link to the reference page; for full
worked examples on real data, see the two notebooks linked at the end.

## What's here

The operators group by what they measure.

**Signing trades** recovers whether each trade was buyer- or seller-initiated,
which a public tape usually does not tell you:

- [`TickRuleSign`](functions_micro/TickRuleSign.md): the tick rule, from price
  changes alone.
- [`LeeReadySign`](functions_micro/LeeReadySign.md): the Lee-Ready quote test with
  a tick-rule fallback, when a mid-quote is available.
- [`SignedVolume`](functions_micro/SignedVolume.md): a sign times size, giving
  aggressor-signed order flow.
- [`BulkVolumeClassifier`](functions_micro/BulkVolumeClassifier.md): the
  buy-initiated share of a bar from its return and volatility, with no per-trade
  signs.

**Order-flow imbalance** measures net buying pressure, the standard short-horizon
driver of price:

- [`OFI`](functions_micro/OFI.md): normalized imbalance, `(buy - sell) / (buy + sell)`.
- [`RollingOrderImbalance`](functions_micro/RollingOrderImbalance.md): the trailing
  sum of signed flow.
- [`QueueImbalance`](functions_micro/QueueImbalance.md): the same normalized
  imbalance applied to the resting L1 book sizes rather than to trade flow.
- [`VPIN`](functions_micro/VPIN.md): order-flow toxicity, the average one-sidedness
  of flow over a volume clock (Easley-Lopez de Prado-O'Hara).

**Price impact and liquidity** measure how far flow moves price and what trading
costs:

- [`RollingKyleLambda`](functions_micro/RollingKyleLambda.md) and
  [`EwKyleLambda`](functions_micro/EwKyleLambda.md): Kyle's lambda, the price move
  per unit of signed flow.
- [`AmihudIlliquidity`](functions_micro/AmihudIlliquidity.md): price move per
  dollar traded, comparable across assets.
- [`RollSpread`](functions_micro/RollSpread.md): the effective bid-ask spread
  implied by trade prices.
- [`Propagator`](functions_micro/Propagator.md): the Bouchaud model, where impact
  builds and then relaxes through a decaying kernel.
- [`MicroPrice`](functions_micro/MicroPrice.md): an imbalance-weighted fair value
  that leans toward the thinner side of the book (Stoikov).

**Event intensity** measures the clustering of activity:

- [`HawkesIntensity`](functions_micro/HawkesIntensity.md): a self-exciting arrival
  rate, where each trade briefly raises the expected rate of the next.

## A typical use case

Start from a trade tape (price and size per trade, with timestamps), sign each
trade, turn it into signed order flow, aggregate the flow into bars, and estimate
the price impact of that flow:

```python
import numpy as np
from screamer import TickRuleSign, SignedVolume, RollingKyleLambda, Resample, LogReturn

# price, size, and millisecond timestamps, one row per trade
sign = TickRuleSign()(price)              # +1 buyer-initiated, -1 seller-initiated
flow = SignedVolume()(sign, size)         # aggressor-signed order flow

# one-minute bars: net signed flow and the bar's close-to-close return
net, _     = Resample(freq=60_000, agg="sum")(flow, ts)
close, idx = Resample(freq=60_000, agg="last")(price, ts)
ret        = LogReturn(1)(close)

# price impact: the slope of return on signed flow, over a trailing window
kyle_lambda = RollingKyleLambda(30)(net, ret)
```

The same chain runs unchanged on a live feed, one trade at a time.

## Worked examples

Two notebooks work these models end to end on a real slice of Deribit trades:

- [Order flow and trade signing](notebooks/11-microstructure-order-flow): infer
  the trade sign with the tick rule and check it against the venue's true
  aggressor sign, then build signed volume, order-flow imbalance, rolling
  imbalance, and trade-arrival intensity.
- [Price impact and liquidity](notebooks/12-microstructure-price-impact): estimate
  Kyle's lambda, Amihud illiquidity, the Roll spread, and the Bouchaud propagator,
  and compare liquidity across BTC and ETH.

For the complete list with signatures and formulas, see the function index for
each group: [Trade signing](by_topic/trade-signing),
[Order-flow imbalance](by_topic/order-flow-imbalance),
[Price impact & liquidity](by_topic/price-impact), and
[Order-flow arrivals](by_topic/order-flow-arrivals).
