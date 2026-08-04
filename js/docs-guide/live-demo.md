---
title: Live demo
---

# Live demo

**[Open the live demo](https://screamer-labs.github.io/screamer/live-trades.html)** to watch
screamer smoothing operators run on a live crypto trade feed in your browser.

The page connects to a public exchange WebSocket (Coinbase or Binance, no API key), keeps a
fixed window of the most recent trades, and pushes each incoming trade through four operators as
it arrives. It plots the raw price with an EWMA (span 12), a slower EWMA (span 40), a 30-trade
simple moving average, and an Ehlers SuperSmoother, updating on every trade.

## What it shows

Each operator is called one trade at a time, scalar in and scalar out:

```js
import { ready, EwMean, RollingMean, SuperSmoother } from "@screamer-labs/screamer";

await ready();

const fast = EwMean(undefined, 12);   // EWMA, span 12
const slow = EwMean(undefined, 40);   // EWMA, span 40
const sma  = RollingMean(30);         // simple moving average, 30 trades
const ss   = SuperSmoother(30);       // Ehlers SuperSmoother, period 30

// on each trade from the feed:
function onTrade(price) {
  const a = fast(price);
  const b = slow(price);
  const c = sma(price);   // NaN until the 30-trade window fills
  const d = ss(price);
  // ...plot a, b, c, d against the raw price
}
```

These are the same operators, with the same numerics, you would call on a stored array. Feeding
them a live stream one value at a time and feeding them a historical array in one pass produce
identical results, so a strategy tested on history runs unchanged on the live feed. See
[Parity with Python](./parity.md) and the four input regimes in
[Getting started](./getting-started.md).

## Running it yourself

The demo is a single self-contained HTML file at
[`js/examples/live-trades.html`](https://github.com/screamer-labs/screamer/blob/main/js/examples/live-trades.html).
It loads screamer from a CDN, so it needs internet access for both the module and the trade feed.
If the chart stays empty, the default exchange may be restricted in your region; switch the feed
selector to the other exchange.
