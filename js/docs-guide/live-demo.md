---
title: Live demo
---

# Live demo

**[Open the live demo](https://screamer-labs.github.io/screamer/live-trades.html)** to read a live
crypto tape with screamer, in your browser.

Live trades are noisy. The demo connects to a public exchange feed (Coinbase or Binance, no API
key) and turns the raw prints into three things a trader actually wants:

- **A low-lag price.** `RollingPoly1(window, 0)` is the endpoint of a rolling linear fit. It tracks
  price with far less lag than a moving average, so it de-noises the tape without falling behind.
- **A volume-weighted fair value.** A VWMA built from `RollingSum(price * size) / RollingSum(size)`
  shows where the volume actually traded, not just where the last print landed.
- **Order-flow pressure.** An EW-smoothed, normalized signed-volume imbalance shows whether buyers
  are lifting offers or sellers are hitting bids right now. It is a leading read on short-term
  direction (drawn as a green/red strip under the price), not a trade signal.

The **view** control switches the x-axis between a trade-count clock and a volume clock. On a
volume clock each bar is a fixed slice of traded volume, so a burst of micro-trades collapses into
a couple of bars instead of stretching across the chart.

The **signal** control switches the lower panel between the existing order-flow pressure view and
VPIN. VPIN uses 0.5 BTC volume buckets and reports how one-sided the recent signed flow has been.

The **time** control formats the x-axis in local time or UTC. The chart uses the exchange timestamp
when the feed provides one, and the browser receive time otherwise. A bounded browser buffer keeps
recent raw trades, so changing the view or signal replays the history instead of starting empty.

Each operator is fed one trade at a time as it arrives:

```js
import { ready, RollingPoly1, RollingSum, EwMean } from "@screamer-labs/screamer";

await ready();

const lowLag  = RollingPoly1(30, 0);                    // low-lag price
const sumPV   = RollingSum(50), sumV = RollingSum(50);  // volume-weighted fair value
const flowNet = EwMean(undefined, 50), flowAbs = EwMean(undefined, 50); // order-flow pressure

// sign: +1 when a buyer lifted the offer, -1 when a seller hit the bid
function onTrade(price, size, sign) {
  const denoised = lowLag(price);
  const fair     = sumPV(price * size) / sumV(size);
  const pressure = flowNet(sign * size) / flowAbs(size);   // in [-1, 1], positive = buyers leaning
  // ...draw denoised and fair over the trades, pressure as a green/red strip
}
```

The same operators run just as well on a stored array as on this live feed, so a study done on
history carries over to production unchanged. See [Getting started](./getting-started.md) for the
input regimes and [Node, bundlers, and the browser](./environments.md) for how to load screamer in
each environment.

## Running it yourself

The demo is a single self-contained HTML file at
[`js/examples/live-trades.html`](https://github.com/screamer-labs/screamer/blob/main/js/examples/live-trades.html).
It loads screamer from a CDN, so it needs internet access for both the module and the trade feed.
If the chart stays empty, the selected market may be restricted in your region; switch markets in
the demo.
