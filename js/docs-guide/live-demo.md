---
title: Live demo
---

# Live demo

**[Open the live demo](https://screamer-labs.github.io/screamer/live-trades.html)** to read a live
crypto tape with screamer, in your browser.

Live trades are noisy. The demo connects to a public exchange feed (Coinbase or Binance, no API
key) and displays the raw prints alongside derived price, volume, and indicator panes:

- **OHLC bars and a low-lag price.** `RollingPoly1(window, 0)` is the endpoint of a rolling linear
  fit. It tracks price with less lag than a moving average.
- **A volume-weighted fair value.** A VWMA built from `RollingSum(price * size) / RollingSum(size)`
  shows where the volume traded, not just where the last print landed.
- **Buy- and sell-initiated volume.** The volume pane keeps the aggressor side visible for each
  plotted sample.
- **A small dashboard of independent windows.** The main market window shows OHLC, low-lag price,
  fair value, and aggressor volume. Two indicator windows use the same Screamer registry but keep
  separate y-axis scales, and a live-tape window keeps the latest prints visible while you change
  the chart view.
- **Category scan.** Switch the layout to `category scan` and choose Order flow & volume, Momentum,
  Trend, Volatility, or Cycle. Every indicator in that family appears as its own small card with
  its own scale and current value, making it possible to scan a family without mixing incompatible
  units on one axis.

The **view** control switches between a trade-count clock, timestamped 10-second OHLC bars, and a
volume clock. On a volume clock each bar is a fixed slice of traded volume, so a burst of
micro-trades collapses into a couple of bars instead of stretching across the chart. Time bars use
the exchange timestamp when available and the browser receive time otherwise.

The **indicator A** picker chooses pane A from about twenty Screamer operators grouped into order
flow and volume (order-flow pressure, VPIN, cumulative volume delta, trade imbalance), momentum
(RSI, rate of change, TRIX, z-score), trend, volatility, and Ehlers cycle reads. The **pane B**
control chooses a second indicator from the same registry. Each is fed the same one-sample-at-a-time
stream. The selectors live in the window headers, so the layout remains the same on desktop and
mobile.

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

For a local copy, serve the examples directory rather than opening the file with `file://`:

```powershell
cd js/examples
py -m http.server 8000
```

Then open <http://localhost:8000/live-trades.html>.
