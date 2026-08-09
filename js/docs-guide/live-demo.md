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
- **A two-column analyst workspace.** The left side keeps the market, controls, price, status, and
  rate metrics together. The right side groups every indicator into horizontally scrollable rows by
  family, with one small card per indicator. Each card keeps its own scale and current value, so
  RSI, VPIN, CVD, and volatility measures can be scanned together without mixing units. On a phone,
  the market plot comes first and the indicator cards become a single full-width vertical column.

The **view** control switches between a trade-count clock, timestamped 10-second OHLC bars, and a
volume clock. On a volume clock each bar is a fixed slice of traded volume, so a burst of
micro-trades collapses into a couple of bars instead of stretching across the chart. Time bars use
the exchange timestamp when available and the browser receive time otherwise.

All indicators are fed the same one-sample-at-a-time stream. The indicator cards are grouped by
their registry family: order flow & volume, momentum, trend, volatility, and Ehlers cycle reads.

The **time** control formats the x-axis in local time or UTC. The chart uses the exchange timestamp
when the feed provides one, and the browser receive time otherwise. A bounded browser buffer keeps
recent raw trades, so changing the bar clock replays the history instead of starting empty. The
buffer is maintained separately for each market, so switching away and back restores that market's
recent view without mixing feeds. The browser also retains that market's active operator state, so
the indicator cards return with their values instead of warming up from an empty series.

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
