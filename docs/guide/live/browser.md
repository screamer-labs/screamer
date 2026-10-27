---
exempt_shape: true
---

# Run screamer in the browser

The JavaScript package `@screamer-labs/screamer` runs the same C++ functions,
compiled to WebAssembly, in a web page: load it from a CDN, `await ready()`
once, and call each function with one value per event, as in Python.

```html
<script type="module">
  import { ready, EwMean, RollingZscore }
    from "https://cdn.jsdelivr.net/npm/@screamer-labs/screamer/dist/index.js";
  await ready();

  const level = EwMean({ span: 50 });
  const score = RollingZscore(200);

  const ws = new WebSocket("wss://stream.binance.com:9443/ws/btcusdt@aggTrade");
  ws.onmessage = (event) => {
    const price = parseFloat(JSON.parse(event.data).p);
    const z = score(price - level(price));
    if (Math.abs(z) > 3) console.log("alert", price, z.toFixed(1));
  };
</script>
```

This is the handler of [Run functions in a live event loop](live-loop.md) on the
Binance BTC/USDT trade feed. A JavaScript constructor takes its parameters by
name in one options object, `EwMean({ span: 50 })`, or by position,
`RollingZscore(200)`. The module needs Chrome 95, Firefox 100, Safari 15.2 or
later. To install it for Node.js or a bundler, run
`npm i @screamer-labs/screamer`.

## Go deeper

- [JavaScript and the browser](../../javascript.md): parity with Python, calling conventions, backtests, and the differences between the two packages.
- [Live trades demo](https://screamer-labs.github.io/screamer/live-trades.html): bars and indicators on every trade of a public exchange feed, in one HTML file.
- [screamer.js documentation](https://screamer-labs.github.io/screamer/): pipelines, operator lifetime, and the API reference.
