---
title: Node, bundlers, and the browser
---

# Node, bundlers, and the browser

screamer.js is one package that runs in Node and in the browser. The API is identical everywhere;
only how you point the runtime at the package differs. In every environment the rule is the same:
call `await ready()` once before constructing any operator, because the WebAssembly module loads
asynchronously.

Browser use (a bundler or a CDN) requires version **2.2.1 or newer**. Earlier versions load only
under Node.

## Node

Node 18 or newer. Install and import by name:

```js
// npm i @screamer-labs/screamer
import { ready, RollingMean } from "@screamer-labs/screamer";

await ready();
const sma = RollingMean(3);
console.log(sma([1, 2, 3, 4])); // [ NaN, NaN, 2, 3 ]
```

The package ships one self-contained module with the WebAssembly embedded, so there is no separate
`.wasm` file to resolve.

## Browser with a bundler

Vite, Next.js, webpack, Rollup, esbuild, and similar tools need no special configuration. Install
and import exactly as in Node; the bundler inlines the embedded-WASM module into your output:

```js
// npm i @screamer-labs/screamer
import { ready, RollingMean } from "@screamer-labs/screamer";

await ready();
const sma = RollingMean(3);
```

The bare specifier `@screamer-labs/screamer` resolves through the bundler (or an import map). A raw
browser cannot resolve a bare specifier, which is what the CDN form below handles.

## Browser without a build step (CDN)

Import from a full URL inside a module script. jsDelivr serves the published module directly:

```html
<script type="module">
  import { ready, RollingMean }
    from "https://cdn.jsdelivr.net/npm/@screamer-labs/screamer@2.2.1/dist/index.js";

  await ready();

  const sma = RollingMean(3);
  console.log(Array.from(sma(new Float64Array([1, 2, 3, 4])))); // [ NaN, NaN, 2, 3 ]
</script>
```

Pin the version (`@2.2.1`) so loads are reproducible. The module embeds its WebAssembly, so nothing
else needs to be fetched or configured.

To avoid a third-party origin (for offline use, a strict Content-Security-Policy, or full control),
copy the package's `dist/` directory next to your page and import it with a relative URL. The
[Live demo](./live-demo.md) does this; its
[source](https://github.com/screamer-labs/screamer/blob/main/js/examples/live-trades.html) shows
the pattern.

## The rule in every environment

`await ready()` resolves the WebAssembly module once and caches it; a second call awaits the same
load. Construct operators after it, never before. Each op owns memory on the WASM side, so call
`.dispose()` when you are done with it (or use `using` where your runtime supports it). See
[Lifecycle](./lifecycle.md).

Requirements: Node 18+, or any browser with WebAssembly, which is every current browser on desktop
and mobile.
