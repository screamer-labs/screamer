# screamer.js benchmarks

Reproducible throughput numbers for the JavaScript/WebAssembly build.

```bash
cd js
npm ci
npm run build      # build the WASM module + dist (once)
npm run bench      # node benchmarks/throughput.mjs
```

`throughput.mjs` measures the same operators in two regimes:

- **Batch** (`op(array)`): the whole array runs in one C++ pass over WebAssembly
  (`evalBatchInto`), without recrossing the JS/WASM boundary per element. This is
  the "C++ speed" number.
- **Streaming** (`op(x)`): one value at a time, which crosses the JS/WASM boundary
  once per event, as a live event-driven feed does.

It self-checks that the batch result is identical to the streaming result before
reporting speed, so the batch path is only ever a speed-up, never a different
answer.

## Representative run

Single machine, single process (node 24, Apple Silicon), 10M samples. Your
numbers will vary with hardware; the point is the order of magnitude.

```
batch == streaming: OK (identical results)

BATCH  op(array), one C++ pass
  EwMean(span=20)       183.0 M/s   (54.6 ms / 10M)
  RollingMean(20)       165.7 M/s   (60.4 ms / 10M)
  RollingPoly1(30)      139.3 M/s   (71.8 ms / 10M)
  RollingRSI(14)        144.3 M/s   (69.3 ms / 10M)

STREAMING  op(x) one value at a time
  EwMean(span=20)         2.2 M/s   (1377.0 ms / 3M)
  RollingMean(20)         2.2 M/s   (1385.0 ms / 3M)
```

So an EWMA runs at roughly **180 million updates per second** on batch arrays,
and a couple million per second one event at a time. A browser (Chrome, same V8
plus the same WebAssembly module) measures in the same range.
