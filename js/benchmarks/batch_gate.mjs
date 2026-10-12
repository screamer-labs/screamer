// Batch speed gate: ns per element of a whole-array call, best of 5 runs.
//
//   npm run build:wasm && npm run build && npm run bench:gate
//
// Prints one JSON object, {op: ns_per_element}. Compare against the Python
// binding on the same machine with the same ops and sizes, for example
//
//   python -c "import numpy as np, screamer as s, timeit; x = np.random.rand(1_000_000); \
//              op = s.RollingMean(20); print(min(timeit.repeat(lambda: op(x), number=1, repeat=5)) * 1e3)"
//
// which prints ns per element directly for 1M elements. The gate is each JS op
// within 1.5x of Python.

import { ready, RollingMean, RollingCorr, Add } from "../dist/index.js";

await ready();

const N = 1_000_000;
const x = new Float64Array(N), y = new Float64Array(N);
for (let i = 0; i < N; i++) {
  x[i] = 100 + Math.sin(i / 1000) + (i % 13) * 0.01;
  y[i] = 50 + Math.cos(i / 700) + (i % 7) * 0.02;
}

const best = (fn, runs = 5) => {
  fn();                                           // warm up
  let min = Infinity;
  for (let r = 0; r < runs; r++) {
    const t0 = process.hrtime.bigint();
    fn();
    min = Math.min(min, Number(process.hrtime.bigint() - t0));
  }
  return +(min / N).toFixed(2);
};

const cases = {
  "RollingMean(20)": [RollingMean(20), [x]],
  "RollingCorr(20)": [RollingCorr(20), [x, y]],
  "Add()": [Add(), [x, y]],
};

const out = {};
for (const [name, [op, args]] of Object.entries(cases)) {
  out[name] = best(() => op(...args));
  op.dispose();
}
console.log(JSON.stringify(out));
