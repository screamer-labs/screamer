// screamer.js throughput benchmark.
//
//   npm run build        # build the WASM module + dist (once)
//   npm run bench        # or: node benchmarks/throughput.mjs
//
// Reports two regimes over the same operators:
//   - BATCH: op(array) runs the whole array in one C++ pass (evalBatchInto),
//     the "C++ speed" number.
//   - STREAMING: op(x) one value at a time, which crosses the JS/WASM boundary
//     per event (what a live, event-driven feed does).
//
// Single-machine, single-process numbers. Run it on your own hardware; the
// point is the order of magnitude, not a leaderboard.

import { ready, EwMean, RollingMean, RollingPoly1, RollingRSI } from "../dist/index.js";

await ready();

const seconds = (fn, runs) => {
  for (let i = 0; i < 2; i++) fn();              // warm up the JIT
  const t0 = process.hrtime.bigint();
  for (let r = 0; r < runs; r++) fn();
  return Number(process.hrtime.bigint() - t0) / 1e9 / runs;
};
const rate = (perSec) =>
  (perSec >= 1e6 ? (perSec / 1e6).toFixed(1) + " M/s" : (perSec / 1e3).toFixed(1) + " k/s").padStart(10);

const N = 10_000_000;
const data = new Float64Array(N);
for (let i = 0; i < N; i++) data[i] = 100 + Math.sin(i / 1000) + (i % 13) * 0.01;

const OPS = [
  ["EwMean(span=20)",   () => EwMean(undefined, 20)],
  ["RollingMean(20)",   () => RollingMean(20)],
  ["RollingPoly1(30)",  () => RollingPoly1(30, 0)],
  ["RollingRSI(14)",    () => RollingRSI(14)],
];

console.log(`screamer.js throughput  |  node ${process.version}  |  N = ${N / 1e6}M samples\n`);

// Self-check: op(array) must equal one-at-a-time streaming (NaN-aware). The
// batch path is only a speed-up, never a different result.
{
  const check = data.subarray(0, 50_000);
  let allEqual = true;
  for (const [, make] of OPS) {
    const b = make()(check);
    const s = make();
    for (let i = 0; i < check.length; i++) {
      const x = b[i], y = s(check[i]);
      if (Number.isNaN(x) && Number.isNaN(y)) continue;
      if (Math.abs(x - y) > 1e-9 * Math.max(1, Math.abs(y))) { allEqual = false; break; }
    }
  }
  console.log(`batch == streaming: ${allEqual ? "OK (identical results)" : "MISMATCH"}\n`);
}

console.log("BATCH  op(array), one C++ pass");
for (const [name, make] of OPS) {
  const sec = seconds(() => { const op = make(); op(data); op.dispose(); }, 5);
  console.log(`  ${name.padEnd(20)} ${rate(N / sec)}   (${(sec * 1e3).toFixed(1)} ms / ${N / 1e6}M)`);
}

const Ns = 3_000_000;
console.log("\nSTREAMING  op(x) one value at a time");
for (const [name, make] of OPS.slice(0, 2)) {
  const op = make();
  const sec = seconds(() => { for (let i = 0; i < Ns; i++) op(data[i]); }, 3);
  op.dispose();
  console.log(`  ${name.padEnd(20)} ${rate(Ns / sec)}   (${(sec * 1e3).toFixed(1)} ms / ${Ns / 1e6}M)`);
}
