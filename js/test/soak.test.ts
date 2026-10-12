import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ready, current, Input, Pipeline,
  RollingMean, BollingerBands, BacktestPriceTarget, PortfolioReport, CrossRank,
} from "../src/index.ts";

// Soak tests: a program that builds a new op per call must run for months, so
// building and dropping ops must leave nothing behind on either heap once the
// garbage collector has run. `npm test` runs node with --expose-gc.
//
// Each case builds and drops N ops in chunks, giving the event loop a turn
// after each chunk so the finalizers run, as they would in a live loop. It
// measures the JS heap and the bytes the WASM allocator has in use after a
// warm-up round and after the measured round.

const gc = (globalThis as any).gc as (() => void) | undefined;
const N = 100_000;
const CHUNK = 10_000;
const JS_BOUND = 1 << 20;       // 1 MB in total, about 10 bytes per op
const WASM_BOUND = 4096;        // the WASM heap must come back to where it was

async function settle(): Promise<void> {
  // A full collection queues the finalizers; they run on a later task, and
  // what they free can let the next collection free more.
  for (let i = 0; i < 6; i++) {
    gc!();
    await new Promise((r) => setTimeout(r, 0));
  }
}

async function round(make: () => void, n: number): Promise<void> {
  for (let done = 0; done < n; done += CHUNK) {
    for (let i = 0; i < CHUNK; i++) make();
    await new Promise((r) => setTimeout(r, 0));
  }
  await settle();
}

async function retained(make: () => void, n = N): Promise<{ js: number; wasm: number }> {
  assert.equal(typeof gc, "function", "run with node --expose-gc (npm test does)");
  await ready();
  const M = current();
  await round(make, CHUNK);                       // warm-up: code, caches, pools
  const js0 = process.memoryUsage().heapUsed, w0 = M.heapInUse();
  await round(make, n);
  return { js: process.memoryUsage().heapUsed - js0, wasm: M.heapInUse() - w0 };
}

const row = new Float64Array([1, 2, 3]);
const book = [[1, 1, 1, 0], [2, 1, 1, 0]];
const prices = new Float64Array([1, 2, 3]);

// The cases below that build several things per step run fewer steps.
const N_MIXED = 20_000;

// Each kind built and dropped, and built, called once, and dropped.
const kinds: Record<string, [() => void, () => void]> = {
  "RollingMean (one input)": [
    () => { RollingMean(3); },
    () => { RollingMean(3)(1); },
  ],
  "BollingerBands (several outputs)": [
    () => { BollingerBands(3); },
    () => { BollingerBands(3)(1); },
  ],
  "BacktestPriceTarget (engine)": [
    () => { BacktestPriceTarget(); },
    () => { BacktestPriceTarget()(100, 1); },
  ],
  "PortfolioReport (reducer)": [
    () => { PortfolioReport(); },
    () => { PortfolioReport()(book); },
  ],
  "CrossRank (row op)": [
    () => { CrossRank(); },
    () => { CrossRank()(row); },
  ],
  "Pipeline": [
    () => { const x = Input("x"); new Pipeline([x], [RollingMean(3)(x)]); },
    () => { const x = Input("x"); new Pipeline([x], [RollingMean(3)(x)])(prices); },
  ],
  "Pipeline (aligned outputs)": [
    () => { const x = Input("x"), y = Input("y"); new Pipeline([x, y], [RollingMean(3)(x), y]); },
    () => {
      const x = Input("x"), y = Input("y");
      const p = new Pipeline([x, y], [RollingMean(3)(x), y]);
      p({ x: { values: prices, index: [0, 2, 4] }, y: { values: prices, index: [1, 2, 3] } });
      const d = p.live();
      d.push("x", 0, 1); d.push("y", 1, 2); d.result(); d.push("x", 2, 3); d.flush(); d.result();
    },
  ],
};

for (const [name, [build, buildAndCall]] of Object.entries(kinds)) {
  test(`soak: ${name}, ${N} built and dropped, leaves nothing behind`, async () => {
    const r = await retained(build);
    assert.ok(r.js < JS_BOUND, `JS heap grew by ${r.js} bytes`);
    assert.ok(r.wasm < WASM_BOUND, `WASM heap grew by ${r.wasm} bytes`);
  });
  test(`soak: ${name}, ${N} built, called and dropped, leaves nothing behind`, async () => {
    const r = await retained(buildAndCall);
    assert.ok(r.js < JS_BOUND, `JS heap grew by ${r.js} bytes`);
    assert.ok(r.wasm < WASM_BOUND, `WASM heap grew by ${r.wasm} bytes`);
  });
}

test("soak: streams dropped part way, and live drivers, leave nothing behind", async () => {
  const r = await retained(() => {
    const it = RollingMean(3)([1, 2, 3].values());
    it.next();                                   // started, never finished
    const x = Input("x");
    const d = new Pipeline([x], [RollingMean(2)(x)]).live();
    d.push("x", 0, 1);
  }, N_MIXED);
  assert.ok(r.js < JS_BOUND, `JS heap grew by ${r.js} bytes`);
  assert.ok(r.wasm < WASM_BOUND, `WASM heap grew by ${r.wasm} bytes`);
});

test("soak: dispose() then collection frees once", async () => {
  const r = await retained(() => {
    const a = RollingMean(3); a(1); a.dispose();
    const b = PortfolioReport(); b(book); b.dispose(); b.dispose();
    const c = CrossRank(); c(row); c.dispose();
    const x = Input("x");
    const p = new Pipeline([x], [RollingMean(3)(x)]);
    p(prices); p.dispose(); p.dispose();
  }, N_MIXED);
  assert.ok(r.js < JS_BOUND, `JS heap grew by ${r.js} bytes`);
  assert.ok(r.wasm < WASM_BOUND, `WASM heap grew by ${r.wasm} bytes`);
});

// The finalizers free only what nothing uses: a Pipeline keeps its ops, a
// live driver its Pipeline, and a stream its op.
test("soak: what is in use survives collection", async () => {
  assert.equal(typeof gc, "function", "run with node --expose-gc (npm test does)");
  await ready();
  const x = Input("x");
  const p = new Pipeline([x], [RollingMean(2)(x)]);   // the op is held by p only
  const it = RollingMean(2)([1, 2, 3, 4].values());    // the op is held by it only
  const d = (() => {
    const y = Input("y");
    return new Pipeline([y], [RollingMean(2)(y)]).live();  // held by d only
  })();
  await settle();
  const out = p(new Float64Array([1, 2, 3])) as { values: Float64Array };
  assert.deepEqual(Array.from(out.values).slice(1), [1.5, 2.5]);
  assert.deepEqual([...it].slice(1), [1.5, 2.5, 3.5]);
  d.push("y", 0, 1); d.push("y", 1, 3); d.flush();
  assert.deepEqual(Array.from(d.result().values as Float64Array).slice(1), [2]);
});
