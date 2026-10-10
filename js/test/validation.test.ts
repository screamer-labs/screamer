import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ready, circularShift, segmentShuffle, positionTest,
  BacktestOHLCTarget, BacktestReport,
} from "../src/index.ts";

await ready();

const NaN_ = Number.NaN;
const X = [NaN_, NaN_, 1, 1, 1, 0, 0, -1, -1, -1, -1, 2, 0, 0, 1];

function col(a: { data: Float64Array; shape: number[] }, j: number): number[] {
  const [rows, cols] = a.shape;
  const out: number[] = [];
  for (let r = 0; r < rows; r++) out.push(a.data[r * cols + j]);
  return out;
}

// Same golden values as tests/test_validation.py::test_golden_values: the
// seeded surrogates must be identical in Python and JS/WASM.
test("validation: golden surrogates match Python", () => {
  assert.deepEqual(col(circularShift(X, 2, { minShift: 2, seed: 42 }), 0),
    [NaN_, NaN_, 2, 0, 0, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1]);
  assert.deepEqual(col(segmentShuffle(X, 2, { seed: 42 }), 1),
    [NaN_, NaN_, 0, 0, 1, 1, 1, -1, -1, -1, -1, 2, 0, 0, 1]);
});

test("validation: argument checks raise RangeError", () => {
  assert.throws(() => circularShift(X, 3, { minShift: 0 }), RangeError);
  assert.throws(() => circularShift(X, 3, { minShift: 7 }), RangeError);
  assert.throws(() => circularShift([NaN_, 1], 3), RangeError);
  assert.throws(() => segmentShuffle(X, 3, { seed: -1 }), RangeError);
});

// A deterministic market and a target that holds each side for 5 bars.
function market(T: number) {
  const o: number[] = [], h: number[] = [], l: number[] = [], c: number[] = [];
  let p = 100;
  for (let t = 0; t < T; t++) {
    const prev = p;
    p *= Math.exp(0.01 * Math.sin(1.7 * t) * Math.cos(0.31 * t * t));
    o.push(prev); c.push(p); h.push(Math.max(prev, p) * 1.001); l.push(Math.min(prev, p) * 0.999);
  }
  return { o, h, l, c };
}

test("validation: positionTest observed metric matches BacktestReport", () => {
  const { o, h, l, c } = market(300);
  const tgt = Array.from({ length: 300 }, (_, t) => (Math.floor(t / 5) % 3) - 1);
  const eng = BacktestOHLCTarget(2e-4);
  const res = positionTest(eng, tgt, [o, h, l, c], { n: 50, seed: 3 });
  const out = eng(tgt, o, h, l, c) as { data: Float64Array; shape: number[] };
  const rep = BacktestReport()(col(out, 0), col(out, 1), col(out, 2), col(out, 3)) as
    { data: Float64Array; shape: number[] };
  const sharpe = col(rep, 5).filter(Number.isFinite).at(-1);
  assert.equal(res.observed, sharpe);
  assert.equal(res.null.length, 50);
  assert.ok(res.pValue >= 1 / 51 && res.pValue <= 1);
  assert.ok((res.minShift as number) >= 1);
  const again = positionTest(eng, tgt, [o, h, l, c], { n: 50, seed: 3 });
  assert.deepEqual(Array.from(again.null), Array.from(res.null));
});

test("validation: positionTest detects a planted edge", () => {
  const { o, h, l, c } = market(800);
  // Knows the next bar's direction: the engine executes the bar-t target on bar t+1.
  const tgt = c.map((_, t) => (t + 1 < c.length ? Math.sign(c[t + 1] - o[t + 1]) : 0));
  const res = positionTest(BacktestOHLCTarget(), tgt, [o, h, l, c],
    { n: 200, metric: "total_pnl", method: "segments" });
  assert.ok(res.pValue < 0.01, `p = ${res.pValue}`);
});

// Same golden values as tests/test_validation.py::test_permutation_golden_values.
test("validation: golden permutations match Python", async () => {
  const { permuteBars, permuteReturns } = await import("../src/index.ts");
  const o = [100, 101, 99, 102, 103, 101];
  const h = o.map((v) => v * 1.01), l = o.map((v) => v * 0.98);
  const c = [100.5, 99.5, 101, 102.5, 101.5, 102];
  const close = (a: number[], b: number[]) =>
    a.every((v, i) => (Number.isNaN(v) && Number.isNaN(b[i])) || Math.abs(v - b[i]) <= 1e-12 * Math.abs(b[i]));
  const bars = permuteBars(o, h, l, c, 2, { seed: 7 });
  assert.ok(close(col(bars[0], 1), [100.0, 101.00000000000003, 98.99999999999999,
    100.50246305418726, 101.99507389162568, 103.50738916256157]));
  assert.ok(close(col(bars[3], 1), [100.5, 99.49999999999997, 101.00000000000003,
    100.99512218680589, 103.00492610837439, 101.99999999999996]));
  const r = permuteReturns([NaN_, 10, 11, 10.5, 12, 11.5, 13], 2, { seed: 7 });
  assert.ok(close(col(r, 0), [NaN_, 10.0, 11.30434782608696, 10.833333333333336,
    12.380952380952385, 11.81818181818182, 13.0]));
});

test("validation: strategyTest separates an informative feature from noise", async () => {
  const { strategyTest, BacktestPriceTarget } = await import("../src/index.ts");
  // AR(1) returns with a deterministic pseudo-random driver.
  let seed = 12345;
  const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 - 0.5; };
  const T = 1200, price: number[] = [], feat: number[] = [], noise: number[] = [];
  let r = 0, p = 100, ew = 0;
  for (let t = 0; t < T; t++) {
    r = 0.3 * r + 0.02 * rnd();
    p *= Math.exp(r);
    price.push(p);
    ew = ew + (2 / 11) * (r - ew);
    feat.push(ew);
    noise.push(rnd());
  }
  const strat = (f: Float64Array, px: Float64Array) =>
    BacktestPriceTarget()(Array.from(f, Math.sign), px);
  const good = strategyTest(strat, [feat, price], { n: 200, method: "shift", columns: [0] });
  const bad = strategyTest(strat, [noise, price], { n: 200, method: "shift", columns: [0] });
  assert.ok(good.pValue < 0.05, `informative p = ${good.pValue}`);
  assert.ok(bad.pValue > 0.05, `noise p = ${bad.pValue}`);
  const bars = strategyTest((px: Float64Array) =>
      BacktestPriceTarget()(Array.from(px, () => 1), px), [price],
    { n: 20, method: "returns" });
  assert.equal(bars.null.length, 20);
});
