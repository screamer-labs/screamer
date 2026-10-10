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
