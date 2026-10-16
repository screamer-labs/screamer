import { test } from "node:test";
import assert from "node:assert/strict";
import { ready, TripleBarrier, TripleBarrierOHLC } from "../src/index.ts";

await ready();

// Same inputs and values as the Python reference run.
const n = 40;
const price = Array.from({ length: n }, (_, i) => 100 * Math.exp(0.01 * Math.sin(0.7 * i) + 0.002 * i));
type Out = { data: Float64Array; shape: number[] };
const near = (a: ArrayLike<number>, b: number[], tol = 1e-9) =>
  a.length === b.length && Array.from(a).every((v, i) => Math.abs(v - b[i]) <= tol * Math.max(1, Math.abs(b[i])));
const rows = (o: Out, idx: number[]) => idx.flatMap((r) => Array.from(o.data.slice(r * 3, r * 3 + 3)));
const expected = [1, 0.013950915608770709, 2, 1, 0.015248274571560971, 3, -1, -0.010820569608274822, 4];

test("labels: TripleBarrier fraction matches Python", () => {
  const out = TripleBarrier({ horizon: 5, upper: 0.01, lower: 0.01 })(Float64Array.from(price)) as Out;
  assert.deepEqual(out.shape, [n, 3]);
  assert.ok(near(rows(out, [5, 20, 34]), expected), `${rows(out, [5, 20, 34])}`);
});

test("labels: TripleBarrier price unit matches Python", () => {
  // upper 0.5 and lower 2.0 in price units: exits differ from the fraction case
  const out = TripleBarrier({ horizon: 5, upper: 0.5, lower: 2, barrierUnit: "price" })(Float64Array.from(price)) as Out;
  assert.ok(near(rows(out, [10, 20, 34]), [1, 0.011744401504859336, 4, 1, 0.006637517805758275, 2,
                                           0, -0.009599278457675275, 5]), `${rows(out, [10, 20, 34])}`);
});

test("labels: TripleBarrier vol unit matches Python", () => {
  // upper 2 and lower 0.5 expected deviations: exits differ from the other cases
  const out = TripleBarrier({ horizon: 5, upper: 2, lower: 0.5, barrierUnit: "vol", volSpan: 10 })(Float64Array.from(price)) as Out;
  assert.ok(rows(out, [5]).every(Number.isNaN));
  assert.ok(near(rows(out, [10, 20, 34]), [0, 0.020280610979460212, 5, 1, 0.02377078134957844, 4,
                                           -1, -0.007872072550773934, 3]), `${rows(out, [10, 20, 34])}`);
});

test("labels: TripleBarrierOHLC matches Python", () => {
  const o = price.map((p) => p * 1.0005);
  const h = price.map((p, i) => Math.max(o[i], p) * 1.004);
  const l = price.map((p, i) => Math.min(o[i], p) * 0.996);
  const out = TripleBarrierOHLC({ horizon: 5, upper: 0.01, lower: 0.01 })(Float64Array.from(o), Float64Array.from(h), Float64Array.from(l), Float64Array.from(price)) as Out;
  assert.deepEqual(out.shape, [n, 3]);
  assert.ok(near(rows(out, [5, 20, 34]), [1, 0.010000000000000009, 1, 1, 0.010000000000000009, 2, -1, -0.010000000000000009, 3]));
});

test("labels: unknown barrierUnit throws", () => {
  assert.throws(() => TripleBarrier({ horizon: 5, upper: 0.01, lower: 0.01, barrierUnit: "ticks" }));
});
