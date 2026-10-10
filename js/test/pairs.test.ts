import { test } from "node:test";
import assert from "node:assert/strict";
import { ready, KalmanRegression, RollingTLSBeta, RollingADF, BandSignal } from "../src/index.ts";

await ready();

// Same inputs and values as the Python reference run.
const t = Array.from({ length: 40 }, (_, i) => i);
const x = t.map((i) => 4 + 0.01 * Math.sin(0.3 * i) + 0.002 * i);
const y = t.map((i, j) => 0.3 + 1.2 * x[j] + 0.01 * Math.cos(0.7 * i));
const near = (a: ArrayLike<number>, b: number[], tol = 1e-9) =>
  a.length === b.length && Array.from(a).every((v, i) => Math.abs(v - b[i]) <= tol * Math.max(1, Math.abs(b[i])));
const pick = (a: ArrayLike<number>, idx: number[]) => idx.map((i) => a[i]);

test("pairs: KalmanRegression matches Python", () => {
  const out = KalmanRegression(1e-6, 1e-4)(y, x) as { data: Float64Array; shape: number[] };
  assert.deepEqual(out.shape, [40, 4]);
  const rows = [1, 10, 39].flatMap((r) => Array.from(out.data.slice(r * 4, r * 4 + 4)));
  assert.ok(near(rows, [1.1989424874592927, 0.3129504542698542, -0.002333143763240564, 0.014789076087763668,
    1.13479217200963, 0.5674108946512093, 0.0035554125186028074, 0.012285493136522389,
    1.150177855392803, 0.5031693172594055, -0.009171341017913726, 0.012372309687227039], 1e-7), `${rows}`);
});

test("pairs: RollingTLSBeta and RollingADF match Python", () => {
  assert.ok(near(pick(RollingTLSBeta(10)(y, x), [9, 25, 39]),
    [1.3821248703670799, 0.9361156325331089, 8.848719311542256], 1e-7));
  const series = t.map((i) => Math.sin(0.5 * i) + 0.05 * i);
  assert.ok(near(pick(RollingADF(20, 1)(series), [21, 30, 39]),
    [-9.193243019780736, -11.29542543504454, -9.270643718139713], 1e-7));
});

test("pairs: BandSignal matches Python", () => {
  const z = t.map((i) => Math.sin(0.4 * i) * (i + 1) * 0.12);
  assert.deepEqual(BandSignal(2.0, 0.0, 4.0)(z), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, -1, -1, -1, 0, 0, 1, 1, 1, 1, 1, 1, 0, -1, -1, 0, 0, 0, 0, 0]);
  assert.throws(() => BandSignal(2.0, 3.0));
});
