import { test } from "node:test";
import assert from "node:assert/strict";
import { ready, KalmanFilter, KalmanRegression, RollingTLSBeta, RollingADF, BandSignal } from "../src/index.ts";

await ready();

// Same inputs and values as the Python reference run.
const t = Array.from({ length: 40 }, (_, i) => i);
const x = t.map((i) => 4 + 0.01 * Math.sin(0.3 * i) + 0.002 * i);
const y = t.map((i, j) => 0.3 + 1.2 * x[j] + 0.01 * Math.cos(0.7 * i));
const near = (a: ArrayLike<number>, b: number[], tol = 1e-9) =>
  a.length === b.length && Array.from(a).every((v, i) => Math.abs(v - b[i]) <= tol * Math.max(1, Math.abs(b[i])));
const pick = (a: ArrayLike<number>, idx: number[]) => idx.map((i) => a[i]);

test("pairs: KalmanRegression matches Python", () => {
  const out = KalmanRegression(1e-6, 1e-4)(Float64Array.from(y), x) as { data: Float64Array; shape: number[] };
  assert.deepEqual(out.shape, [40, 4]);
  const rows = [1, 10, 39].flatMap((r) => Array.from(out.data.slice(r * 4, r * 4 + 4)));
  assert.ok(near(rows, [1.1989424874592927, 0.3129504542698542, -0.002333143763240564, 0.014789076087763668,
    1.13479217200963, 0.5674108946512093, 0.0035554125186028074, 0.012285493136522389,
    1.150177855392803, 0.5031693172594055, -0.009171341017913726, 0.012372309687227039], 1e-7), `${rows}`);
});

test("pairs: robust (huber) Kalman updates match Python", () => {
  // One bad print at step 20. The inputs come from Math.sin and Math.cos, whose last bits
  // can differ from numpy's, so the comparison has the same tolerance as the test above.
  const yo = y.slice();
  yo[20] += 0.2;
  const out = KalmanRegression({ delta: 1e-6, obsVar: 1e-4, huber: 2.0 })(Float64Array.from(yo), x) as { data: Float64Array; shape: number[] };
  const rows = [19, 20, 39].flatMap((r) => Array.from(out.data.slice(r * 4, r * 4 + 4)));
  assert.ok(near(rows, [1.1939134192501708, 0.32968647138988627, 0.0035329616307819833, 0.012434129432032209,
    1.3128913565338007, -0.13751464027507937, 0.1962536801715764, 0.01250234817984822,
    1.1405075862250522, 0.5424941493512908, -0.009114971339498013, 0.012372324860883415], 1e-7), `${rows}`);
  const z = t.map((i) => Math.sin(0.4 * i) * 3 + 0.1 * i);
  z[15] += 50;
  assert.ok(near(pick(KalmanFilter(0.1, 1.0, 0.0, 1.0, 2.5)(z), [14, 15, 16, 39]),
    [-0.6852343002984592, 0.3756760957352624, 0.8734531901016516, 4.993678476150991], 1e-12));
  assert.deepEqual(KalmanFilter({ huber: Infinity })(z), KalmanFilter()(z));
  // After the first observation, an infinite one is skipped.
  assert.deepEqual(KalmanFilter(0.1, 1.0, 2.0, 0.1, 2.0)([2, Infinity, -Infinity]), [2, 2, 2]);
  assert.throws(() => KalmanFilter({ huber: 0 }), /huber must be > 0/);
  assert.throws(() => KalmanRegression({ huber: NaN }), /huber must be finite or Infinity, got NaN/);
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
