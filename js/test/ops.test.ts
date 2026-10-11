import { test } from "node:test";
import assert from "node:assert/strict";
import { ready, RollingMean, MovingAverage, EwMean, BacktestPortfolioTarget } from "../src/index.ts";

test("generated factory: ready() + RollingMean(3) scalar stream", async () => {
  await ready();
  const op = RollingMean(3);
  assert.ok(Number.isNaN(op(1))); // strict: NaN until the window fills
  assert.ok(Number.isNaN(op(2)));
  assert.equal(op(3), 2); // mean of [1,2,3]
  op.dispose();
});

test("generated factory: optional args default to NaN sentinel (EwMean)", async () => {
  await ready();
  // span provided, others left as the missing-optional NaN default.
  const op = EwMean(NaN, 3);
  assert.equal(typeof op(1), "number");
  op.dispose();
});

test("generated factory: vector<double> arg (MovingAverage taps)", async () => {
  await ready();
  const op = MovingAverage([0.5, 0.5]);
  assert.equal(typeof op(1), "number");
  op.dispose();
});

test("G3: mutating a parameter array after construction changes nothing", async () => {
  await ready();
  // A stream runs on a copy built after the array changed.
  const x = Array.from({ length: 50 }, (_, i) => Math.sin(i));
  const taps = [0.25, 0.5, 0.25];
  const ma = MovingAverage(taps);
  taps[0] = 9;
  const ref = MovingAverage([0.25, 0.5, 0.25]);
  assert.deepEqual([...ma(x.values())], [...ref(x.values())]);
});

test("G3: per-asset vector parameters are copied too", async () => {
  await ready();
  const T = 30, K = 2;
  const rows = Array.from({ length: T }, (_, t) => ({ w: new Float64Array(K).fill(0.5), px: Float64Array.from({ length: K }, (_, k) => 100 + t * K + k) }));
  const fee = [0.001, 0.002];
  const eng = BacktestPortfolioTarget(fee);
  fee[0] = 0.5;
  const ref = BacktestPortfolioTarget([0.001, 0.002]);
  const run = (e: any) => [...e(rows.map((r) => [r.w, r.px]).values())].map((r: any) => Array.from(r.data ?? r));
  assert.deepEqual(run(eng), run(ref));
  assert.notDeepEqual(run(eng), run(BacktestPortfolioTarget([0.5, 0.002])), "the fee must matter here");
});
