import { test } from "node:test";
import assert from "node:assert/strict";
import { ready, RollingMean, MovingAverage, EwMean } from "../src/index.ts";

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
