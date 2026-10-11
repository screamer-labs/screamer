import { test } from "node:test";
import assert from "node:assert/strict";
import { ready, RollingMean, RollingCorr, Add, Sub, BollingerBands, BacktestPriceTarget, reshape } from "../src/index.ts";
import type { NdArray } from "../src/index.ts";

const T = 50, K = 3;
const X: NdArray = { data: Float64Array.from({ length: T * K }, (_, i) => Math.sin(i) * 10 + i / 7), shape: [T, K] };
const y = Float64Array.from({ length: T }, (_, t) => Math.cos(t) * 5 + t / 3);
const col = (a: NdArray, k: number, width = a.shape[1]) =>
  Float64Array.from({ length: a.shape[0] }, (_, t) => a.data[t * width + k]);

test("one-input op on a (T, K) NdArray runs each column", async () => {
  await ready();
  const out = RollingMean(5)(X) as NdArray;
  assert.deepEqual(out.shape, [T, K]);
  for (let k = 0; k < K; k++) assert.deepEqual(col(out, k), RollingMean(5)(col(X, k)));
});

test("a (T,) input broadcasts against (T, K)", async () => {
  await ready();
  const out = RollingCorr(10)(X, y) as NdArray;
  assert.deepEqual(out.shape, [T, K]);
  for (let k = 0; k < K; k++) assert.deepEqual(col(out, k), RollingCorr(10)(col(X, k), y));
});

test("a multi-output engine broadcasts one price", async () => {
  await ready();
  const target: NdArray = { data: Float64Array.from({ length: T * K }, (_, i) => (i % 5) - 2), shape: [T, K] };
  const out = BacktestPriceTarget()(target, y) as NdArray;
  assert.deepEqual(out.shape, [T, K, 4]);
});

test("a packed (T, K, 2) array is split along its last axis", async () => {
  await ready();
  const packed: NdArray = { data: new Float64Array(T * K * 2).map((_, i) => (i % 2 ? 3 : 1)), shape: [T, K, 2] };
  const out = Sub()(packed) as NdArray;
  assert.deepEqual(out.shape, [T, K]);
  assert.ok(out.data.every((v) => v === -2));
});

test("a size-1 field axis is consumed", async () => {
  await ready();
  const out = BollingerBands(5)({ data: X.data, shape: [T, K, 1] }) as NdArray;
  assert.deepEqual(out.shape, [T, K, 3]);
});

test("mismatched assets raise", async () => {
  await ready();
  assert.throws(() => Add()({ data: new Float64Array(T * 3), shape: [T, 3] }, { data: new Float64Array(T * 4), shape: [T, 4] }), /do not broadcast/);
});

test("an iterator of K-value events equals the (T, K) batch", async () => {
  await ready();
  const rows = Array.from({ length: T }, (_, t) => X.data.subarray(t * K, (t + 1) * K));
  const got = [...(RollingMean(5)(rows[Symbol.iterator]()) as Iterable<Float64Array>)];
  const want = RollingMean(5)(X) as NdArray;
  got.forEach((r, t) => assert.deepEqual(r, want.data.subarray(t * K, (t + 1) * K)));
});

test("an event shape change raises", async () => {
  await ready();
  const it = (RollingMean(3)([new Float64Array(3), new Float64Array(4)][Symbol.iterator]()) as Iterator<any>);
  it.next();
  assert.throws(() => it.next(), /same shape/);
});

test("reshape", async () => {
  await ready();
  assert.deepEqual(reshape(X, "TA").shape, [T, K, 1]);
  assert.equal(reshape(X, "TA").data, X.data);                 // same buffer
  const t = reshape({ data: Float64Array.from([1, 2, 3, 4, 5, 6]), shape: [2, 3] }, "AT->TA");
  assert.deepEqual(t.shape, [3, 2]);
  assert.deepEqual(Array.from(t.data), [1, 4, 2, 5, 3, 6]);
  assert.throws(() => reshape({ data: new Float64Array(8), shape: [2, 4] }, "TF->T"), /only a size-1 axis/);
});
