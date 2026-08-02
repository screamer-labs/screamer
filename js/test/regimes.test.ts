import { test } from "node:test";
import assert from "node:assert/strict";
import { ready, RollingMean, RollingMinMax, Add, toNested } from "../src/index.ts";

// Same NaN-aware sequence comparison used by runtime.test.ts, applied here
// against the *ergonomic* factories (RollingMean/RollingMinMax from the public
// package surface), not wrapOp directly.
function eqSeq(actual: ArrayLike<number>, expected: ArrayLike<number>) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < expected.length; i++) {
    const a = actual[i], e = expected[i];
    if (Number.isNaN(e)) assert.ok(Number.isNaN(a), `index ${i}: expected NaN, got ${a}`);
    else assert.equal(a, e, `index ${i}`);
  }
}

async function* asyncGenOf(values: number[]) {
  for (const v of values) yield v;
}
function* syncGenOf(values: number[]) {
  for (const v of values) yield v;
}

test("regimes: RollingMean(3) across scalar/typed/array/sync-gen/async-gen", async () => {
  await ready();
  const values = [1, 2, 3, 4, 5];
  const expected = [NaN, NaN, 2, 3, 4];

  // Scalar event -> number.
  const scalarOp = RollingMean(3);
  const s = scalarOp(2);
  assert.equal(typeof s, "number");
  scalarOp.dispose();

  // Float64Array batch -> Float64Array, container-preserving.
  const typedOp = RollingMean(3);
  const fa = typedOp(new Float64Array(values));
  assert.ok(fa instanceof Float64Array);
  eqSeq(fa, expected);
  typedOp.dispose();

  // number[] batch -> number[], container-preserving.
  const arrOp = RollingMean(3);
  const na = arrOp(values);
  assert.ok(Array.isArray(na));
  eqSeq(na, expected);
  arrOp.dispose();

  // Sync generator -> generator yielding the same series as the batch.
  const syncOp = RollingMean(3);
  const streamed = [...syncOp(syncGenOf(values))];
  eqSeq(streamed, expected);
  syncOp.dispose();

  // Async generator -> async iterable yielding the same series.
  const asyncOp = RollingMean(3);
  const out: number[] = [];
  for await (const y of asyncOp(asyncGenOf(values)) as AsyncIterable<number>) out.push(y);
  eqSeq(out, expected);
  asyncOp.dispose();
});

test("regimes: RollingMinMax(3) multi-out batch is {data, shape} and toNested() nests it", async () => {
  await ready();
  const op = RollingMinMax(3);
  const res = op(new Float64Array([1, 2, 3, 4, 5])) as { data: Float64Array; shape: number[] };
  assert.ok(res.data instanceof Float64Array);
  assert.deepEqual(res.shape, [5, 2]);

  const nested = toNested(res) as number[][];
  assert.equal(nested.length, 5);
  for (const row of nested) assert.equal(row.length, 2);
  // window fills at index 2: min/max of [1,2,3] = [1,3]
  eqSeq(nested[2], [1, 3]);
  eqSeq(nested[4], [3, 5]);
  op.dispose();
});

test("regimes: Add() 2-input columnar batch computes elementwise sums", async () => {
  await ready();
  const op = Add();
  const out = op(new Float64Array([1, 2, 3]), new Float64Array([10, 20, 30])) as Float64Array;
  assert.ok(out instanceof Float64Array);
  eqSeq(out, [11, 22, 33]);
  op.dispose();
});

test("regimes: Add() rejects a mismatched-length columnar batch with a TypeError instead of NaN-filling/truncating", async () => {
  await ready();
  const op = Add();
  assert.throws(
    () => op(new Float64Array([1, 2, 3]), new Float64Array([10, 20])),
    TypeError,
  );
  assert.throws(
    () => op([1, 2], [1, 2, 3]),
    TypeError,
  );
  op.dispose();
});

test("regimes: op([]) is still a valid empty batch (not treated as a mixed/invalid array)", async () => {
  await ready();
  const op = RollingMean(3);
  const out = op([]) as number[];
  assert.ok(Array.isArray(out));
  assert.equal(out.length, 0);
  op.dispose();
});
