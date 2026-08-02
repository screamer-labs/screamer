import { test } from "node:test";
import assert from "node:assert/strict";
import { init } from "../src/index.ts";
import { wrapOp } from "../src/runtime.ts";
import type { NdArray } from "../src/ndarray.ts";

// NaN-aware equality for a sequence of numbers.
function eqSeq(actual: ArrayLike<number>, expected: ArrayLike<number>) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < expected.length; i++) {
    const a = actual[i], e = expected[i];
    if (Number.isNaN(e)) assert.ok(Number.isNaN(a), `index ${i}: expected NaN, got ${a}`);
    else assert.equal(a, e, `index ${i}`);
  }
}

test("wrapOp: RollingMean(3, strict) across all regimes", async () => {
  const M = await init();
  const op = wrapOp(M, new M.RollingMean(3, "strict"));

  // N numbers -> one streaming event -> number
  const s = op(1);
  assert.equal(typeof s, "number");

  // single Float64Array -> Float64Array batch
  const fa = op(new Float64Array([1, 2, 3, 4, 5]));
  assert.ok(fa instanceof Float64Array);
  eqSeq(fa, [NaN, NaN, 2, 3, 4]);

  // single number[] -> number[] batch, container preserved
  const na = op([1, 2, 3]);
  assert.ok(Array.isArray(na));
  eqSeq(na, [NaN, NaN, 2]);

  // sync iterable -> generator streaming the same values
  const streamed = [...op([1, 2, 3, 4, 5][Symbol.iterator]())];
  eqSeq(streamed, [NaN, NaN, 2, 3, 4]);

  op.dispose();
});

test("wrapOp: RollingMinMax(3) multi-out batch returns NdArray", async () => {
  const M = await init();
  const op = wrapOp(M, new M.RollingMinMax(3));

  const res = op(new Float64Array([1, 2, 3, 4, 5])) as NdArray;
  assert.ok(res && res.data instanceof Float64Array);
  assert.deepEqual(res.shape, [5, 2]);
  assert.equal(res.data.length, 10);

  op.dispose();
});
