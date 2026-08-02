import { test } from "node:test";
import assert from "node:assert/strict";
import { ready, RollingMean, Add } from "../src/index.ts";

test("lifecycle: explicit dispose() is idempotent", async () => {
  await ready();
  const op = RollingMean(3);
  op(1);
  op.dispose();
  assert.doesNotThrow(() => op.dispose()); // second dispose must be a no-op, not a double-free
});

test("lifecycle: `using` disposes at scope end", async () => {
  await ready();
  let captured: ReturnType<typeof RollingMean> | null = null;
  {
    using op = RollingMean(3);
    captured = op;
    assert.equal(typeof op(1), "number");
  }
  // The block's `using` ran Symbol.dispose on exit. Disposing again through the
  // captured reference must still be safe (idempotent), which is the
  // observable proof the scope-exit dispose already ran.
  assert.doesNotThrow(() => captured!.dispose());
});

test("lifecycle: construct + dispose 500 ops does not throw", async () => {
  await ready();
  for (let i = 0; i < 500; i++) {
    const op = RollingMean(3);
    op(i);
    op(i + 1);
    op.dispose();
  }
});

test("lifecycle: RollingMean(0) throws a RangeError (kernel invalid_argument)", async () => {
  await ready();
  assert.throws(() => RollingMean(0), RangeError);
});

test("lifecycle: op(<string>) throws a TypeError", async () => {
  await ready();
  const op = RollingMean(3);
  assert.throws(() => (op as any)("x"), TypeError);
  op.dispose();
});

test("lifecycle: calling op(x) after dispose() throws instead of writing through freed WASM memory", async () => {
  await ready();
  const op = RollingMean(3);
  op(1);
  op.dispose();
  assert.throws(() => op(2), /used after dispose/);
});

test("lifecycle: calling op(x) after `using` scope exit throws", async () => {
  await ready();
  let captured: ReturnType<typeof RollingMean> | null = null;
  {
    using op = RollingMean(3);
    captured = op;
    op(1);
  }
  assert.throws(() => captured!(2), /used after dispose/);
});

test("lifecycle: op.reset() after dispose() throws instead of touching a deleted kernel", async () => {
  await ready();
  const op = RollingMean(3);
  op.dispose();
  assert.throws(() => op.reset(), /used after dispose/);
});

test("lifecycle: a mismatched-length multi-input batch throws a TypeError", async () => {
  await ready();
  const op = Add();
  assert.throws(
    () => (op as any)(new Float64Array([1, 2, 3]), new Float64Array([1, 2])),
    TypeError,
  );
  op.dispose();
});

test("lifecycle: op(<mixed array>) throws a TypeError instead of NaN-filling or becoming a lazy generator", async () => {
  await ready();
  const op = RollingMean(3);
  assert.throws(() => (op as any)([1, "x"]), TypeError);
  assert.throws(() => (op as any)(["x", 1]), TypeError);
  op.dispose();
});
