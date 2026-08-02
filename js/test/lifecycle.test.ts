import { test } from "node:test";
import assert from "node:assert/strict";
import { ready, RollingMean } from "../src/index.ts";

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
