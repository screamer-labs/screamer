import { test } from "node:test";
import assert from "node:assert/strict";
import { ready, Input, Pipeline, Diff, RollingMean } from "../src/index.ts";

// NaN-aware element equality between two same-length numeric sequences.
function assertNanEq(got: ArrayLike<number>, expected: ArrayLike<number>, tol = 1e-9) {
  assert.equal(got.length, expected.length, `length ${got.length} != ${expected.length}`);
  for (let i = 0; i < expected.length; i++) {
    const e = expected[i], g = got[i];
    if (Number.isNaN(e)) assert.ok(Number.isNaN(g), `index ${i}: expected NaN, got ${g}`);
    else assert.ok(Math.abs(g - e) <= tol, `index ${i}: ${g} != ${e}`);
  }
}

test("pipeline: Diff(1)(RollingMean(3)(x)) matches the point-op chain", async () => {
  await ready();
  const data = new Float64Array([1, 2, 3, 4, 5, 6]);

  const x = Input("x");
  const y = Diff(1)(RollingMean(3)(x));
  const p = new Pipeline([x], [y]);
  const { values } = p(data) as { values: Float64Array; index: Float64Array };

  // Reference: the same computation done eagerly with fresh point-op instances.
  const rm = RollingMean(3)(data) as Float64Array;
  const expected = Diff(1)(rm) as Float64Array;

  assertNanEq(values, expected);

  const { index } = p(data) as { values: Float64Array; index: Float64Array };
  assertNanEq(index, [0, 1, 2, 3, 4, 5]);

  p.dispose();
});

test("pipeline: a 2-op chain (RollingSum then Diff) matches the point-op path", async () => {
  await ready();
  const { RollingSum } = await import("../src/index.ts");
  const data = new Float64Array([2, 4, 6, 8, 10]);

  const x = Input("x");
  const y = Diff(1)(RollingSum(2)(x));
  const p = new Pipeline([x], [y]);
  const { values } = p(data) as { values: Float64Array };

  const rs = RollingSum(2)(data) as Float64Array;
  const expected = Diff(1)(rs) as Float64Array;
  assertNanEq(values, expected);

  p.dispose();
});

test("pipeline: dispose() twice does not throw", async () => {
  await ready();
  const x = Input("x");
  const y = RollingMean(3)(x);
  const p = new Pipeline([x], [y]);
  p(new Float64Array([1, 2, 3]));
  p.dispose();
  assert.doesNotThrow(() => p.dispose());
});
