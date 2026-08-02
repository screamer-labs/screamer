import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ready,
  Input,
  Pipeline,
  Sub,
  combineLatest,
  resample,
} from "../src/index.ts";

// NaN-aware element equality between two same-length numeric sequences.
function assertNanEq(got: ArrayLike<number>, expected: ArrayLike<number>, tol = 1e-9) {
  assert.equal(got.length, expected.length, `length ${got.length} != ${expected.length}`);
  for (let i = 0; i < expected.length; i++) {
    const e = expected[i], g = got[i];
    if (Number.isNaN(e)) assert.ok(Number.isNaN(g), `index ${i}: expected NaN, got ${g}`);
    else assert.ok(Math.abs(g - e) <= tol, `index ${i}: ${g} != ${e}`);
  }
}

test("combinators: combineLatest([a,b]) feeding Sub matches the Python Pipeline", async () => {
  await ready();

  const a = Input("a");
  const b = Input("b");
  const y = Sub()(combineLatest([a, b]));
  const p = new Pipeline([a, b], [y]);

  // As-of (when_all) join on staggered integer index axes, then column0-column1.
  const av = new Float64Array([1, 2, 3, 4]);
  const ai = new Float64Array([0, 2, 4, 6]);
  const bv = new Float64Array([10, 20, 30]);
  const bi = new Float64Array([1, 3, 5]);

  const { values, index } = p({
    a: { values: av, index: ai },
    b: { values: bv, index: bi },
  }) as { values: Float64Array; index: Float64Array };

  // Ground truth cross-checked against the Python Pipeline for the same graph:
  //   Sub()(combine_latest(a, b)) -> a_latest - b_latest at each merged tick.
  assertNanEq(values, [-9, -8, -18, -17, -27, -26]);
  assertNanEq(index, [1, 2, 3, 4, 5, 6]);

  p.dispose();
});

test("combinators: resample(x, {mode:'by_index', agg:'mean', every:5}) buckets the index", async () => {
  await ready();

  const x = Input("x");
  const y = resample(x, { mode: "by_index", agg: "mean", every: 5 });
  const p = new Pipeline([x], [y]);

  // index 0..9, values 1..10; bars [0,5) and [5,10) along the index.
  const xv = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  const xi = new Float64Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

  const { values, index } = p({ values: xv, index: xi }) as {
    values: Float64Array;
    index: Float64Array;
  };

  // mean(1..5)=3 labelled at grid edge 0; mean(6..10)=8 labelled at 5.
  assertNanEq(values, [3, 8]);
  assertNanEq(index, [0, 5]);

  p.dispose();
});
