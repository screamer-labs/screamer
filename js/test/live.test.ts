import { test } from "node:test";
import assert from "node:assert/strict";
import { ready, Input, Pipeline, RollingMean, Diff } from "../src/index.ts";
import type { Output } from "../src/index.ts";

// NaN-aware element equality.
function assertNanEq(got: ArrayLike<number>, expected: ArrayLike<number>, label: string, tol = 1e-9) {
  assert.equal(got.length, expected.length, `${label}: length ${got.length} != ${expected.length}`);
  for (let i = 0; i < expected.length; i++) {
    const e = expected[i], g = got[i];
    if (Number.isNaN(e)) assert.ok(Number.isNaN(g), `${label}[${i}]: expected NaN, got ${g}`);
    else assert.ok(Math.abs(g - e) <= tol, `${label}[${i}]: ${g} != ${e}`);
  }
}

function asFloat64(out: Output): Float64Array {
  const v = out.values as any;
  return v instanceof Float64Array ? v : (v.data as Float64Array);
}

// The batch==stream invariant: driving a pipeline event-by-event through
// .live() must reproduce its batch call() output on the same data exactly.
test("live: functor chain streamed event-by-event equals batch", async () => {
  await ready();

  const x = Input("x");
  const y = Diff(1)(RollingMean(3)(x));
  const p = new Pipeline([x], [y]);

  const values = new Float64Array([0, 1, 2, 3, 4, 5, 6, 7]);
  const index = new Float64Array([0, 1, 2, 3, 4, 5, 6, 7]);

  try {
    // Batch reference.
    const batch = p({ values, index }) as Output;

    // Stream the same data one event at a time, then flush and drain.
    const live = p.live();
    for (let i = 0; i < values.length; i++) live.push("x", index[i], values[i]);
    live.flush();
    const streamed = live.result();

    assertNanEq(asFloat64(streamed), asFloat64(batch), "values");
    assertNanEq(streamed.index, batch.index, "index");

    // Sanity: the chain's known output (Diff of RollingMean(3) of a unit ramp).
    assertNanEq(asFloat64(streamed), [NaN, NaN, NaN, 1, 1, 1, 1, 1], "expected");
  } finally {
    p.dispose();
  }
});

test("live: push takes a sequence as one wide event; pushWide is an alias", async () => {
  await ready();
  const a = Input("bars");
  const p = new Pipeline([a], [a]);
  try {
    const d = p.live();
    d.push("bars", 0, [100, 102, 99.5, 101.5]);
    d.push("bars", 1, new Float64Array([1, 2, 3, 4]));
    d.push("bars", 2, { data: new Float64Array([5, 6, 7, 8]), shape: [4] });
    d.pushWide("bars", 3, [9, 10, 11, 12]);
    d.push("bars", 4, [true, false, 1, null] as any);
    assertNanEq(asFloat64(d.result()), [100, 102, 99.5, 101.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 0, 1, NaN], "values");
    const d2 = p.live();
    assert.throws(() => d2.push("bars", 0, []), (e: unknown) => e instanceof RangeError);
    assert.throws(() => d2.push("bars", 0, { data: new Float64Array(3), shape: [2, 2] }), (e: unknown) => e instanceof RangeError);
    assert.throws(() => d2.push("bars", 0, { data: new Float64Array(4), shape: [2, 2] }), (e: unknown) => e instanceof RangeError);
    assert.throws(() => d2.push("bars", 0, [1, {}, 3, 4] as any), (e: unknown) => e instanceof TypeError);
    d2.push("bars", 0, [1, 2, 3, 4]);
    assert.throws(() => d2.push("bars", 1, [1, 2]), (e: unknown) => e instanceof RangeError);
  } finally {
    p.dispose();
  }
});
