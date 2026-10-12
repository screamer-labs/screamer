import { test } from "node:test";
import assert from "node:assert/strict";
import { ready, Input, Pipeline, RollingMean, RollingCorr, Diff, Sub, select, combineLatest } from "../src/index.ts";
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
    assert.throws(() => d2.push("bars", 1, 5), (e: unknown) => e instanceof RangeError && /after events of 4/.test(e.message));
  } finally {
    p.dispose();
  }
});

test("live: the width of an input is locked until a new session", async () => {
  await ready();
  const a = Input("x");
  const p = new Pipeline([a], [a]);
  const range = (re: RegExp) => (e: unknown) => e instanceof RangeError && re.test((e as Error).message);
  try {
    let d = p.live();
    d.push("x", 0, 5);
    assert.throws(() => d.push("x", 1, [1, 2, 3, 4]), range(/after events of 1/));
    assert.throws(() => d.pushWide("x", 1, [1, 2]), range(/after events of 1/));
    d.push("x", 1, 6);
    assertNanEq(asFloat64(d.result()), [5, 6], "scalar");
    d = p.live();
    d.push("x", 0, [1, 2, 3, 4]);
    assert.throws(() => d.push("x", 1, 5), range(/after events of 4/));
    assert.throws(() => d.push("x", 1, [1, 2]), range(/after events of 4/));
    d.push("x", 1, [5, 6, 7, 8]);
    assertNanEq(asFloat64(d.result()), [1, 2, 3, 4, 5, 6, 7, 8], "wide");
    d = p.live();   // a new session starts with a fresh width
    d.push("x", 0, [1, 2]);
    assertNanEq(asFloat64(d.result()), [1, 2], "fresh");
    d = p.live();
    d.push("x", 0, 9);
    assertNanEq(asFloat64(d.result()), [9], "fresh scalar");
  } finally {
    p.dispose();
  }
});

test("live: null and numeric strings inside a wide event follow the batch feed", async () => {
  await ready();
  const a = Input("x");
  const p = new Pipeline([a], [a]);
  try {
    const d = p.live();
    d.push("x", 0, [null, 1] as any);
    d.push("x", 1, ["2", "3"] as any);
    assertNanEq(asFloat64(d.result()), [NaN, 1, 2, 3], "values");
  } finally {
    p.dispose();
  }
});

test("live: a nested sequence is a RangeError, like Python's ValueError", async () => {
  await ready();
  const a = Input("x");
  const p = new Pipeline([a], [a]);
  try {
    const d = p.live();
    assert.throws(() => d.push("x", 0, [[1, 2]] as any), (e: unknown) => e instanceof RangeError);
    assert.throws(() => d.push("x", 0, [new Float64Array(2)] as any), (e: unknown) => e instanceof RangeError);
    assert.throws(() => d.pushWide("x", 0, [[1], [2]] as any), (e: unknown) => e instanceof RangeError);
  } finally {
    p.dispose();
  }
});

test("live: a rejected first event sets no width", async () => {
  await ready();
  const a = Input("a");
  const p = new Pipeline([a], [RollingCorr(3)(a)]);
  try {
    const d = p.live();
    assert.throws(() => d.push("a", 0, 1));           // a number into a two-column edge
    for (let k = 0; k < 4; k++) d.push("a", k, [k, k * k]);
    assert.equal(asFloat64(d.result()).length, 4);
  } finally {
    p.dispose();
  }
});

test("live and batch: a fractional or NaN index is a TypeError", async () => {
  await ready();
  const a = Input("x");
  const p = new Pipeline([a], [a]);
  try {
    const d = p.live();
    for (const bad of [1.5, NaN, Infinity]) {
      assert.throws(() => d.push("x", bad, 1), (e: unknown) => e instanceof TypeError && /integer-valued/.test((e as Error).message));
      assert.throws(() => d.push("x", bad, [1, 2]), (e: unknown) => e instanceof TypeError);
      assert.throws(() => p({ values: [1, 2], index: [0, bad] }), (e: unknown) => e instanceof TypeError && /integer-valued/.test((e as Error).message));
    }
    d.push("x", 2, 1);
    assertNanEq(asFloat64(d.result()), [1], "ok");
  } finally {
    p.dispose();
  }
});

test("live: a rejected first event with a stateful sibling leaves no trace", async () => {
  await ready();
  const rows: number[][] = [];
  for (let k = 0; k < 6; k++) rows.push([100 + k, 101 + 2 * k, 99 - k, 102 + k * k]);
  const builds: Array<() => Pipeline> = [
    () => { const a = Input("a"); return new Pipeline([a], [Sub()(combineLatest([RollingMean(2)(select(a, [0])), select(a, [3])]))]); },
    () => { const a = Input("a"); return new Pipeline([a], [combineLatest([RollingMean(2)(select(a, [0])), select(a, [3])])]); },
  ];
  for (const build of builds) {
    const p = build();
    try {
      const run = (bad: boolean): Float64Array => {
        const d = p.live();
        if (bad) assert.throws(() => d.push("a", 0, [100, 200]), (e: unknown) => e instanceof RangeError);
        rows.forEach((r, k) => d.push("a", k, r));
        d.flush();
        return asFloat64(d.result());
      };
      const fresh = run(false);
      assertNanEq(run(true), fresh, "after a rejected event");
    } finally {
      p.dispose();
    }
  }
});

test("live and batch: one index rule, with advance", async () => {
  await ready();
  const a = Input("x");
  const p = new Pipeline([a], [a]);
  const kind = (e: unknown) => e instanceof TypeError && /integer-valued/.test((e as Error).message);
  try {
    const d = p.live();
    for (const bad of [2 ** 63, 1e300, -Infinity, NaN, 0.5]) {
      assert.throws(() => d.push("x", bad, 1), kind);
      assert.throws(() => d.advance(bad), kind);
      assert.throws(() => p({ values: [1], index: [bad] }), kind);
    }
    d.advance(5);
  } finally {
    p.dispose();
  }
});
