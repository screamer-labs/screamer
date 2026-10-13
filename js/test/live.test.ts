import { test } from "node:test";
import assert from "node:assert/strict";
import { ready, Input, Pipeline, RollingMean, RollingCorr, Diff, Sub, select, combineLatest, resample } from "../src/index.ts";
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

test("live: null inside a wide event is NaN and a string is a TypeError, as in the batch feed", async () => {
  await ready();
  const a = Input("x");
  const p = new Pipeline([a], [a]);
  try {
    const d = p.live();
    d.push("x", 0, [null, 1] as any);
    assert.throws(() => d.push("x", 1, ["2", "3"] as any), TypeError);
    d.push("x", 1, [2, 3]);
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
    for (const [bad, msg] of [[1.5, /integer-valued/], [NaN, /finite/], [Infinity, /finite/]] as const) {
      assert.throws(() => d.push("x", bad, 1), (e: unknown) => e instanceof TypeError && msg.test((e as Error).message));
      assert.throws(() => d.push("x", bad, [1, 2]), (e: unknown) => e instanceof TypeError);
      assert.throws(() => p({ values: [1, 2], index: [0, bad] }), (e: unknown) => e instanceof TypeError && msg.test((e as Error).message));
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
  const cases: Array<[unknown, RegExp]> = [
    [2 ** 63, /int64 range/], [1e300, /int64 range/], [-(2 ** 63) * 2, /int64 range/],
    [-Infinity, /finite/], [NaN, /finite/], [0.5, /integer-valued/],
    ["5", /must be a number.*string/], [null, /must be a number.*null/],
    [true, /must be a number.*boolean/], [{ valueOf: () => 5 }, /must be a number/],
    [undefined, /must be a number/],
  ];
  try {
    const d = p.live();
    for (const [bad, msg] of cases) {
      const kind = (e: unknown) => e instanceof TypeError && msg.test((e as Error).message);
      assert.throws(() => d.push("x", bad as any, 1), kind);
      assert.throws(() => d.push("x", bad as any, [1, 2]), kind);
      assert.throws(() => d.pushWide("x", bad as any, [1, 2]), kind);
      assert.throws(() => d.advance(bad as any), kind);
      assert.throws(() => p({ values: [1], index: [bad as any] }), kind);
    }
    d.advance(5);
    d.push("x", 6, 1);
    d.push("x", 2 ** 53, 2);
    assertNanEq(p({ values: [1, 2], index: new Float64Array([0, 1]) }).index as Float64Array, [0, 1], "f64 index");
  } finally {
    p.dispose();
  }
});

// Rows for a 5-input-column bar stream and a clock at 2, 5, 8, 11.
function barFeeds(width: number) {
  const n = 12;
  const v = new Float64Array(n * width);
  for (let i = 0; i < v.length; i++) v[i] = i;
  return {
    x: { values: { data: v, shape: [n, width] }, index: Float64Array.from({ length: n }, (_, i) => i) },
    c: { values: [1, 1, 1, 1], index: [2, 5, 8, 11] },
  };
}

function clockAndBars(): Pipeline {
  const x = Input("x"), c = Input("c");
  return new Pipeline([x, c], [resample(x, { clock: true, driver: c, agg: "last" }), resample(x, { every: 100, agg: "ohlcv_bars" })]);
}

function sameOutputs(a: Output[], b: Output[], label: string) {
  assert.equal(a.length, b.length);
  for (let o = 0; o < a.length; o++) {
    assertNanEq(asFloat64(a[o]), asFloat64(b[o]), `${label} values ${o}`);
    assertNanEq(a[o].index, b[o].index, `${label} index ${o}`);
  }
}

test("pipeline: a reused pipeline forgets learned resample plans", async () => {
  await ready();
  const p = clockAndBars();
  try {
    for (const w of [5, 6, 5, 7]) {
      const fresh = clockAndBars();
      try {
        sameOutputs(p(barFeeds(w) as any) as Output[], fresh(barFeeds(w) as any) as Output[], `width ${w}`);
      } finally {
        fresh.dispose();
      }
    }
  } finally {
    p.dispose();
  }
});

test("live: a rejected ohlcv_bars width, then a new session, equals a fresh pipeline", async () => {
  await ready();
  const x = Input("x");
  const p = new Pipeline([x], [resample(x, { every: 4, agg: "ohlcv_bars" })]);
  try {
    const d = p.live();
    assert.throws(() => d.push("x", 0, [1, 2]), (e: unknown) => e instanceof RangeError && /ohlcv_bars/.test((e as Error).message));
    for (const w of [6, 5]) {
      const rows = Array.from({ length: 9 }, (_, k) => Array.from({ length: w }, (_, j) => k * 10 + j));
      const s = p.live();
      rows.forEach((r, k) => s.push("x", k, r));
      s.flush();
      const live = s.result() as Output;   // before the batch call, which resets the engine
      const batch = p({ values: { data: Float64Array.from(rows.flat()), shape: [rows.length, w] },
                        index: rows.map((_, k) => k) }) as Output;
      assert.equal(live.index.length, 3, `width ${w}`);
      assertNanEq(asFloat64(live), asFloat64(batch), `width ${w}`);
    }
  } finally {
    p.dispose();
  }
});

test("live: a poisoned session refuses result until a new session", async () => {
  await ready();
  const a = Input("a"), b = Input("b");
  const p = new Pipeline([a, b], [combineLatest([a, b], { maxPending: 2 })]);
  try {
    const d = p.live();
    assert.throws(() => { for (let k = 0; k < 5; k++) d.push("a", k, 1); }, /buffer overflow/);
    for (const call of [() => d.result(), () => d.flush(), () => d.advance(9), () => d.push("b", 0, 1)])
      assert.throws(call, /reset/);
    const s = p.live();
    s.push("a", 0, 1);
    s.push("b", 0, 2);
    s.flush();
    assertNanEq(asFloat64(s.result()), [1, 2], "fresh");
  } finally {
    p.dispose();
  }
});

test("pipeline: a functor fed by two nodes needs each to carry its columns", async () => {
  await ready();
  const a = Input("a");
  const p = new Pipeline([a], [Sub()(select(a, [0]), select(a, [1]))]);
  try {
    assert.throws(() => p({ values: { data: new Float64Array([1, 2, 3, 4]), shape: [2, 2] }, index: [0, 1] }),
      (e: unknown) => e instanceof RangeError && /CombineLatest/.test((e as Error).message));
    assert.throws(() => p.live().push("a", 0, [1, 2]),
      (e: unknown) => e instanceof RangeError && /CombineLatest/.test((e as Error).message));
  } finally {
    p.dispose();
  }
});

test("pipeline: a batch feed of NdArray values is read, a bad shape is a RangeError", async () => {
  await ready();
  const a = Input("a");
  const p = new Pipeline([a], [a]);
  const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]);
  try {
    // Flat values are one value per row (G32): never read as rows of the
    // index's length.
    assert.throws(() => p({ values: data, index: [0, 1] }),
      { name: "RangeError", message: "Pipeline: input 0 has 8 rows of values but 2 index values" });
    const nd = p({ values: { data, shape: [2, 4] } as any, index: [0, 1] }) as Output;
    assertNanEq(asFloat64(nd), data, "keyed NdArray");
    assert.deepEqual((nd.values as any).shape, [2, 4]);
    assertNanEq(asFloat64(p({ values: { data, shape: [2, 4] } as any }) as Output), data, "no index");
    assertNanEq(asFloat64(p({ data, shape: [2, 4] } as any) as Output), data, "bare NdArray");
    assertNanEq(asFloat64(p({ a: { data, shape: [8] } } as any) as Output), data, "1-D in a record");
    for (const bad of [{ data, shape: [2, 2, 2] }, { data, shape: [3, 4] }, { data, shape: [] }])
      assert.throws(() => p({ values: bad as any, index: [0, 1] }), RangeError);
    assert.throws(() => p({ values: { data, shape: [2, 4] } as any, index: [0, 1, 2] }), RangeError);
    assert.throws(() => p({ values: { foo: 1 } as any, index: [0] }), TypeError);
  } finally {
    p.dispose();
  }
});

// A value's valueOf can dispose the Pipeline while push converts it; the push
// then raises a TypeError and never reaches the released graph.
test("live: a push whose value disposes the Pipeline raises TypeError", async () => {
  await ready();
  for (const wide of [false, true]) {
    const x = Input("x");
    const p = new Pipeline([x], [wide ? select(x, [0, 1]) : RollingMean(3)(x)]);
    const live = p.live();
    live.push("x", 0, wide ? [1.0, 1.0] : 1.0);
    const evil = { valueOf: () => { p.dispose(); return 2.0; } };
    assert.throws(() => live.push("x", 1, (wide ? [evil, 3.0] : evil) as any), TypeError);
    p.dispose();
  }
});

// flush() ends the input (the twin of tests/test_live_flush_ends_input.py).
test("live: push and advance after flush() throw RangeError; a new session starts over", async () => {
  await ready();
  const a = Input("a"), b = Input("b");
  const p = new Pipeline([a, b], [combineLatest([a, b])]);
  try {
    const s = p.live();
    s.push("a", 1, 1); s.push("b", 1, 2);
    s.flush();
    s.flush();                                  // a repeated flush does nothing
    assert.deepEqual(Array.from((s.result() as Output).index as any), [1]);
    const ended = (e: unknown) => e instanceof RangeError &&
      /flush\(\) ended this session's input; start a new live\(\) session/.test((e as Error).message);
    assert.throws(() => s.push("a", 3, 1), ended);
    assert.throws(() => s.push("a", 3, [1]), ended);
    assert.throws(() => s.advance(5), ended);
    assert.equal(((s.result() as Output).index as any).length, 0);
    const t = p.live();
    for (const [nm, k, v] of [["a", 3, 1], ["b", 2, 2], ["a", 4, 3], ["b", 5, 4]] as const) t.push(nm, k, v);
    t.flush();
    assert.deepEqual(Array.from((t.result() as Output).index as any), [3, 4, 5]);
  } finally { p.dispose(); }
});
