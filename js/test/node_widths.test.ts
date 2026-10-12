import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ready, Input, Pipeline, filter, select, resample, combineLatest,
  RollingMean, ExpandingMax, MovingAverage, type Output,
} from "../src/index.ts";

// Pipeline node widths and outputs, and the Task 18b leftovers (Task 18c):
// A11 outputs of mixed widths, G10 Filter, G11 width errors, G14 ohlcv_bars,
// G16 mixed event containers, Q1 vector settings. Mirrors
// tests/test_pipeline_node_widths.py.

const nd = (rows: number[][]) => ({ data: Float64Array.from(rows.flat()), shape: [rows.length, rows[0].length] });
const arr = (x: any): number[] => Array.from((x?.data ?? x) as Float64Array);
const isRange = (re: RegExp) => (e: unknown) => e instanceof RangeError && re.test((e as Error).message);

test("A11: outputs of mixed widths keep their own columns", async () => {
  await ready();
  const a = Input("a");
  const p = new Pipeline([a], [a, select(a, [1]), select(a, [2, 0])]);
  try {
    const rows = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    const out = p(nd(rows) as any) as Output[];
    assert.deepEqual(arr(out[0].values), rows.flat());
    assert.deepEqual((out[0].values as any).shape, [3, 3]);
    assert.deepEqual(arr(out[1].values), [2, 5, 8]);
    assert.deepEqual(arr(out[2].values), [3, 1, 6, 4, 9, 7]);
    assert.deepEqual((out[2].values as any).shape, [3, 2]);
    for (const o of out) assert.deepEqual(Array.from(o.index), [0, 1, 2]);
  } finally {
    p.dispose();
  }
});

test("G10: Filter refuses a wide data or mask stream, batch and live", async () => {
  await ready();
  const d = Input("d"), m = Input("m");
  const p = new Pipeline([d, m], [filter(d, m)]);
  try {
    const wide = nd([[1, 1], [1, 1]]);
    assert.throws(() => p({ d: wide, m: [1, 0] } as any), isRange(/Filter takes a 1-column data stream/));
    assert.throws(() => p({ d: [1, 2], m: wide } as any), isRange(/Filter takes a 1-column mask stream/));
    const s = p.live();
    assert.throws(() => s.push("m", 0, [1, 1]), isRange(/1-column mask/));
    // Rejected before any node saw it: the session runs as a fresh one.
    [[5, 1], [6, 0], [7, 2]].forEach(([x, k], i) => { s.push("d", i, x); s.push("m", i, k); });
    s.flush();
    const live = s.result();
    const batch = p({ d: [5, 6, 7], m: [1, 0, 2] }) as Output;
    assert.deepEqual(arr(live.values), arr(batch.values));
    assert.deepEqual(arr(live.values), [5, 7]);
  } finally {
    p.dispose();
  }
});

test("G11: every node raises RangeError for a width it cannot take, batch and live", async () => {
  await ready();
  const graphs: [string, () => Pipeline, RegExp][] = [
    ["operator", () => { const a = Input("a"); return new Pipeline([a], [RollingMean(2)(a) as any]); }, /operator takes 1 column/],
    ["select", () => { const a = Input("a"); return new Pipeline([a], [select(a, [2])]); }, /select needs column 2/],
    ["reducer resample", () => { const a = Input("a"), x = Input("x"); return new Pipeline([a], [resample(a, { every: 2, reducer: ExpandingMax()(x) as any })]); }, /reducer resample takes 1 column/],
    ["plan resample", () => { const a = Input("a"); return new Pipeline([a], [resample(a, { every: 2, agg: "ohlcv" })]); }, /plan takes 2/],
    ["cumulative resample", () => { const a = Input("a"), b = Input("b"); return new Pipeline([a, b], [resample(a, { threshold: 5, driver: b, agg: "last" })]); }, /cumulative resample takes/],
    ["clock max_age", () => { const a = Input("a"), b = Input("b"); return new Pipeline([a, b], [resample(a, { clock: true, driver: b, agg: "last", fill: "carry", maxAge: 3 })]); }, /max_age takes 1 value column/],
    ["filter", () => { const a = Input("a"), b = Input("b"); return new Pipeline([a, b], [filter(a, b)]); }, /Filter takes a 1-column data/],
  ];
  for (const [name, make, re] of graphs) {
    const p = make();
    try {
      const wide = name === "plan resample" ? nd([[1, 2, 3], [4, 5, 6]]) : nd([[1, 2], [3, 4]]);
      const two = p.inputs.length === 2;
      const feeds: any = two ? { a: { values: wide, index: [0, 1] }, b: { values: [1, 1], index: [0, 1] } } : { values: wide, index: [0, 1] };
      assert.throws(() => p(feeds), isRange(re), `${name} batch`);
      const s = p.live();
      if (two) s.push("b", 0, 1);
      assert.throws(() => s.push("a", 0, Array.from(wide.data.subarray(0, wide.shape[1]))), isRange(re), `${name} live`);
    } finally {
      p.dispose();
    }
  }
});

test("G11: past the pre-check, each node refuses a width it cannot take", async () => {
  await ready();
  // An empty feed leaves b's width unknown to the pre-check; an on_any join
  // over it emits rows of its compile-time width, which the node refuses.
  const cases: [string, (j: any, m: any) => any, RegExp][] = [
    ["operator", (j) => RollingMean(2)(j), /an operator takes 1 column\(s\), but its input has 2/],
    ["select", (j) => select(j, [5]), /select needs column 5, but its input has 2/],
    ["filter", (j, m) => filter(j, m), /Filter takes a 1-column data stream, but its data has 2/],
    ["reducer resample", (j) => resample(j, { every: 2, reducer: ExpandingMax()(Input("r")) as any }), /reducer resample takes 1 column, but its input has 2/],
    ["clock max_age", (j, m) => resample(j, { clock: true, driver: m, agg: "last", fill: "carry", maxAge: 3 }), /max_age takes 1 value column, but its input has 2/],
  ];
  for (const [name, build, re] of cases) {
    const a = Input("a"), b = Input("b"), m = Input("m");
    const uses = name === "filter" || name === "clock max_age";
    const ins = uses ? [a, b, m] : [a, b];
    const p = new Pipeline(ins, [build(combineLatest([a, b], { emit: "on_any" }), m)]);
    try {
      const x = { values: [1, 2, 3, 4], index: [0, 1, 2, 3] };
      const feeds: any = { a: x, b: { values: [], index: [] } };
      if (uses) feeds.m = x;
      assert.throws(() => p(feeds), isRange(re), name);
    } finally {
      p.dispose();
    }
  }
});

test("cumulative resample with a driver joins value and driver first, as in Python", async () => {
  await ready();
  const a = Input("a"), b = Input("b");
  const p = new Pipeline([a, b], [resample(a, { threshold: 5, driver: b, agg: "last" })]);
  const q = new Pipeline([a, b], [resample(combineLatest([a, b]), { threshold: 5, agg: "last" })]);
  try {
    const feeds = { a: [1, 2, 3, 4], b: [3, 3, 3, 3] };
    const out = p(feeds) as Output, ref = q(feeds) as Output;
    assert.deepEqual(arr(out.values), arr(ref.values));
    assert.deepEqual(arr(out.values), [2, 4]);
  } finally {
    p.dispose();
    q.dispose();
  }
});

test("G14: ohlcv_bars takes at least 5 columns", async () => {
  await ready();
  const a = Input("a"), c = Input("c");
  const p = new Pipeline([a], [resample(a, { every: 2, agg: "ohlcv_bars" })]);
  const q = new Pipeline([a, c], [resample(a, { clock: true, driver: c, agg: "ohlcv_bars" })]);
  try {
    for (const w of [2, 3, 4]) {
      const rows = Array.from({ length: 4 }, (_, k) => Array.from({ length: w }, (_, j) => k * 10 + j));
      assert.throws(() => p({ values: nd(rows), index: [0, 1, 2, 3] } as any), isRange(/at least 5 columns/), `every, width ${w}`);
      assert.throws(() => q({ a: { values: nd(rows), index: [0, 1, 2, 3] }, c: { values: [1], index: [2] } } as any), isRange(/at least 5 columns/), `clock, width ${w}`);
      assert.throws(() => p.live().push("a", 0, rows[0]), isRange(/at least 5 columns/), `live, width ${w}`);
    }
    const rows = [[1, 5, 0, 2, 10], [2, 6, 1, 3, 20]];
    const out = p({ values: nd(rows), index: [0, 1] } as any) as Output;
    assert.deepEqual(arr(out.values), [1, 6, 0, 3, 30]);
  } finally {
    p.dispose();
    q.dispose();
  }
});

test("G16: an Array of events may mix a Float64Array and a number[] after a typed first event", async () => {
  await ready();
  const want = [[NaN, NaN], [2, 3]];
  const asRows = (r: any[]) => r.map((x) => Array.from(x as Float64Array));
  assert.deepEqual(asRows(RollingMean(2)([Float64Array.from([1, 2]), [3, 4]] as any) as any), want);
  assert.deepEqual(asRows(RollingMean(2)([Float64Array.from([1, 2]), Float64Array.from([3, 4])]) as any), want);
  assert.deepEqual(asRows(RollingMean(2)([Float64Array.from([1, 2]), [3, null]] as any) as any), [[NaN, NaN], [2, NaN]]);
  // An NdArray event fixes an N-D shape; a nested Array of it follows.
  const nested = RollingMean(2)([nd([[1, 2], [3, 4]]), [[3, 4], [5, 6]]] as any) as any[];
  assert.deepEqual(Array.from(nested[1].data), [2, 3, 4, 5]);
  assert.deepEqual(nested[1].shape, [2, 2]);
  // Another length is a RangeError, a string a TypeError.
  assert.throws(() => RollingMean(2)([Float64Array.from([1, 2]), [3, 4, 5]] as any), isRange(/same shape/));
  assert.throws(() => RollingMean(2)([Float64Array.from([1, 2]), 3] as any), isRange(/same shape/));
  assert.throws(() => RollingMean(2)([Float64Array.from([1, 2]), [3, "a"]] as any), TypeError);
  // A number[] first event keeps its meaning (an event of numbers), as in Python.
  assert.throws(() => RollingMean(2)([[1, 2], Float64Array.from([3, 4])] as any), isRange(/expected a number/));
});

test("Q1: a vector setting is a number, an Array or a typed array, and every element a number", async () => {
  await ready();
  const bad: [string, () => unknown, RegExp][] = [
    ["Set of strings", () => MovingAverage(new Set(["0.5", "0.5"]) as any), /got a Set/],
    ["Set of numbers", () => MovingAverage(new Set([0.5, 0.25]) as any), /got a Set/],
    ["generator of numbers", () => MovingAverage((function* () { yield 0.5; yield 0.5; })() as any), /got a /],
    ["nested array of a string", () => MovingAverage([["3"]] as any), /got an array/],
    ["nested array of a number", () => MovingAverage([[3]] as any), /got an array/],
    ["array with an object", () => MovingAverage([0.5, {} as any]), /got an Object/],
  ];
  for (const [name, f, re] of bad)
    assert.throws(f, (e: unknown) => e instanceof TypeError && /^taps must be a number or an array of numbers, got /.test((e as Error).message) && re.test((e as Error).message), name);
  // An Array, a typed array or a number builds.
  assert.deepEqual(MovingAverage([0.5, 0.25])([2, 4, 6]), [NaN, 2.5, 4]);
  assert.deepEqual(MovingAverage(Float64Array.from([0.5, 0.25]) as any)([2, 4, 6]), [NaN, 2.5, 4]);
  assert.deepEqual(MovingAverage(2)([2, 4, 6]), [4, 8, 12]);
});
