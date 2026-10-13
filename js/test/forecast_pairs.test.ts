import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { ready, forecastPairs, Input, Pipeline, isNode, RollingMean, dropna } from "../src/index.ts";
import type { NdArray, Output } from "../src/index.ts";

// forecastPairs against Python's forecast_pairs (devtools/wasm/gen_oracle.py
// writes the fixture). Both build the one C++ graph, so every value matches
// exactly: eager, the graph as a Pipeline node, and a live session.
const HERE = dirname(fileURLToPath(import.meta.url));
type Num = number | null;
interface Case {
  name: string;
  options: { count?: number; duration?: number; dropna?: boolean };
  X: { shape: number[]; values: Num[]; index: number[] };
  y: { values: Num[]; index: number[] };
  X_shifted: { shape: number[]; values: Num[] };
  y_out: Num[];
  node: { values: Num[]; index: number[] };
}
const cases: Case[] = JSON.parse(
  readFileSync(join(HERE, "..", "..", "wasm", "smoke", "oracle_forecast_pairs.json"), "utf8"));

const f64 = (xs: Num[]) => Float64Array.from(xs, (v) => (v === null ? NaN : v));
const same = (a: number, b: number) => (Number.isNaN(b) ? Number.isNaN(a) : Object.is(a, b) || a === b);

function assertSame(got: ArrayLike<number>, want: Float64Array, what: string) {
  assert.equal(got.length, want.length, `${what}: length`);
  for (let i = 0; i < want.length; i++)
    assert.ok(same(got[i], want[i]), `${what}: index ${i}: ${got[i]} vs ${want[i]}`);
}

function xValues(c: Case): Float64Array | NdArray {
  const data = f64(c.X.values);
  return c.X.shape.length === 2 ? { data, shape: c.X.shape } : data;
}

const isCount = (c: Case) => c.options.count !== undefined;

function eager(c: Case) {
  return isCount(c)
    ? forecastPairs(xValues(c), f64(c.y.values), c.options)
    : forecastPairs({ values: xValues(c), index: c.X.index }, { values: f64(c.y.values), index: c.y.index },
      c.options);
}

// The events of X and y in index order, X first on a tie (the batch merge order).
function events(c: Case): Array<[string, number, number | number[]]> {
  const K = c.X.shape[1] ?? 1;
  const xv = f64(c.X.values);
  const ev: Array<[string, number, number | number[], number]> = [];
  c.X.index.forEach((t, i) => ev.push(["X", t, K === 1 ? xv[i] : Array.from(xv.subarray(i * K, i * K + K)), 0]));
  c.y.index.forEach((t, i) => ev.push(["y", t, f64(c.y.values)[i], 1]));
  ev.sort((a, b) => a[1] - b[1] || a[3] - b[3]);
  return ev.map(([n, t, v]) => [n, t, v]);
}

test("forecastPairs: the fixture covers count, duration, (T, K) features, NaN, dropna and no rows", () => {
  const names = cases.map((c) => c.name);
  for (const n of ["count", "count dropna", "count (T,K) dropna", "duration", "duration dropna",
    "duration (T,K)", "duration empty"])
    assert.ok(names.includes(n), n);
});

test("forecastPairs: eager results equal Python's exactly", async () => {
  await ready();
  for (const c of cases) {
    const [xs, ys] = eager(c);
    const want = f64(c.X_shifted.values);
    if (c.X_shifted.shape.length === 2) {
      const nd = xs as NdArray;
      assert.deepEqual(nd.shape, c.X_shifted.shape, `${c.name}: shape`);
      assertSame(nd.data, want, `${c.name}: X_shifted`);
    } else {
      assert.ok(xs instanceof Float64Array, `${c.name}: a 1-D X gives a Float64Array`);
      assertSame(xs as Float64Array, want, `${c.name}: X_shifted`);
    }
    assertSame(ys, f64(c.y_out), `${c.name}: y`);
  }
});

test("forecastPairs: as a Pipeline node, batch and live equal Python's rows and index", async () => {
  await ready();
  for (const c of cases) {
    const X = Input("X"), y = Input("y");
    const node = forecastPairs(X, y, c.options);
    assert.ok(isNode(node));
    const p = new Pipeline([X, y], [node]);
    const xFeed = { values: xValues(c), index: c.X.index };
    const batch = p({ X: xFeed, y: { values: f64(c.y.values), index: c.y.index } }) as Output;
    const want = f64(c.node.values);
    const vals = (v: Output["values"]) => ((v as NdArray).data ?? (v as Float64Array));
    assertSame(vals(batch.values), want, `${c.name}: batch rows`);
    assertSame(batch.index, Float64Array.from(c.node.index), `${c.name}: batch index`);
    for (const every of [1, 3, 1_000_000]) {
      const live = p.live();
      const rows: number[] = [], index: number[] = [];
      const take = () => {
        const r = live.result() as Output;
        rows.push(...vals(r.values));
        index.push(...r.index);
      };
      events(c).forEach(([name, t, v], k) => {
        live.push(name, t, v);
        if ((k + 1) % every === 0) take();
      });
      live.flush();
      take();
      assertSame(rows, want, `${c.name}: live rows, result() every ${every}`);
      assertSame(index, Float64Array.from(c.node.index), `${c.name}: live index`);
    }
    p.dispose();
  }
});

test("forecastPairs: eager equals the node's columns, and rows and NdArray features agree", async () => {
  await ready();
  const c = cases.find((k) => k.name === "count (T,K) dropna")!;
  const [xs, ys] = eager(c);
  const K = c.X.shape[1], T = c.X.shape[0], d = f64(c.X.values);
  const rows = Array.from({ length: T }, (_, t) => Array.from(d.subarray(t * K, t * K + K), (v) => (Number.isNaN(v) ? null : v)));
  const [xr, yr] = forecastPairs(rows, f64(c.y.values), c.options);
  assert.deepEqual((xr as NdArray).shape, (xs as NdArray).shape);
  assertSame((xr as NdArray).data, (xs as NdArray).data, "rows");
  assertSame(yr, ys, "rows y");
  // each column alone gives the same pairs (count mode lags every column)
  const [x0] = forecastPairs(Array.from({ length: T }, (_, t) => rows[t][0]), f64(c.y.values),
    { count: 2 });
  const [xa] = forecastPairs(rows, f64(c.y.values), { count: 2 });
  const a = xa as NdArray;
  assertSame(Float64Array.from({ length: a.shape[0] }, (_, r) => a.data[r * K]), x0 as Float64Array, "column 0");
});

test("forecastPairs: options and inputs are checked as in Python", async () => {
  await ready();
  const v = [1, 2, 3, 4, 5, 6];
  assert.throws(() => forecastPairs(v, v, {}), (e: any) =>
    e instanceof RangeError && e.message === "pass exactly one of count= or duration=");
  assert.throws(() => forecastPairs(v, v, { count: 1, duration: 1 }), RangeError);
  assert.throws(() => forecastPairs(v, v, { count: 0 }), (e: any) =>
    e instanceof RangeError && /count must be at least 1/.test(e.message));
  assert.throws(() => forecastPairs(v, v, { count: "1" as any }), (e: any) =>
    e instanceof TypeError && e.message === "forecastPairs: count must be a number, got a string");
  assert.throws(() => forecastPairs(v, v, { count: 1, every: 2 } as any), (e: any) =>
    e instanceof TypeError && e.message === "forecastPairs: unknown option 'every'");
  assert.throws(() => forecastPairs(v, v, { count: 1, dropna: 1 as any }), TypeError);
  assert.throws(() => forecastPairs(v, v.slice(2), { count: 1 }), (e: any) =>
    e instanceof RangeError && e.message === "X and y must share the same length (time axis)");
  assert.throws(() => forecastPairs(v, v, { duration: 1 }), (e: any) =>
    e instanceof TypeError && /duration mode needs X and y/.test(e.message));
  assert.throws(() => forecastPairs({ values: v }, { values: v }, { count: 1 }), TypeError);
  assert.throws(() => forecastPairs(["a", 2] as any, [1, 2], { count: 1 }), (e: any) =>
    e instanceof TypeError && /item 0 of X is a string/.test(e.message));
  assert.throws(() => forecastPairs(Input("X"), v as any, { count: 1 }), TypeError);
  assert.throws(() => forecastPairs(v, v, { duration: NaN }), RangeError);
  assert.throws(() => forecastPairs({ values: v }, { values: v }, { duration: -1 }), (e: any) =>
    e instanceof RangeError && /duration must be at least 0/.test(e.message));
  assert.throws(() => forecastPairs(v, v, { count: 1e30 }), (e: any) =>
    e instanceof RangeError && /count must be at least 1 and at most 2147483647, got 1e\+30/.test(e.message));
  // y is one column in both modes; X is (T,) or (T, K)
  const two = { data: new Float64Array(12), shape: [6, 2] };
  const three = { data: new Float64Array(12), shape: [6, 2, 1] };
  assert.throws(() => forecastPairs(v, two, { count: 1 }), (e: any) =>
    e instanceof RangeError && e.message === "forecast_pairs: y holds 2 columns, expected 1");
  assert.throws(() => forecastPairs({ values: v }, { values: two }, { duration: 1 }), RangeError);
  assert.throws(() => forecastPairs(three, v, { count: 1 }), (e: any) =>
    e instanceof RangeError && e.message === "forecast_pairs: X has 3 dimensions, expected (T,) or (T, K)");
  assert.throws(() => forecastPairs({ values: three }, { values: v }, { duration: 1 }), RangeError);
  // a node checks y's width when the widths are known
  const Xn = Input("X"), yn2 = Input("y");
  const pn = new Pipeline([Xn, yn2], [forecastPairs(Xn, yn2, { count: 1 })]);
  assert.throws(() => pn({ X: v, y: two }), (e: any) =>
    e instanceof RangeError && /y holds 2 columns/.test(e.message));
  pn.dispose();
  // a float horizon is truncated and null is NaN, as in Python
  const [a] = forecastPairs(v, v, { count: 1.5 });
  const [b] = forecastPairs(v, v, { count: 1 });
  assertSame(a as Float64Array, b as Float64Array, "count 1.5");
  const [xn, yn] = forecastPairs([1, null, 3], [1, 2, null], { count: 1 });
  assertSame(xn as Float64Array, Float64Array.from([NaN, 1, NaN]), "null X");
  assertSame(yn, Float64Array.from([1, 2, NaN]), "null y");
  // duration with no index uses row numbers, as Python's (values, None)
  const [xd] = forecastPairs({ values: v }, { values: v }, { duration: 1 });
  assertSame(xd as Float64Array, Float64Array.from([1, 2, 3, 4, 5]), "duration, row numbers");
});

test("GraphBuilder: a builder compiles once", async () => {
  await ready();
  const { current } = await import("../src/index.ts");
  const { wasmCall } = await import("../src/errors.ts");
  const M: any = current();
  const gb = new M.GraphBuilder();
  try {
    const x = gb.addInput();
    const ins = new M.VectorSizeT(); ins.push_back(x);
    const out = new M.VectorSizeT(); out.push_back(gb.addDropna(ins, false));
    gb.setOutputs(out);
    ins.delete(); out.delete();
    const cg = gb.compile();
    cg.delete();
    assert.throws(() => wasmCall(() => gb.compile()), /compiled already/);
  } finally {
    gb.delete();
  }
});

test("Pipeline: a rank-3 feed raises in every graph; forecastPairs refuses an X of no columns", async () => {
  await ready();
  const v = Float64Array.from({ length: 8 }, (_, i) => i);
  const rank3 = { data: new Float64Array(16), shape: [8, 2, 1] };
  const empty = { data: new Float64Array(0), shape: [8, 0] };
  const graphs: Array<[string, () => any, string[]]> = [
    ["functor", () => { const x = Input("x"); return new Pipeline([x], [RollingMean(2)(x)]); }, ["x"]],
    ["dropna", () => { const x = Input("x"); return new Pipeline([x], [dropna(x)]); }, ["x"]],
    ["forecast count", () => {
      const X = Input("X"), y = Input("y");
      return new Pipeline([X, y], [forecastPairs(X, y, { count: 1 })]);
    }, ["X", "y"]],
    ["forecast duration", () => {
      const X = Input("X"), y = Input("y");
      return new Pipeline([X, y], [forecastPairs(X, y, { duration: 1 })]);
    }, ["X", "y"]],
  ];
  for (const [name, make, inputs] of graphs) {
    const p = make();
    const feeds = (first: unknown) =>
      Object.fromEntries(inputs.map((n, i) => [n, i === 0 ? first : v]));
    assert.throws(() => p(feeds(rank3)), (e: any) => e instanceof RangeError &&
      e.message === "Pipeline: the values of input 0 have shape [8,2,1], expected (T,) or (T, w)", name);
    if (name.startsWith("forecast"))     // the graph requires X to have a column
      assert.throws(() => p(feeds(empty)), (e: any) => e instanceof RangeError &&
        e.message === "forecast_pairs: X has 0 columns, expected at least 1", name);
    p.dispose();
  }
});

test("forecastPairs: a large count costs memory only as events arrive", async () => {
  await ready();
  const v = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  const [xs, ys] = forecastPairs(v, v, { count: 2 ** 31 - 1 });
  assert.ok(Array.from(xs as Float64Array).every(Number.isNaN));
  assertSame(ys, Float64Array.from(v), "y");
  const wide = { data: new Float64Array(30).fill(1), shape: [10, 3] };
  const [xw] = forecastPairs(wide, v, { count: 2 ** 31 - 1 });
  assert.deepEqual((xw as NdArray).shape, [10, 3]);
});

test("forecastPairs: a (T, 0) X raises eagerly and in a node, in both modes", async () => {
  await ready();
  const v = [0, 1, 2, 3, 4, 5];
  const empty = { data: new Float64Array(0), shape: [6, 0] };
  const msg = "forecast_pairs: X has 0 columns, expected at least 1";
  const is = (e: any) => e instanceof RangeError && e.message === msg;
  assert.throws(() => forecastPairs(empty, v, { count: 1 }), is);
  assert.throws(() => forecastPairs({ values: empty }, { values: v }, { duration: 1 }), is);
  for (const opts of [{ count: 1 }, { duration: 1 }]) {
    const X = Input("X"), y = Input("y");
    const p = new Pipeline([X, y], [forecastPairs(X, y, opts)]);
    assert.throws(() => p({ X: empty, y: v }), is);
    const out = p({ X: v, y: v }) as Output;           // nothing was pushed
    assert.deepEqual((out.values as NdArray).shape[1], 2);
    assert.throws(() => p.live().push("X", 0, []), RangeError);
    p.dispose();
  }
});
