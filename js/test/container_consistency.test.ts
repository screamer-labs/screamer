// 3.0.0 container consistency, tracker rows A7 to A10. Each case mirrors
// tests/test_container_consistency.py and tests/test_compat.py in Python.
import { test } from "node:test";
import assert from "node:assert/strict";
import * as api from "../src/index.ts";
import type { NdArray } from "../src/index.ts";

const a = api as any;
const nd = (shape: number[], data: ArrayLike<number>): NdArray => ({ shape, data: Float64Array.from(data) });
const flat = (x: any): number[] => (typeof x === "number" ? [x] : Array.from(x instanceof Float64Array ? x : Array.isArray(x) ? x : x.data));

// NaN-aware equality of two flat sequences.
function eq(got: ArrayLike<number>, want: ArrayLike<number>, what = "") {
  assert.equal(got.length, want.length, `${what} length`);
  for (let i = 0; i < want.length; i++) {
    if (Number.isNaN(want[i])) assert.ok(Number.isNaN(got[i]), `${what} [${i}] expected NaN, got ${got[i]}`);
    else assert.ok(Math.abs(got[i] - want[i]) <= 1e-12, `${what} [${i}] ${got[i]} != ${want[i]}`);
  }
}

const T = 8;
const xs = Array.from({ length: T }, (_, i) => Math.sin(i + 1) * 3 + i);
const ys = Array.from({ length: T }, (_, i) => Math.cos(2 * i + 1) * 2 - i);
const zs = Array.from({ length: T }, (_, i) => ((i * 7) % 5) - 2);

// ------------------------------------------------------------------ A7

const engine = (Tn = 5, K = 3) => nd([Tn, K, 4], Array.from({ length: Tn * K * 4 }, (_, i) => Math.sin(i) * 2 + i / 10));

test("A7: an Array of engine rows gives an Array of result rows", async () => {
  await api.ready();
  const e = engine(), [Tn, K] = e.shape;
  const want = a.PortfolioReport()(e) as NdArray;
  const rows = Array.from({ length: Tn }, (_, t) =>
    Array.from({ length: K }, (_, k) => Array.from(e.data.slice((t * K + k) * 4, (t * K + k + 1) * 4))));
  const got = a.PortfolioReport()(rows);
  assert.ok(Array.isArray(got));
  assert.equal(got.length, Tn);
  for (let t = 0; t < Tn; t++) {
    assert.ok(got[t] instanceof Float64Array);
    eq(got[t], want.data.slice(t * 6, (t + 1) * 6), `row ${t}`);
  }
});

test("A7: an NdArray, one event, and a stream keep their results", async () => {
  await api.ready();
  const e = engine(), [Tn, K] = e.shape;
  const want = a.PortfolioReport()(e) as NdArray;
  assert.deepEqual(want.shape, [Tn, 6]);
  const events = Array.from({ length: Tn }, (_, t) => nd([K, 4], e.data.slice(t * K * 4, (t + 1) * K * 4)));
  const streamed = [...a.PortfolioReport()(events.values())];
  streamed.forEach((r: number[], t: number) => eq(r, want.data.slice(t * 6, (t + 1) * 6), `stream ${t}`));
  const one = a.PortfolioReport()(events[0]);
  assert.ok(Array.isArray(one) && one.length === 6);
  // An Array of (K, 4) NdArray events is an Array of rows too, as a Python list
  // of (K, 4) arrays is.
  const listed = a.PortfolioReport()(events);
  assert.ok(Array.isArray(listed) && listed.length === Tn);
  listed.forEach((r: Float64Array, t: number) => eq(r, want.data.slice(t * 6, (t + 1) * 6), `NdArray row ${t}`));
});

// ------------------------------------------------------------------ A8

const TWO = () => [a.RollingCorr(3), a.Add(), a.RollingBeta(4), a.RollingLinearRegression(4)];
const THREE = () => [a.Where(), a.ATR(3), a.Stoch(3)];

test("A8: a number[] mixed with a Float64Array gives the array result, either order", async () => {
  await api.ready();
  for (const op of TWO()) {
    const want = op(Float64Array.from(xs), Float64Array.from(ys));
    for (const args of [[xs, Float64Array.from(ys)], [Float64Array.from(xs), ys]]) {
      const got = op(...args);
      assert.ok(!Array.isArray(got), "an array container");
      eq(flat(got), flat(want));
      if (!(want instanceof Float64Array)) assert.deepEqual(got.shape, want.shape);
    }
  }
});

test("A8: three inputs, any mix of number[] and arrays", async () => {
  await api.ready();
  const hi = xs.map((v) => Math.abs(v) + 2), lo = ys.map((v) => -Math.abs(v) - 2);
  for (const op of THREE()) {
    const want = op(Float64Array.from(hi), Float64Array.from(lo), Float64Array.from(zs));
    for (let mask = 1; mask < 7; mask++) {
      const args = [hi, lo, zs].map((v, i) => (mask >> i & 1 ? v : Float64Array.from(v)));
      const got = op(...args);
      assert.ok(!Array.isArray(got), `mask ${mask}`);
      eq(flat(got), flat(want), `mask ${mask}`);
    }
  }
});

test("A8: a number[] broadcasts against (T, K) asset columns", async () => {
  await api.ready();
  const X = nd([T, 3], xs.flatMap((v, t) => [v, ys[t], zs[t]]));
  const want = a.RollingCorr(3)(X, Float64Array.from(zs)) as NdArray;
  const got = a.RollingCorr(3)(X, zs) as NdArray;
  assert.deepEqual(got.shape, [T, 3]);
  eq(got.data, want.data);
  const rev = a.RollingCorr(3)(zs, X) as NdArray;
  eq(rev.data, (a.RollingCorr(3)(Float64Array.from(zs), X) as NdArray).data);
});

test("A8: mixed lengths throw RangeError", async () => {
  await api.ready();
  assert.throws(() => a.RollingCorr(3)(xs.slice(0, 5), Float64Array.from(ys)), RangeError);
  assert.throws(() => a.RollingCorr(3)(Float64Array.from(xs), ys.slice(0, 5)), RangeError);
  assert.throws(() => a.RollingCorr(3)(nd([T, 2], new Float64Array(2 * T)), ys.slice(0, 5)), RangeError);
});

// ------------------------------------------------------------------ A9

const pipe = () => { const i = a.Input("a"); return a.Pipeline([i], [a.RollingMean(1)(i)]); };

test("A9: a null value in a Pipeline feed is NaN", async () => {
  await api.ready();
  eq(pipe()([1, null, 3]).values, [1, NaN, 3], "bare feed");
  eq(pipe()({ values: [1, null, 3] }).values, [1, NaN, 3], "values feed");
  const two = (() => {
    const x = a.Input("x"), y = a.Input("y");
    return a.Pipeline([x, y], [a.RollingMean(1)(x), a.RollingMean(1)(y)]);
  })();
  const [ox, oy] = two({ x: [1, null], y: { values: [null, 2] } });
  eq(ox.values, [1, NaN], "keyed feed");
  eq(oy.values, [NaN, 2], "keyed values feed");
});

test("A9: a null index keeps its coercion", async () => {
  await api.ready();
  assert.deepEqual(Array.from(pipe()({ values: [1, 2], index: [null, 1] }).index), [0, 1]);
});

test("A9: live push and pushWide read a null value as NaN", async () => {
  await api.ready();
  const lv = pipe().live();
  lv.push("a", 0, null);
  lv.push("a", 1, 2);
  lv.flush();
  eq(lv.result().values, [NaN, 2], "push");
  const w = (() => { const i = a.Input("a"); return a.Pipeline([i], [i]); })().live();
  w.pushWide("a", 0, [1, null, 3]);
  w.flush();
  eq(flat(w.result().values), [1, NaN, 3], "pushWide");
});

test("A9: null in an operator's number[] batch is a TypeError, as None is in Python", async () => {
  await api.ready();
  assert.throws(() => a.RollingMean(2)([1, null, 3]), (e: unknown) => e instanceof TypeError &&
    (e as Error).message === "RollingMean: event 1 is a null, expected a number");
  assert.throws(() => a.RollingCorr(2)([1, null], [1, 2]), TypeError);
  assert.throws(() => a.RollingCorr(2)([1, null], Float64Array.from([1, 2])), TypeError);
});

// ------------------------------------------------------------------ A10

test("A10: an Array of number events gives an Array of results", async () => {
  await api.ready();
  const got = a.Add()([[1, 2], [3, 4]]);
  assert.ok(Array.isArray(got));
  assert.deepEqual(got, [3, 7]);
  const pairs = xs.map((v, i) => [v, ys[i]]);
  const multi = a.RollingLinearRegression(4)(pairs);
  const want = a.RollingLinearRegression(4)(Float64Array.from(xs), Float64Array.from(ys)) as NdArray;
  assert.ok(Array.isArray(multi) && multi.length === T);
  multi.forEach((r: any, t: number) => {
    assert.ok(Array.isArray(r) && r.length === 4);
    eq(r, want.data.slice(t * 4, (t + 1) * 4), `event ${t}`);
  });
  const typed = a.Add()([Float64Array.from([1, 2]), Float64Array.from([3, 4])]);
  assert.deepEqual(typed, [3, 7]);
});

test("A10: an Array of Float64Array asset events into a 1-input op", async () => {
  await api.ready();
  // Python: RollingMean(2)([np.zeros(3), np.ones(3)]) is a list of two (3,) arrays.
  const got = a.RollingMean(2)([new Float64Array(3), Float64Array.from([1, 1, 1])]);
  assert.ok(Array.isArray(got) && got.length === 2);
  assert.ok(got.every((r: any) => r instanceof Float64Array && r.length === 3));
  eq(got[0], [NaN, NaN, NaN]);
  eq(got[1], [0.5, 0.5, 0.5]);
  // Equal to the stream of the same events.
  const rows = [Float64Array.from([1, 2, 3]), Float64Array.from([4, 5, 6]), Float64Array.from([7, 9, 8])];
  const listed = a.RollingMean(2)(rows), streamed = [...a.RollingMean(2)(rows.values())];
  listed.forEach((r: Float64Array, t: number) => eq(r, streamed[t], `event ${t}`));
  eq(listed[1], [2.5, 3.5, 4.5]);   // Python gives the same values
  eq(listed[2], [5.5, 7, 7]);
  // A multi-output op gives a (K, M) NdArray per event, as its stream does.
  const mm = a.RollingMinMax(2)(rows);
  const mmStream = [...a.RollingMinMax(2)(rows.values())];
  mm.forEach((r: NdArray, t: number) => { assert.deepEqual(r.shape, [3, 2]); eq(r.data, mmStream[t].data); });
  eq(mm[2].data, [4, 7, 5, 9, 6, 8]);
});

test("A10: an Array of NdArray asset events", async () => {
  await api.ready();
  const evs = [nd([3, 2], [1, 2, 3, 4, 5, 6]), nd([3, 2], [2, 2, 4, 4, 9, 6]), nd([3, 2], [3, 1, 2, 5, 5, 7])];
  const got = a.RollingCorr(2)(evs);
  const streamed = [...a.RollingCorr(2)(evs.values())];
  assert.ok(Array.isArray(got) && got.length === 3);
  got.forEach((r: any, t: number) => { assert.ok(r instanceof Float64Array && r.length === 3); eq(r, streamed[t]); });
  eq(got[2], [-1, -1, -1]);
  const one = a.RollingMean(2)([nd([2, 2], [1, 2, 3, 4]), nd([2, 2], [3, 4, 5, 6])]);
  assert.ok(Array.isArray(one));
  assert.deepEqual(one[1].shape, [2, 2]);
  eq(one[1].data, [2, 3, 4, 5]);
});

test("A10: asset events of different shapes throw RangeError", async () => {
  await api.ready();
  assert.throws(() => a.RollingMean(2)([new Float64Array(3), new Float64Array(2)]), RangeError);
  assert.throws(() => a.RollingCorr(2)([nd([3, 2], new Float64Array(6)), nd([2, 2], new Float64Array(4))]), RangeError);
});

test("A10: a number[] into a 1-input op still gives an Array", async () => {
  await api.ready();
  const got = a.RollingMean(2)([1, 2, 3]);
  assert.ok(Array.isArray(got));
  eq(got, [NaN, 1.5, 2.5]);
});

test("A10: all number[] inputs into a multi-input op give an Array, as Python lists give a list", async () => {
  await api.ready();
  assert.deepEqual(a.Add()([1, 2, 3], [10, 20, 30]), [11, 22, 33]);
  for (const op of TWO()) {
    const want = op(Float64Array.from(xs), Float64Array.from(ys));
    const got = op(xs, ys);
    assert.ok(Array.isArray(got) && got.length === T);
    if (want instanceof Float64Array) eq(got, want);
    else got.forEach((r: number[], t: number) => {      // nOut > 1: number[] per event, Python's tuple
      assert.ok(Array.isArray(r) && r.length === want.shape[1]);
      eq(r, want.data.slice(t * r.length, (t + 1) * r.length), `event ${t}`);
    });
    assert.ok(!Array.isArray(op(Float64Array.from(xs), Float64Array.from(ys))), "typed inputs keep an array");
  }
  const hi = xs.map((v) => Math.abs(v) + 2), lo = ys.map((v) => -Math.abs(v) - 2);
  for (const op of THREE()) {
    const got = op(hi, lo, zs);
    assert.ok(Array.isArray(got) && got.length === T);
    eq(got.flat(), flat(op(Float64Array.from(hi), Float64Array.from(lo), Float64Array.from(zs))));
  }
  assert.throws(() => a.Add()([1, 2, 3], [1, 2]), RangeError);
});
