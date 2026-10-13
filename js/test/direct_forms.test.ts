// The array and iterator forms of the stream functions, and the lazy
// Pipeline (tracker B9). The expected values and messages are Python's:
// Resample, Select, Filter, Dropna, Delay and Pipeline on the same data.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ready, current, Input, Pipeline, RollingMean, resample, select, filter, dropna, delay,
} from "../src/index.ts";

const ev = (value: unknown, index: number) => ({ value, index });
const nanEq = (a: ArrayLike<number>, b: ArrayLike<number>) =>
  assert.deepEqual(Array.from(a, String), Array.from(b, String));
const all = (it: Iterable<any>) => [...it];
const x5 = Float64Array.from([0, 1, 2, 3, 4]);
const wide = { data: Float64Array.from([0, 1, 2, 3, 4, 5]), shape: [3, 2] };

test("lazy Pipeline: rows as a batch call gives them", async () => {
  await ready();
  const x = Input("x"), y = Input("y");
  const p = new Pipeline([x], [RollingMean(2)(x)]);
  const rows = all(p([ev(1, 0), ev(2, 1), ev(3, 5)].values() as any));
  assert.deepEqual(rows.map((r) => r.index), [0, 1, 5]);
  nanEq(rows.map((r) => r.value), [NaN, 1.5, 2.5]);
  const batch = p({ values: [1, 2, 3], index: [0, 1, 5] }) as any;
  nanEq(rows.map((r) => r.value), batch.values);
  // Two aligned outputs: { values, index } rows, as Python's (v0, v1, index).
  const p2 = new Pipeline([x, y], [RollingMean(2)(x), RollingMean(2)(y)]);
  const two = all(p2({ x: [ev(1, 0), ev(2, 2)].values(), y: [ev(5, 1), ev(6, 2)].values() } as any));
  assert.deepEqual(two.map((r) => r.index), [1, 2]);
  nanEq(two.flatMap((r) => r.values), [NaN, NaN, 1.5, 5.5]);
  // Python's errors.
  assert.throws(() => p([1, 2].values() as any).next(),
    (e: any) => e instanceof TypeError && e.message === "Pipeline: event 0 of input 0 is a number, expected a { value, index } event");
  const p3 = new Pipeline([x, y], [RollingMean(2)(x), RollingMean(2)(y)], { alignOutputs: false });
  assert.throws(() => p3({ x: [ev(1, 0)].values(), y: [ev(5, 1)].values() } as any), TypeError);
  assert.throws(() => p2({ x: [ev(1, 0)].values(), y: [1] } as any), (e: any) => e instanceof TypeError && /all lazy iterators or all concrete/.test(e.message));
});

test("lazy Pipeline: nothing is read before next(); return() closes the sources; runs do not mix", async () => {
  await ready();
  const x = Input("x");
  const p = new Pipeline([x], [RollingMean(2)(x)]);
  let pulled = 0, closed = false;
  function* src() { try { for (let i = 0; i < 100; i++) { pulled++; yield ev(i, i); } } finally { closed = true; } }
  const run = p(src() as any) as Generator<any>;
  assert.equal(pulled, 0);
  const a = run.next().value, b = p([ev(10, 0), ev(20, 1)].values() as any);
  nanEq([a.value], [NaN]);
  nanEq(all(b).map((r) => r.value), [NaN, 15]);   // its own state
  nanEq([run.next().value.value], [0.5]);
  run.return(undefined);
  assert.ok(closed);
  // Disposing the Pipeline ends an open run.
  const r2 = p([ev(1, 0), ev(2, 1)].values() as any) as Generator<any>;
  r2.next();
  p.dispose();
  assert.throws(() => r2.next(), TypeError);
});

test("lazy Pipeline: dropped runs leave nothing on the WASM heap", async () => {
  await ready();
  const gc = (globalThis as any).gc as () => void;
  const M = current();
  const x = Input("x");
  const p = new Pipeline([x], [RollingMean(2)(x)]);
  const once = () => { const r = p([ev(1, 0), ev(2, 1)].values() as any) as Generator<any>; r.next(); };
  const settle = async () => { for (let i = 0; i < 6; i++) { gc(); await new Promise((r) => setTimeout(r, 0)); } };
  for (let i = 0; i < 200; i++) once();
  await settle();
  const h0 = M.heapInUse();
  for (let i = 0; i < 2000; i++) once();
  await settle();
  assert.ok(M.heapInUse() - h0 < 4096, `WASM heap grew by ${M.heapInUse() - h0}`);
});

test("resample: the array and iterator forms give Python's bars", async () => {
  await ready();
  let r: any = resample(x5, { every: 2 });
  nanEq(r.values, [1, 3, 4]); nanEq(r.index, [0, 2, 4]);
  r = resample(wide, { every: 2 });
  assert.deepEqual(r.values.shape, [2, 2]); nanEq(r.values.data, [2, 3, 4, 5]);
  r = resample(wide, { every: 2, agg: "ohlc" });
  assert.deepEqual(r.values.shape, [2, 2, 4]);
  nanEq(r.values.data, [0, 2, 0, 2, 1, 3, 1, 3, 4, 4, 4, 4, 5, 5, 5, 5]);
  r = resample({ values: x5, index: [0, 1, 5, 6, 9] }, { count: 2 });
  nanEq(r.values, [1, 3, 4]); nanEq(r.index, [0, 5, 9]);
  r = resample(x5, { threshold: 2, driver: new Float64Array(5).fill(1) });
  nanEq(r.values, [1, 3, 4]); nanEq(r.index, [0, 2, 4]);
  r = resample(x5, { clock: true, driver: Float64Array.from([1, 3]) });
  nanEq(r.values, [0, 1]); nanEq(r.index, [0, 1]);
  const lazy = all(resample([ev(1, 0), ev(2, 1), ev(3, 5)].values(), { every: 2 }));
  assert.deepEqual(lazy, [{ value: 2, index: 0 }, { value: 3, index: 4 }]);
  // Asset events keep their shape, as Python's lazy resample does.
  const assets = all(resample([ev([1, 2], 0), ev([3, 4], 1), ev([5, 6], 2)].values(), { every: 2, agg: "ohlc" }));
  assert.deepEqual(assets[0].value.shape, [2, 4]);
  nanEq(assets[0].value.data, [1, 3, 1, 3, 2, 4, 2, 4]);
  assert.throws(() => resample([1, 2].values(), { every: 2 }).next(),
    (e: any) => e instanceof TypeError && e.message === "resample: event 0 of input 0 is a number, expected a { value, index } event");
});

test("select: the array and iterator forms", async () => {
  await ready();
  let r: any = select(wide, 1);
  nanEq(r.values, [1, 3, 5]); assert.equal(r.index, null);
  r = select(wide, [1]);
  nanEq(r.values, [1, 3, 5]);
  const one = Float64Array.from([0, 1, 2]);
  r = select(one, 0);
  nanEq(r.values, [0, 1, 2]); assert.notEqual(r.values, one);   // a new array
  r = select({ values: one, index: [3, 4, 5] }, 0);
  nanEq(r.index, [3, 4, 5]);
  assert.throws(() => select(wide, 2), (e: any) => e instanceof RangeError && e.message === "select: column 2 out of range for width 2");
  assert.deepEqual(all(select([ev([1, 2], 0), ev([3, 4], 1)].values(), 1)), [{ value: 2, index: 0 }, { value: 4, index: 1 }]);
});

test("filter, dropna, delay: the array and iterator forms", async () => {
  await ready();
  let r: any = filter(Float64Array.from([0, 1, 2, 3]), Float64Array.from([1, 0, 1, NaN]));
  nanEq(r.values, [0, 2]); nanEq(r.index, [0, 2]);
  r = filter({ values: [0, 1, 2, 3], index: [0, 1, 2, 3] }, { values: [1, 0, 1, 1], index: [0, 1, 2, 3] });
  nanEq(r.values, [0, 2, 3]); nanEq(r.index, [0, 2, 3]);
  assert.deepEqual(all(filter([ev(1, 0), ev(2, 1)].values(), [ev(1, 0), ev(0, 1)].values())), [{ value: 1, index: 0 }]);
  r = dropna(Float64Array.from([1, NaN, 3]));
  nanEq(r.values, [1, 3]); assert.equal(r.index, null);
  r = dropna([{ values: [1, NaN, 3] }, { values: [NaN, 2, 4] }]);
  nanEq(r.values, [1, 2, 3, 4]);
  assert.throws(() => dropna([{ values: [1] }, { values: [2], index: [0] }]), RangeError);
  assert.deepEqual(all(dropna([ev(1, 0), ev(NaN, 1)].values())), [{ value: 1, index: 0 }]);
  r = delay({ values: [0, 1, 2], index: [0, 1, 5] }, 2);
  nanEq(r.values, [0, 1, 2]); nanEq(r.index, [2, 3, 7]);
  assert.throws(() => delay(Float64Array.from([0, 1, 2]), 2), TypeError);
});
