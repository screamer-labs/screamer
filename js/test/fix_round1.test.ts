// The FC review fixes (task-fc-review.md F1-F9): each expected value or
// message is Python's on the same data.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ready, Input, Pipeline, RollingMean, VPIN, combineLatest, merge, resample, select,
} from "../src/index.ts";

const ev = (value: unknown, index: number) => ({ value, index });
const all = (it: Iterable<any>) => [...it];
const nanEq = (a: ArrayLike<number>, b: ArrayLike<number>) =>
  assert.deepEqual(Array.from(a, String), Array.from(b, String));

test("F1: a lazy Pipeline takes inputs of different widths, as batch and Python do", async () => {
  await ready();
  const x = Input("x"), y = Input("y");
  const p = new Pipeline([x, y], [combineLatest([x, y])]);
  const rows = all(p({ x: [ev([0, 1], 0), ev([2, 3], 2)].values(), y: [ev(5, 1), ev(6, 2)].values() } as any));
  assert.deepEqual(rows, [{ value: [0, 1, 5], index: 1 }, { value: [2, 3, 6], index: 2 }]);
  const batch = p({ x: { values: { data: Float64Array.from([0, 1, 2, 3]), shape: [2, 2] }, index: [0, 2] },
                    y: { values: [5, 6], index: [1, 2] } }) as any;
  nanEq(rows.flatMap((r) => r.value), batch.values.data);
  // A clock over wide values (Python: [(array([3., 4.]), 1)]).
  const bars = all(resample([ev([1, 2], 0), ev([3, 4], 1)].values(), { clock: true, driver: [ev(0, 1), ev(0, 3)].values() }));
  assert.equal(bars.length, 1);
  nanEq(bars[0].value, [3, 4]);
  assert.equal(bars[0].index, 1);
  // merge itself keeps its one-width rule.
  assert.throws(() => all(merge([[ev([0, 1], 0)].values(), [ev(5, 1)].values()])), RangeError);
});

test("F2: select gives the Pipeline's shapes and keeps duplicate columns", async () => {
  await ready();
  const one = Float64Array.from([0, 1, 2]);
  let r: any = select(one, [0]);
  assert.ok(r.values instanceof Float64Array); nanEq(r.values, [0, 1, 2]);
  assert.notEqual(r.values, one);
  r = select(one, [0, 0]);
  assert.deepEqual(r.values.shape, [3, 2]); nanEq(r.values.data, [0, 0, 1, 1, 2, 2]);
  assert.deepEqual(all(select([ev(1, 0)].values(), [0])), [{ value: 1, index: 0 }]);
  assert.deepEqual(all(select([ev(1, 0)].values(), [0, 0])), [{ value: [1, 1], index: 0 }]);
  assert.deepEqual(all(select([ev([1], 0)].values(), [0])), [{ value: 1, index: 0 }]);
});

test("F3: a NaN index is a RangeError with one text on every path", async () => {
  await ready();
  const x = Input("x");
  const p = new Pipeline([x], [RollingMean(2)(x)]);
  const engine = "index must be finite, got NaN; the streaming engine is int64-indexed";
  const range = (f: () => unknown, msg: string) => assert.throws(f, (e: any) => e instanceof RangeError && e.message === msg);
  range(() => p({ values: [1, 2], index: [0, NaN] }), engine);
  range(() => p.live().push("x", NaN, 1), engine);
  range(() => all(p([ev(1, 0), ev(2, NaN)].values() as any)),
    "Pipeline: the index of input 0 is NaN at event 1; an index must be a number");
  range(() => all(combineLatest([[ev(1, 0), ev(2, NaN)].values()])),
    "combineLatest: the index of input 0 is NaN at event 1; an index must be a number");
  range(() => all(merge([[ev(1, 0), ev(2, NaN)].values()])),
    "merge: the index of input 0 is NaN at event 1; an index must be a number");
  // Python's texts for a fractional and a decreasing index of a lazy run.
  assert.throws(() => all(p([ev(1, 0.5)].values() as any)), (e: any) => e instanceof TypeError &&
    e.message.startsWith("Pipeline: the index of event 0 of input 0: index must be integer-valued"));
  range(() => all(p([ev(1, 2), ev(2, 1)].values() as any)),
    "Pipeline: the index of input 0 decreases at event 1, from 2 to 1; a stream's index must be non-decreasing (equal indexes are allowed)");
});

test("F5, F6: origin through the shared rule; a threshold driver has no index", async () => {
  await ready();
  const x = Float64Array.from([0, 1, 2]);
  assert.throws(() => resample(x, { every: 2, origin: NaN }),
    (e: any) => e instanceof RangeError && e.message === "resample: origin must be an integer, got nan");
  assert.throws(() => resample(x, { every: 2, origin: 1e30 }),
    (e: any) => e instanceof RangeError && e.message === "resample: origin must be within the int64 range, got 1e+30");
  assert.throws(() => resample({ values: x, index: [0, 1, 2] }, { threshold: 1, driver: { values: [1, 1, 1], index: [0, 1, 2] } }),
    (e: any) => e instanceof RangeError &&
      e.message === "resample: a threshold driver is paired with the values by position; pass the driver without an index");
});

test("F9: a non-finite or huge volume never loops (VPIN, threshold resample)", async () => {
  await ready();
  const t0 = performance.now();
  for (const v of [Infinity, -Infinity, NaN]) {
    const op = VPIN(1, 3);
    op(1, 1); op(1, 1); op(1, 1);
    assert.ok(Number.isNaN(op(1, v) as number));
    nanEq([op(1, 1) as number], [0]);   // the state is untouched
  }
  nanEq(VPIN(1, 50)(Float64Array.from([1e308, 1, 1e308]), Float64Array.from([0, 1, 1e307])) as Float64Array,
        [1, 0.96, 0.8181818181818182]);
  nanEq([VPIN(1e-300, 50)(1e308, 0) as number], [1.0000000000000002]);
  assert.ok(performance.now() - t0 < 1000);
  // A threshold driver of +-inf is missing, as NaN (Python: [2, 4] at [0, 3]).
  const r = resample(Float64Array.from([0, 1, 2, 3, 4]), { threshold: 2, driver: Float64Array.from([1, Infinity, 1, -Infinity, 1]) });
  nanEq(r.values as Float64Array, [2, 4]); nanEq(r.index, [0, 3]);
});
