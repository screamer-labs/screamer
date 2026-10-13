// G27 and G32: merge of 1-D and wide streams (Python's merge), and the
// feed-length rule of every { values, index } entry point.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ready, merge, Input, Pipeline, Lag, combineLatest, forecastPairs, split, resample, delay,
  type NdArray, type Output,
} from "../src/index.ts";

const nd = (data: number[], shape: number[]): NdArray => ({ data: new Float64Array(data), shape });

// The merge by a stable sort on (index, input): rows, sources, index.
function refMerge(streams: { rows: number[][]; index: number[] }[]) {
  const ev: { row: number[]; src: number; idx: number }[] = [];
  streams.forEach((s, k) => s.rows.forEach((row, r) => ev.push({ row, src: k, idx: s.index[r] })));
  ev.sort((a, b) => a.idx - b.idx || a.src - b.src);
  return { rows: ev.map((e) => e.row), sources: ev.map((e) => e.src), index: ev.map((e) => e.idx) };
}

let seed = 27;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648) - 0.5;
const stream = (T: number, K: number) => {
  const rows = Array.from({ length: T }, () => Array.from({ length: K }, rand));
  let t = 0;
  const index = Array.from({ length: T }, () => (t += Math.floor((rand() + 0.5) * 3)));
  return { rows, index };
};

test("merge: wide streams merge row by row in index order, ties by input", async () => {
  await ready();
  for (const K of [1, 2, 5])
    for (const sizes of [[5, 3], [0, 4], [4, 0], [6, 6, 2], [0, 0]]) {
      const streams = sizes.map((T) => stream(T, K));
      const inputs = streams.map((s) => ({
        values: K === 1 ? s.rows.map((r) => r[0]) : nd(s.rows.flat(), [s.rows.length, K]),
        index: s.index,
      }));
      const got = merge(inputs as any);
      const want = refMerge(streams);
      assert.deepEqual(Array.from(got.sources), want.sources, `K=${K} ${sizes}`);
      assert.deepEqual(Array.from(got.index!), want.index);
      if (K === 1) {
        assert.ok(got.values instanceof Float64Array);
        assert.deepEqual(Array.from(got.values as Float64Array), want.rows.map((r) => r[0]));
      } else {
        assert.deepEqual((got.values as NdArray).shape, [want.rows.length, K]);
        assert.deepEqual(Array.from((got.values as NdArray).data), want.rows.flat());
      }
    }
});

test("merge: positional arrays merge by row number, with no index", async () => {
  await ready();
  const got = merge([nd([0, 1, 2, 3, 4, 5], [3, 2]), nd([10, 11], [1, 2])]);
  assert.equal(got.index, null);
  assert.deepEqual(Array.from(got.sources), [0, 1, 0, 0]);
  assert.deepEqual(Array.from((got.values as NdArray).data), [0, 1, 10, 11, 2, 3, 4, 5]);
  // T rows read as an NdArray
  const rows = merge([[[0, 1], [2, 3]], [[10, 11]]] as any);
  assert.deepEqual((rows.values as NdArray).shape, [3, 2]);
  const flat = merge([[1, 2, 3], new Float64Array([10, 20])]);
  assert.deepEqual(Array.from(flat.values as Float64Array), [1, 10, 2, 20, 3]);
});

test("merge: rows of rank two keep their shape", async () => {
  await ready();
  const a = nd(Array.from({ length: 12 }, (_, i) => i), [3, 2, 2]);
  const b = nd(Array.from({ length: 8 }, (_, i) => 100 + i), [2, 2, 2]);
  const got = merge([{ values: a, index: [0, 1, 5] }, { values: b, index: [1, 2] }]);
  assert.deepEqual((got.values as NdArray).shape, [5, 2, 2]);
  assert.deepEqual(Array.from(got.sources), [0, 0, 1, 1, 0]);
});

test("merge: inputs of other widths raise RangeError naming both", async () => {
  await ready();
  const width = (i: number, w: number, f: number, fw: number) =>
    `merge: input ${i} has rows of width ${w} but input ${f} has rows of width ${fw}; every input of a merge has the same width`;
  assert.throws(() => merge([{ values: nd([1, 2, 3, 4, 5, 6], [3, 2]), index: [0, 1, 2] },
                             { values: nd([1, 2, 3, 4, 5, 6], [2, 3]), index: [0, 1] }]),
    { name: "RangeError", message: width(1, 3, 0, 2) });
  assert.throws(() => merge([[1, 2, 3], nd([1, 2, 3, 4], [2, 2])]), { name: "RangeError", message: width(1, 2, 0, 1) });
  assert.throws(() => merge([[], nd([1, 2, 3, 4], [2, 2])]), { name: "RangeError", message: width(1, 2, 0, 1) });
  assert.throws(() => merge([nd(new Array(8).fill(1), [2, 2, 2]), nd([1, 2, 3, 4, 5, 6, 7, 8], [2, 4])]), {
    name: "RangeError",
    message: "merge: input 1 has rows of shape (4,) but input 0 has rows of shape (2, 2); every input of a merge has the same row shape",
  });
  // empty wide inputs of the same width merge
  const e = merge([{ values: nd([], [0, 2]), index: [] }, { values: nd([1, 2, 3, 4], [2, 2]), index: [1, 2] }]);
  assert.deepEqual((e.values as NdArray).shape, [2, 2]);
  assert.deepEqual(Array.from(e.sources), [1, 1]);
});

test("merge: the iterator form gives the batch rows, one event at a time", async () => {
  await ready();
  for (const K of [1, 3]) {
    const streams = [5, 0, 3].map((T) => stream(T, K));
    const batch = merge(streams.map((s) => ({
      values: K === 1 ? s.rows.map((r) => r[0]) : nd(s.rows.flat(), [s.rows.length, K]), index: s.index,
    })) as any);
    const its = streams.map((s) => s.rows.map((row, r) => ({ value: K === 1 ? row[0] : row, index: s.index[r] })));
    const events = [...merge(its.map((e) => e[Symbol.iterator]()) as any)];
    assert.deepEqual(events.map((e) => e.source), Array.from(batch.sources));
    assert.deepEqual(events.map((e) => e.index), Array.from(batch.index!));
    const flat = events.flatMap((e) => (typeof e.value === "number" ? [e.value] : Array.from(e.value as Float64Array)));
    assert.deepEqual(flat, Array.from(K === 1 ? batch.values as Float64Array : (batch.values as NdArray).data));
    if (K > 1) assert.ok(events.every((e) => e.value instanceof Float64Array && e.value.length === K));
    async function* gen<T>(xs: T[]) { for (const x of xs) yield x; }
    const asyncEvents = [];
    for await (const e of merge(its.map((e) => gen(e)) as any)) asyncEvents.push(e);
    assert.deepEqual(asyncEvents, events);
  }
});

test("merge: positional iterators merge rows by arrival", async () => {
  await ready();
  const events = [...merge([[[0, 1], [2, 3], [4, 5]][Symbol.iterator](), [[10, 11]][Symbol.iterator]()] as any)];
  assert.deepEqual(events.map((e) => e.source), [0, 1, 0, 0]);
  assert.ok(events.every((e) => e.index === null));
  assert.deepEqual(Array.from(events[1].value as Float64Array), [10, 11]);
  assert.deepEqual([...merge([[1, 2][Symbol.iterator](), [][Symbol.iterator]()] as any)].map((e) => e.value), [1, 2]);
});

test("merge: an iterator row of another width raises naming the event", async () => {
  await ready();
  assert.throws(() => [...merge([[{ value: [1, 2], index: 0 }, { value: 1, index: 2 }][Symbol.iterator](),
                                 [{ value: [3, 4], index: 1 }][Symbol.iterator]()] as any)],
    { name: "RangeError", message: "merge: event 1 of input 0 has width 1 but input 0 has rows of width 2; every input of a merge has the same width" });
});

test("merge: a wrong kind is a TypeError, never a C++ text", async () => {
  await ready();
  const bad: Array<() => unknown> = [
    () => merge([["a", "b"], [1, 2]] as any),
    () => merge([{ values: [1, 2], index: ["a", "b"] }, { values: [1, 2], index: [0, 1] }] as any),
    () => [...merge([["a"][Symbol.iterator](), [1][Symbol.iterator]()] as any)],
    () => [...merge([[{ value: 1, index: "x" }][Symbol.iterator](), [{ value: 1, index: 1 }][Symbol.iterator]()] as any)],
    () => merge(5 as any),
  ];
  for (const f of bad)
    assert.throws(f, (e: unknown) => e instanceof TypeError && !/bad_cast|std::/.test((e as Error).message));
  assert.throws(() => merge([{ values: [1], index: [0] }, [1]] as any), { name: "RangeError" });
  assert.throws(() => merge(Input("x") as any), /not supported as a Pipeline graph node/);
});

// ---- G32: one index value per row ------------------------------------------

test("feeds: an index of another length raises RangeError naming both lengths", async () => {
  await ready();
  const T = 10, V = Float64Array.from({ length: T }, (_, i) => i);
  const W = nd(Array.from({ length: 2 * T }, (_, i) => i), [T, 2]);
  const rowsOf = (n: number) => Array.from({ length: n }, (_, i) => i);
  const x = Input("x"), y = Input("y");
  const lag = new Pipeline([x], [Lag(1)(x)]);
  const two = new Pipeline([x, y], [combineLatest([x, y])]);
  const bars = new Pipeline([x], [resample(x, { every: 2 })]);
  const later = new Pipeline([x], [delay(x, 2)]);
  try {
    for (const n of [5, 15]) {
      const msg = (who: string) => ({ name: "RangeError", message: `${who} has ${T} rows of values but ${n} index values` });
      const ix = rowsOf(n);
      assert.throws(() => lag({ values: V, index: ix }), msg("Pipeline: input 0"));
      assert.throws(() => lag({ values: W, index: ix }), msg("Pipeline: input 0"));
      assert.throws(() => two({ x: { values: V, index: rowsOf(T) }, y: { values: V, index: ix } }), msg("Pipeline: input 1"));
      assert.throws(() => bars({ values: V, index: ix }), msg("Pipeline: input 0"));
      assert.throws(() => later({ values: V, index: ix }), msg("Pipeline: input 0"));
      assert.throws(() => merge([{ values: V, index: ix }, { values: V, index: rowsOf(T) }]), msg("merge: input 0"));
      assert.throws(() => combineLatest([{ values: V, index: ix }, { values: V, index: rowsOf(T) }]), msg("combineLatest: input 0"));
      assert.throws(() => split(V, new Array(T).fill(0), { index: ix }), msg("split: the stream"));
      assert.throws(() => forecastPairs({ values: V, index: ix }, { values: V, index: rowsOf(T) }, { duration: 2 }),
        msg("forecastPairs: input 0"));
      assert.throws(() => forecastPairs({ values: V, index: rowsOf(T) }, { values: V, index: ix }, { duration: 2 }),
        msg("forecastPairs: input 1"));
    }
    // A flat array is one value per row, never rows of the index's length.
    assert.throws(() => lag({ values: Float64Array.from({ length: 2 * T }), index: rowsOf(T) }),
      { name: "RangeError", message: `Pipeline: input 0 has ${2 * T} rows of values but ${T} index values` });
    const ok = later({ values: W, index: rowsOf(T) }) as Output;
    assert.deepEqual((ok.values as NdArray).shape, [T, 2]);
  } finally {
    for (const p of [lag, two, bars, later]) p.dispose();
  }
});

test("streams: a malformed index raises a clean error at every entry point, never C++ text", async () => {
  await ready();
  const V = [1, 2, 3], I = [0, 1, 2];
  const bads: unknown[] = [{ data: new Float64Array([0, 1, 2]), shape: [3, 1] }, ["a", "b", "c"], [[0], [1], [2]], 5, [0, null, 2], [0n, 1n, 2n]];
  const x = Input("x"), y = Input("y");
  const lag = new Pipeline([x], [Lag(1)(x)]);
  const two = new Pipeline([x, y], [combineLatest([x, y])]);
  const entry: Record<string, (i: unknown) => unknown> = {
    merge: (i) => merge([{ values: V, index: I }, { values: V, index: i as any }]),
    "merge iterator": (i) => [...merge([[{ value: 1, index: i }][Symbol.iterator](), [{ value: 1, index: 0 }][Symbol.iterator]()] as any)],
    combineLatest: (i) => combineLatest([{ values: V, index: I }, { values: V, index: i as any }]),
    Pipeline: (i) => lag({ values: V, index: i as any }),
    "Pipeline two inputs": (i) => two({ x: { values: V, index: I }, y: { values: V, index: i as any } }),
    split: (i) => split(V, [0, 0, 0], { index: i as any }),
    forecastPairs: (i) => forecastPairs({ values: V, index: i as any }, { values: V, index: I }, { duration: 1 }),
  };
  try {
    for (const [name, f] of Object.entries(entry))
      for (const b of bads) {
        if (name === "merge iterator" && typeof b === "number") continue;   // a valid event index
        assert.throws(() => f(b), (e: unknown) => (e instanceof TypeError || e instanceof RangeError) &&
          !/std::|nanobind|^vector$/.test((e as Error).message), `${name} ${JSON.stringify(b, (_k, v) => typeof v === "bigint" ? String(v) : v)}`);
      }
  } finally {
    lag.dispose(); two.dispose();
  }
});
