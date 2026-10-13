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

test("combinators: combineLatest joins a 4-column bar node and a signal", async () => {
  await ready();
  const x = Input("x");
  const y = Input("y");
  const p = new Pipeline([x, y], [combineLatest([resample(x, { count: 2, agg: "ohlc" }), y])]);
  const xs = Float64Array.from({ length: 20 }, (_, i) => i);
  const { values } = p({ x: xs, y: xs }) as { values: { data: Float64Array; shape: number[] } };
  assert.deepEqual(values.shape, [20, 5]);
  assertNanEq(values.data.subarray(2 * 5, 2 * 5 + 4), [2, 3, 2, 3]);
  p.dispose();
});

// ---- combineLatest direct forms (arrays, [values, index] pairs, iterators) ----
// They mirror tests/test_combine_latest_multicolumn.py; the join is the C++
// one Python's combine_latest calls.

type Nd = { data: Float64Array; shape: number[] };
const T = 12;
const bars: Nd = { data: Float64Array.from({ length: T * 4 }, (_, i) => i), shape: [T, 4] };
const sig = Float64Array.from({ length: T }, (_, i) => i * 10);
const panel: Nd = { data: Float64Array.from({ length: T * 3 }, (_, i) => i), shape: [T, 3] };
const rowOf = (a: Nd, t: number) => a.data.subarray(t * a.shape[1], (t + 1) * a.shape[1]);
function* gen<X>(xs: Iterable<X>) { yield* xs; }
async function* agen<X>(xs: Iterable<X>) { for (const x of xs) yield x; }
const rowsOf = (a: Nd) => Array.from({ length: a.shape[0] }, (_, t) => rowOf(a, t));
const columnStack = (a: Nd, b: Float64Array) =>
  Array.from({ length: a.shape[0] }, (_, t) => [...rowOf(a, t), b[t]]).flat();
async function collect<X>(it: AsyncIterable<X>): Promise<X[]> {
  const out: X[] = [];
  for await (const x of it) out.push(x);
  return out;
}
const flatRows = (rows: Array<[any, any]>) =>
  rows.flatMap(([r]) => Array.from(r instanceof Float64Array ? r : r.data ?? r) as number[]);

test("combineLatest: arrays concatenate fields, paired by position", async () => {
  await ready();
  const [values, index] = combineLatest([bars, sig]) as [Nd, null];
  assert.equal(index, null);
  assert.deepEqual(values.shape, [T, 5]);
  assertNanEq(values.data, columnStack(bars, sig));
  const [one] = combineLatest([sig]) as [Nd, null];
  assert.deepEqual(one.shape, [T, 1]);
});

test("combineLatest: [values, index] pairs join as of", async () => {
  await ready();
  const ia = Float64Array.from({ length: T }, (_, i) => i * 2);
  const [v, i] = combineLatest([[bars, ia], [sig.slice(0, 3), [1, 5, 9]]]) as [Nd, Float64Array];
  assert.equal(v.shape[1], 5);
  const row = Array.from(i).indexOf(10);
  assertNanEq(v.data.subarray(row * 5, row * 5 + 5), [...rowOf(bars, 5), sig[2]]);
});

test("combineLatest: on_any gives NaN until an input is seen", async () => {
  await ready();
  const at = (k: number) => Float64Array.from({ length: T }, (_, i) => i + k);
  const [v] = combineLatest([[bars, at(5)], [sig, at(0)]], { emit: "on_any" }) as [Nd, Float64Array];
  assert.deepEqual(v.shape, [T + 5, 5]);
  assert.ok(Array.from(v.data.subarray(0, 4)).every(Number.isNaN));
  assert.equal(v.data[4], sig[0]);
});

test('combineLatest: layout ["TA", "T"] broadcasts assets, giving [T, K, 2]', async () => {
  await ready();
  const [v, i] = combineLatest([panel, sig], { layout: ["TA", "T"] }) as [Nd, null];
  assert.equal(i, null);
  assert.deepEqual(v.shape, [T, 3, 2]);
  for (let t = 0; t < T; t++)
    for (let a = 0; a < 3; a++) {
      assert.equal(v.data[(t * 3 + a) * 2], panel.data[t * 3 + a]);
      assert.equal(v.data[(t * 3 + a) * 2 + 1], sig[t]);
    }
  // indexed, the same join
  const ix = Float64Array.from({ length: T }, (_, t) => t);
  const [w, j] = combineLatest([[panel, ix], [sig, ix]], { layout: ["TA", "T"] }) as [Nd, Float64Array];
  assert.deepEqual(w.shape, [T, 3, 2]);
  assertNanEq(w.data, v.data);
  assertNanEq(j, ix);
});

test("combineLatest: one layout string applies to every input; 3-D inputs are (T, A, F)", async () => {
  await ready();
  const [v] = combineLatest([panel, panel], { layout: "TA" }) as [Nd, null];
  assert.deepEqual(v.shape, [T, 3, 2]);
  const cube: Nd = { data: new Float64Array(T * 3 * 4), shape: [T, 3, 4] };
  const [c] = combineLatest([cube, sig]) as [Nd, null];
  assert.deepEqual(c.shape, [T, 3, 5]);
});

test("combineLatest: layout errors", async () => {
  await ready();
  assert.throws(() => combineLatest([bars, sig], { layout: ["TA"] }),
    (e: any) => e instanceof RangeError && /one layout per input \(2\)/.test(e.message));
  assert.throws(() => combineLatest([panel, sig], { layout: ["TX", "T"] }),
    (e: any) => e instanceof RangeError && /unknown axis letter 'X'/.test(e.message));
  assert.throws(() => combineLatest([panel, { data: new Float64Array(T * 2), shape: [T, 2] }], { layout: "TA" }),
    (e: any) => e instanceof RangeError && /do not broadcast/.test(e.message));
  assert.throws(() => combineLatest([panel, sig], { layout: [3 as any, "T"] }), TypeError);
  assert.throws(() => combineLatest([Input("a"), Input("b")], { layout: "TA" }),
    (e: any) => e instanceof RangeError && /applies to arrays and iterators/.test(e.message));
});

test("combineLatest: options, kinds and mixing", async () => {
  await ready();
  assert.throws(() => combineLatest([sig, sig], { emit: "sometimes" as any }),
    (e: any) => e instanceof RangeError && /emit must be "when_all" or "on_any"/.test(e.message));
  assert.throws(() => combineLatest([sig, sig], { emits: "on_any" } as any), TypeError);
  assert.throws(() => combineLatest([sig, sig], { maxPending: 5 }), TypeError);
  assert.throws(() => combineLatest([sig, sig.slice(0, 5)]),
    (e: any) => e instanceof RangeError && /lengths must match/.test(e.message));
  assert.throws(() => combineLatest([sig, [sig, sig]]),
    (e: any) => e instanceof RangeError && /cannot align positional and indexed/.test(e.message));
  assert.throws(() => combineLatest([[1, "x" as any], [1, 2]]), TypeError);
  assert.throws(() => combineLatest([[sig.slice(0, 2), [true, false] as any], [sig.slice(0, 2), [0, 1]]]), TypeError);
  assert.throws(() => combineLatest([gen([1]), sig]), TypeError);
  assert.throws(() => combineLatest([gen([1]), agen([1])]), TypeError);
  assert.throws(() => combineLatest([]), RangeError);
  const [v] = combineLatest([[1, null, 3], [4, 5, 6]]) as [Nd, null];
  assert.ok(Number.isNaN(v.data[2]));
  // int and float indexes mix
  const [w, i] = combineLatest([[sig.slice(0, 2), [0, 1]], [sig.slice(0, 2), [0.5, 1.5]]]) as [Nd, Float64Array];
  assertNanEq(i, [0.5, 1, 1.5]);
  assert.deepEqual(w.shape, [3, 2]);
});

test("combineLatest: positional iterators are paired by position", async () => {
  await ready();
  const out = [...combineLatest([gen(rowsOf(bars)), gen(sig)]) as Iterable<[Float64Array, null]>];
  assert.equal(out.length, T);
  assert.ok(out.every(([r, i]) => r instanceof Float64Array && i === null));
  assertNanEq(flatRows(out), columnStack(bars, sig));
  // one field per source: number[] rows
  const narrow = [...combineLatest([gen([1, null]), gen([3, 4])]) as Iterable<[number[], null]>];
  assert.deepEqual(narrow[0], [[1, 3], null]);
  assert.ok(Number.isNaN(narrow[1][0][0]));
  assert.throws(() => [...combineLatest([gen([1, 2, 3]), gen([1, 2])]) as Iterable<unknown>],
    (e: any) => e instanceof RangeError && /stay in step/.test(e.message));
  assert.throws(() => [...combineLatest([gen([1, "x"]), gen([1, 2])]) as Iterable<unknown>], TypeError);
  assert.deepEqual([...combineLatest([gen([]), gen([])]) as Iterable<unknown>], []);
});

test("combineLatest: indexed iterators match the batch join", async () => {
  await ready();
  const ia = Array.from({ length: T }, (_, i) => i * 2), ib = [1, 5, 9];
  const [batch, bidx] = combineLatest([[bars, ia], [sig.slice(0, 3), ib]]) as [Nd, Float64Array];
  const lazy = [...combineLatest([
    gen(rowsOf(bars).map((r, t) => [r, ia[t]])), gen(Array.from(sig.slice(0, 3)).map((x, t) => [x, ib[t]])),
  ]) as Iterable<[Float64Array, number]>];
  assertNanEq(flatRows(lazy), batch.data);
  assert.deepEqual(lazy.map(([, i]) => i), Array.from(bidx));
  // scalar events: number[] rows, or a number for one input
  const two = [...combineLatest([gen([[1, 0], [2, 2]]), gen([[10, 1]])]) as Iterable<[number[], number]>];
  assert.deepEqual(two, [[[1, 10], 1], [[2, 10], 2]]);
  const one = [...combineLatest([gen([[1, 0], [2, 2]])]) as Iterable<[number, number]>];
  assert.deepEqual(one, [[1, 0], [2, 2]]);
  // on_any with an empty source matches the batch join
  const [e, ei] = combineLatest([[sig.slice(0, 3), [0, 1, 2]], [new Float64Array(0), []]], { emit: "on_any" }) as [Nd, Float64Array];
  const le = [...combineLatest([gen([[0, 0], [10, 1], [20, 2]]), gen([])], { emit: "on_any" }) as Iterable<[number[], number]>];
  assertNanEq(le.flatMap(([r]) => r), e.data);
  assert.deepEqual(le.map(([, i]) => i), Array.from(ei));
  assert.throws(() => [...combineLatest([gen([[1, 0]]), gen([2])]) as Iterable<unknown>],
    (e: any) => e instanceof RangeError && /cannot mix positional/.test(e.message));
  assert.throws(() => [...combineLatest([gen([[1, 0.5]]), gen([[2, 1]])]) as Iterable<unknown>], TypeError);
});

test('combineLatest: iterator layout ["TA", "T"] gives [K, 2] rows, positional and indexed', async () => {
  await ready();
  const [batch] = combineLatest([panel, sig], { layout: ["TA", "T"] }) as [Nd, null];
  const pos = [...combineLatest([gen(rowsOf(panel)), gen(sig)], { layout: ["TA", "T"] }) as Iterable<[Nd, null]>];
  assert.deepEqual(pos[0][0].shape, [3, 2]);
  assertNanEq(flatRows(pos), batch.data);
  const idx = [...combineLatest([gen(rowsOf(panel).map((r, t) => [r, t])), gen(Array.from(sig).map((x, t) => [x, t]))],
    { layout: ["TA", "T"] }) as Iterable<[Nd, number]>];
  assertNanEq(flatRows(idx), batch.data);
  assert.deepEqual(idx.map(([, i]) => i), Array.from({ length: T }, (_, t) => t));
});

test("combineLatest: a reordering layout reads iterator events as the batch join does", async () => {
  await ready();
  // (T, F, A) data read as "TFA": each event is (F, A)
  const cube: Nd = { data: Float64Array.from({ length: T * 6 }, (_, i) => i), shape: [T, 2, 3] };
  const [batch] = combineLatest([cube, sig], { layout: ["TFA", "T"] }) as [Nd, null];
  assert.deepEqual(batch.shape, [T, 3, 3]);
  const events = Array.from({ length: T }, (_, t) => ({ data: cube.data.slice(t * 6, t * 6 + 6), shape: [2, 3] }));
  const lazy = [...combineLatest([gen(events), gen(sig)], { layout: ["TFA", "T"] }) as Iterable<[Nd, null]>];
  assertNanEq(flatRows(lazy), batch.data);
});

test("combineLatest: async iterators, positional and indexed", async () => {
  await ready();
  const pos = await collect(combineLatest([agen(rowsOf(bars)), agen(sig)]) as AsyncIterable<[Float64Array, null]>);
  assertNanEq(flatRows(pos), columnStack(bars, sig));
  const [batch] = combineLatest([panel, sig], { layout: ["TA", "T"] }) as [Nd, null];
  const idx = await collect(combineLatest(
    [agen(rowsOf(panel).map((r, t) => [r, t])), agen(Array.from(sig).map((x, t) => [x, t]))],
    { layout: ["TA", "T"] }) as AsyncIterable<[Nd, number]>);
  assert.deepEqual(idx[0][0].shape, [3, 2]);
  assertNanEq(flatRows(idx), batch.data);
  await assert.rejects(collect(combineLatest([agen([1, 2]), agen([1])]) as AsyncIterable<unknown>),
    (e: any) => e instanceof RangeError && /stay in step/.test(e.message));
});

test("combineLatest: a stream reads nothing before its first next(), and closes its sources", async () => {
  await ready();
  let pulled = 0, closed = 0;
  const src = () => ({
    [Symbol.iterator]() { return this; },
    next() { pulled++; return { done: false, value: [pulled, pulled] }; },
    return() { closed++; return { done: true, value: undefined }; },
  });
  const it = combineLatest([src(), src()]) as Iterator<unknown>;
  assert.equal(pulled, 0);
  it.next();
  it.return!(undefined);
  assert.equal(closed, 2);
});
