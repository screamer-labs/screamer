import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ready,
  Input,
  Pipeline,
  Sub,
  combineLatest,
  dropna,
  resample,
  split,
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

// ---- combineLatest direct forms (arrays, {values, index} streams, iterators) ----
// They mirror tests/test_combine_latest_multicolumn.py; the join is the C++
// one Python's combine_latest calls. js/test/combine_latest_parity.test.ts
// replays a Python fixture.

type Nd = { data: Float64Array; shape: number[] };
type Batch = { values: Nd; index: Float64Array | null };
type Row = { value: any; index: number | null };
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
const flatRows = (rows: Row[]) =>
  rows.flatMap(({ value: r }) => Array.from(r instanceof Float64Array ? r : r.data ?? r) as number[]);
const cl = (inputs: any[], opts?: any) => combineLatest(inputs, opts) as any;
const ev = (value: unknown, index: number) => ({ value, index });

test("combineLatest: arrays concatenate fields, paired by position", async () => {
  await ready();
  const { values, index } = cl([bars, sig]) as Batch;
  assert.equal(index, null);
  assert.deepEqual(values.shape, [T, 5]);
  assertNanEq(values.data, columnStack(bars, sig));
  const { values: one } = cl([sig]) as Batch;
  assert.deepEqual(one.shape, [T, 1]);
});

test("combineLatest: {values, index} streams join as of", async () => {
  await ready();
  const ia = Float64Array.from({ length: T }, (_, i) => i * 2);
  const { values: v, index: i } = cl([{ values: bars, index: ia }, { values: sig.slice(0, 3), index: [1, 5, 9] }]) as Batch;
  assert.equal(v.shape[1], 5);
  const row = Array.from(i!).indexOf(10);
  assertNanEq(v.data.subarray(row * 5, row * 5 + 5), [...rowOf(bars, 5), sig[2]]);
  // {values} without an index is positional
  const { index: none } = cl([{ values: sig }, { values: sig, index: null }]) as Batch;
  assert.equal(none, null);
});

test("combineLatest: a plain Array is always values, never a [values, index] pair", async () => {
  await ready();
  // Two typed arrays in an Array are two rows of one input, as a Python list
  // of two arrays is: here two rows of 3 values against two values.
  const { values, index } = cl([[new Float64Array([1, 2, 3]), new Float64Array([4, 5, 6])], [7, 8]]) as Batch;
  assert.equal(index, null);
  assert.deepEqual(values.shape, [2, 4]);
  assertNanEq(values.data, [1, 2, 3, 7, 4, 5, 6, 8]);
  // Two-value rows are rows.
  const { values: w } = cl([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) as Batch;
  assertNanEq(w.data, [1, 2, 5, 6, 3, 4, 7, 8]);
});

test("combineLatest: on_any gives NaN until an input is seen", async () => {
  await ready();
  const at = (k: number) => Float64Array.from({ length: T }, (_, i) => i + k);
  const { values: v } = cl([{ values: bars, index: at(5) }, { values: sig, index: at(0) }], { emit: "on_any" }) as Batch;
  assert.deepEqual(v.shape, [T + 5, 5]);
  assert.ok(Array.from(v.data.subarray(0, 4)).every(Number.isNaN));
  assert.equal(v.data[4], sig[0]);
});

test('combineLatest: layout ["TA", "T"] broadcasts assets, giving [T, K, 2]', async () => {
  await ready();
  const { values: v, index: i } = cl([panel, sig], { layout: ["TA", "T"] }) as Batch;
  assert.equal(i, null);
  assert.deepEqual(v.shape, [T, 3, 2]);
  for (let t = 0; t < T; t++)
    for (let a = 0; a < 3; a++) {
      assert.equal(v.data[(t * 3 + a) * 2], panel.data[t * 3 + a]);
      assert.equal(v.data[(t * 3 + a) * 2 + 1], sig[t]);
    }
  // indexed, the same join
  const ix = Float64Array.from({ length: T }, (_, t) => t);
  const { values: w, index: j } = cl([{ values: panel, index: ix }, { values: sig, index: ix }], { layout: ["TA", "T"] }) as Batch;
  assert.deepEqual(w.shape, [T, 3, 2]);
  assertNanEq(w.data, v.data);
  assertNanEq(j!, ix);
});

test("combineLatest: a positional batch counts rows along the T axis its layout names", async () => {
  await ready();
  // ["AT", "T"]: a (K, T) array is K assets over T rows
  const at: Nd = { data: Float64Array.from({ length: 3 * T }, (_, i) => i), shape: [3, T] };
  const { values: a } = cl([at, sig], { layout: ["AT", "T"] }) as Batch;
  assert.deepEqual(a.shape, [T, 3, 2]);
  for (let t = 0; t < T; t++) {
    assert.equal(a.data[(t * 3 + 1) * 2], at.data[1 * T + t]);
    assert.equal(a.data[(t * 3 + 1) * 2 + 1], sig[t]);
  }
  // ["TFA", "T"]: a (T, F, A) array
  const tfa: Nd = { data: Float64Array.from({ length: T * 6 }, (_, i) => i), shape: [T, 2, 3] };
  const { values: b } = cl([tfa, sig], { layout: ["TFA", "T"] }) as Batch;
  assert.deepEqual(b.shape, [T, 3, 3]);
  assert.deepEqual(Array.from(b.data.subarray(0, 9)), [0, 3, 0, 1, 4, 0, 2, 5, 0]);
});

test("combineLatest: one layout string applies to every input; 3-D inputs are (T, A, F)", async () => {
  await ready();
  const { values: v } = cl([panel, panel], { layout: "TA" }) as Batch;
  assert.deepEqual(v.shape, [T, 3, 2]);
  const cube: Nd = { data: new Float64Array(T * 3 * 4), shape: [T, 3, 4] };
  const { values: c } = cl([cube, sig]) as Batch;
  assert.deepEqual(c.shape, [T, 3, 5]);
});

test("combineLatest: layout errors", async () => {
  await ready();
  assert.throws(() => cl([bars, sig], { layout: ["TA"] }),
    (e: any) => e instanceof RangeError && /one layout per input \(2\)/.test(e.message));
  assert.throws(() => cl([panel, sig], { layout: ["TX", "T"] }),
    (e: any) => e instanceof RangeError && /unknown axis letter 'X'/.test(e.message));
  assert.throws(() => cl([panel, { data: new Float64Array(T * 2), shape: [T, 2] }], { layout: "TA" }),
    (e: any) => e instanceof RangeError && /do not broadcast/.test(e.message));
  assert.throws(() => cl([panel, sig], { layout: [3, "T"] }), TypeError);
  assert.throws(() => combineLatest([Input("a"), Input("b")], { layout: "TA" }),
    (e: any) => e instanceof RangeError && /applies to arrays and iterators/.test(e.message));
});

test("combineLatest: options, kinds and mixing", async () => {
  await ready();
  assert.throws(() => cl([sig, sig], { emit: "sometimes" }),
    (e: any) => e instanceof RangeError && /emit must be "when_all" or "on_any"/.test(e.message));
  assert.throws(() => cl([sig, sig], { emits: "on_any" }), TypeError);
  assert.throws(() => cl([sig, sig], { maxPending: 5 }), TypeError);
  assert.throws(() => cl([sig, sig.slice(0, 5)]),
    (e: any) => e instanceof RangeError && /lengths must match/.test(e.message));
  assert.throws(() => cl([sig, { values: sig, index: sig }]),
    (e: any) => e instanceof RangeError && /cannot align positional and indexed/.test(e.message));
  assert.throws(() => cl([{ values: sig, index: sig.slice(0, 4) }, { values: sig, index: sig }]),
    (e: any) => e instanceof RangeError && e.message === "combineLatest: input 0 has 12 rows of values but 4 index values");
  assert.throws(() => cl([{ values: sig, idx: sig }]), TypeError);
  assert.throws(() => cl([[1, "x"], [1, 2]]), TypeError);
  assert.throws(() => cl([{ values: sig.slice(0, 2), index: [true, false] }, { values: sig.slice(0, 2), index: [0, 1] }]), TypeError);
  assert.throws(() => cl([gen([1]), sig]), TypeError);
  assert.throws(() => cl([gen([1]), agen([1])]), TypeError);
  assert.throws(() => cl([]), RangeError);
  const { values: v } = cl([[1, null, 3], [4, 5, 6]]) as Batch;
  assert.ok(Number.isNaN(v.data[2]));
  // int and float indexes mix
  const { values: w, index: i } = cl([{ values: sig.slice(0, 2), index: [0, 1] }, { values: sig.slice(0, 2), index: [0.5, 1.5] }]) as Batch;
  assertNanEq(i!, [0.5, 1, 1.5]);
  assert.deepEqual(w.shape, [3, 2]);
});

test("combineLatest: positional iterators are paired by position", async () => {
  await ready();
  const out = [...cl([gen(rowsOf(bars)), gen(sig)])] as Row[];
  assert.equal(out.length, T);
  assert.ok(out.every(({ value, index }) => value instanceof Float64Array && index === null));
  assertNanEq(flatRows(out), columnStack(bars, sig));
  // one field per source: number[] rows
  const narrow = [...cl([gen([1, null]), gen([3, 4])])] as Row[];
  assert.deepEqual(narrow[0], { value: [1, 3], index: null });
  assert.ok(Number.isNaN(narrow[1].value[0]));
  assert.throws(() => [...cl([gen([1, 2, 3]), gen([1, 2])])],
    (e: any) => e instanceof RangeError && /stay in step/.test(e.message));
  assert.throws(() => [...cl([gen([[1, 2], [1, 2, 3]]), gen([1, 2])])],
    (e: any) => e instanceof RangeError && /has 3 values, but its first event had 2/.test(e.message));
  assert.throws(() => [...cl([gen([1, "x"]), gen([1, 2])])], TypeError);
  assert.deepEqual([...cl([gen([]), gen([])])], []);
});

test("combineLatest: two-value Array events are rows, never [value, index]", async () => {
  await ready();
  const rows = [...cl([gen([[1, 2], [3, 4]]), gen([[5, 6], [7, 8]])])] as Row[];
  assert.deepEqual(rows.map((r) => [Array.from(r.value), r.index]), [[[1, 2, 5, 6], null], [[3, 4, 7, 8], null]]);
  const { values } = cl([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) as Batch;
  assertNanEq(flatRows(rows), values.data);
});

test("combineLatest: indexed iterators match the batch join", async () => {
  await ready();
  const ia = Array.from({ length: T }, (_, i) => i * 2), ib = [1, 5, 9];
  const { values: batch, index: bidx } = cl([{ values: bars, index: ia }, { values: sig.slice(0, 3), index: ib }]) as Batch;
  const lazy = [...cl([
    gen(rowsOf(bars).map((r, t) => ev(r, ia[t]))), gen(Array.from(sig.slice(0, 3)).map((x, t) => ev(x, ib[t]))),
  ])] as Row[];
  assertNanEq(flatRows(lazy), batch.data);
  assert.deepEqual(lazy.map(({ index }) => index), Array.from(bidx!));
  // scalar events: number[] rows, one input too (Python's ((1.0,), 0))
  const two = [...cl([gen([ev(1, 0), ev(2, 2)]), gen([ev(10, 1)])])] as Row[];
  assert.deepEqual(two, [{ value: [1, 10], index: 1 }, { value: [2, 10], index: 2 }]);
  const one = [...cl([gen([ev(1, 0), ev(2, 2)])])] as Row[];
  assert.deepEqual(one, [{ value: [1], index: 0 }, { value: [2], index: 2 }]);
  // a fractional index joins, as in the batch form; equal indexes collapse
  const frac = [...cl([gen([ev(1, 0.5), ev(2, 1.5)]), gen([ev(10, 0.5), ev(20, 1.25)])])] as Row[];
  const { values: fb, index: fi } = cl([{ values: [1, 2], index: [0.5, 1.5] }, { values: [10, 20], index: [0.5, 1.25] }]) as Batch;
  assertNanEq(flatRows(frac), fb.data);
  assert.deepEqual(frac.map(({ index }) => index), Array.from(fi!));
  // a NaN index is a RangeError, as in Python
  assert.throws(() => [...cl([gen([ev(1, 0), ev(2, NaN)])])],
    (e: any) => e instanceof RangeError && /the index of input 0 is NaN at event 1/.test(e.message));
  // a two-value row with its index
  const wide = [...cl([gen([ev([1, 2], 0)]), gen([ev(3, 0)])])] as Row[];
  assert.deepEqual(wide.map((r) => [Array.from(r.value), r.index]), [[[1, 2, 3], 0]]);
  // on_any with an empty source matches the batch join
  const { values: e, index: ei } = cl([{ values: sig.slice(0, 3), index: [0, 1, 2] }, { values: new Float64Array(0), index: [] }], { emit: "on_any" }) as Batch;
  const le = [...cl([gen([ev(0, 0), ev(10, 1), ev(20, 2)]), gen([])], { emit: "on_any" })] as Row[];
  assertNanEq(le.flatMap(({ value }) => value), e.data);
  assert.deepEqual(le.map(({ index }) => index), Array.from(ei!));
  assert.throws(() => [...cl([gen([ev(1, 0)]), gen([2])])],
    (e: any) => e instanceof RangeError && /cannot mix positional/.test(e.message));
  assert.throws(() => [...cl([gen([ev(1, 0), [2, 1]]), gen([ev(2, 1)])])],
    (e: any) => e instanceof TypeError && /expected a \{ value, index \} event/.test(e.message));
  assert.throws(() => [...cl([gen([ev(1, "0" as any)])])], TypeError);
});

test('combineLatest: iterator layout ["TA", "T"] gives [K, 2] rows, positional and indexed', async () => {
  await ready();
  const { values: batch } = cl([panel, sig], { layout: ["TA", "T"] }) as Batch;
  const pos = [...cl([gen(rowsOf(panel)), gen(sig)], { layout: ["TA", "T"] })] as Row[];
  assert.deepEqual(pos[0].value.shape, [3, 2]);
  assertNanEq(flatRows(pos), batch.data);
  const idx = [...cl([gen(rowsOf(panel).map((r, t) => ev(r, t))), gen(Array.from(sig).map((x, t) => ev(x, t)))],
    { layout: ["TA", "T"] })] as Row[];
  assertNanEq(flatRows(idx), batch.data);
  assert.deepEqual(idx.map(({ index }) => index), Array.from({ length: T }, (_, t) => t));
});

test("combineLatest: a reordering layout reads iterator events as the batch join does", async () => {
  await ready();
  // (T, F, A) data read as "TFA": each event is (F, A)
  const cube: Nd = { data: Float64Array.from({ length: T * 6 }, (_, i) => i), shape: [T, 2, 3] };
  const { values: batch } = cl([cube, sig], { layout: ["TFA", "T"] }) as Batch;
  assert.deepEqual(batch.shape, [T, 3, 3]);
  const events = Array.from({ length: T }, (_, t) => ({ data: cube.data.slice(t * 6, t * 6 + 6), shape: [2, 3] }));
  const lazy = [...cl([gen(events), gen(sig)], { layout: ["TFA", "T"] })] as Row[];
  assertNanEq(flatRows(lazy), batch.data);
});

test("combineLatest: async iterators, positional and indexed", async () => {
  await ready();
  const pos = await collect(cl([agen(rowsOf(bars)), agen(sig)])) as Row[];
  assertNanEq(flatRows(pos), columnStack(bars, sig));
  const { values: batch } = cl([panel, sig], { layout: ["TA", "T"] }) as Batch;
  const idx = await collect(cl(
    [agen(rowsOf(panel).map((r, t) => ev(r, t))), agen(Array.from(sig).map((x, t) => ev(x, t)))],
    { layout: ["TA", "T"] })) as Row[];
  assert.deepEqual(idx[0].value.shape, [3, 2]);
  assertNanEq(flatRows(idx), batch.data);
  await assert.rejects(collect(cl([agen([1, 2]), agen([1])])),
    (e: any) => e instanceof RangeError && /stay in step/.test(e.message));
});

test("combineLatest: a stream reads nothing before its first next(), and closes its sources", async () => {
  await ready();
  let pulled = 0, closed = 0;
  const src = () => ({
    [Symbol.iterator]() { return this; },
    next() { pulled++; return { done: false, value: ev(pulled, pulled) }; },
    return() { closed++; return { done: true, value: undefined }; },
  });
  const it = cl([src(), src()]) as Iterator<unknown>;
  assert.equal(pulled, 0);
  it.next();
  it.return!(undefined);
  assert.equal(closed, 2);
});

test("split: its outputs feed combineLatest as {values, index} streams", async () => {
  await ready();
  const parts = split([1, 2, 3, 4], [0, 1, 0, 1], { index: [0, 1, 2, 3] });
  const { values, index } = cl(parts) as Batch;
  assertNanEq(index!, [1, 2, 3]);
  assertNanEq(values.data, [1, 2, 3, 2, 3, 4]);
});

test("split: its outputs feed a Pipeline as they are, with or without an index", async () => {
  await ready();
  const x = Input("x"), y = Input("y");
  const p = new Pipeline([x, y], [combineLatest([x, y])]);
  const [a, b] = split([1, 2, 3, 4], [0, 1, 0, 1], { index: [0, 1, 2, 3] });
  const out = p({ x: a, y: b }) as any;
  assertNanEq(out.index, [1, 2, 3]);
  const [c, d] = split([1, 2, 3, 4], [0, 1, 0, 1]);
  assert.equal(c.index, null);
  const pos = p({ x: c, y: d }) as any;
  assertNanEq(pos.index, [0, 1]);
  p.dispose();
});

test("combineLatest: a step of numbers skips WASM and gives what the C++ zip gives", async () => {
  await ready();
  // Numbers take the JavaScript fast path; one-value Float64Arrays take the
  // C++ zip. Rows, NaN and messages are the same.
  const a = [1, 2, NaN, 4], b = [10, 20, 30, 40];
  const box = (xs: number[]) => xs.map((x) => new Float64Array([x]));
  const fast = [...cl([gen(a), gen(b)])] as Row[];
  const slow = [...cl([gen(box(a)), gen(box(b))])] as Row[];
  const mixed = [...cl([gen([1, 2, null, 4]), gen(b)])] as Row[];   // null steps go to C++
  for (const rows of [slow, mixed]) {
    assert.equal(rows.length, fast.length);
    rows.forEach((r, t) => {
      assert.ok(Array.isArray(r.value) && r.index === null);
      assertNanEq(r.value, fast[t].value);
    });
  }
  const message = (f: () => unknown) => { try { f(); } catch (e: any) { return [e.constructor.name, e.message]; } return null; };
  const outOfStep = message(() => [...cl([gen([1, 2, 3]), gen([1, 2])])]);
  assert.deepEqual(message(() => [...cl([gen(box([1, 2, 3])), gen(box([1, 2]))])]), outOfStep);
  assert.equal(outOfStep![0], "RangeError");
  assert.deepEqual(message(() => [...cl([gen([1, [2, 3]]), gen([1, 2])])]),
    ["RangeError", "combineLatest: an event of input 0 has 2 values, but its first event had 1"]);
});

test("combineLatest: {values, index} takes plain Arrays and typed arrays; other objects raise", async () => {
  await ready();
  const { values, index } = cl([{ values: [1, 2], index: [0, 1] }, { values: [3, 4], index: new Int32Array([0, 1]) }]) as Batch;
  assertNanEq(values.data, [1, 3, 2, 4]);
  assertNanEq(index!, [0, 1]);
  assert.throws(() => cl([{ values: 3, index: [0] }]), TypeError);
  assert.throws(() => cl([{ index: [0, 1] }]), TypeError);
  assert.throws(() => cl([{ values: [1, 2], index: [0, 1], extra: 1 }]), TypeError);
});

test("E4: emit on_any with a wide Input whose first rows are dropped raises; when_all works", async () => {
  await ready();
  // Dropna advances its port's watermark past the dropped rows, so a row is
  // due before the first 4-column event reaches the node. Mirrors
  // tests/test_combine_latest_multicolumn.py.
  const T = 12;
  const rows = Array.from({ length: T }, (_, k) => [4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3].map((v) => (k < 4 ? NaN : v)));
  const sig = Array.from({ length: T }, (_, k) => 10 * k);
  const index = Array.from({ length: T }, (_, k) => k);
  const feeds: any = {
    x: { values: { data: Float64Array.from(rows.flat()), shape: [T, 4] }, index },
    y: { values: sig, index },
  };
  const isOnAny = (e: unknown) => e instanceof RangeError && /emit="when_all"/.test((e as Error).message);
  const x = Input("x"), y = Input("y");
  const p = new Pipeline([x, y], [combineLatest([dropna(x), y], { emit: "on_any" })]);
  const q = new Pipeline([x, y], [combineLatest([dropna(x), y])]);
  try {
    assert.throws(() => p(feeds), isOnAny, "batch");
    const s = p.live();
    assert.throws(() => {
      for (let k = 0; k < T; k++) { s.push("x", k, rows[k]); s.push("y", k, sig[k]); }
    }, isOnAny, "live");
    const out: any = q(feeds);
    assert.deepEqual(out.values.shape, [8, 5]);
    assert.deepEqual(Array.from(out.index), [4, 5, 6, 7, 8, 9, 10, 11]);
  } finally {
    p.dispose();
    q.dispose();
  }
});

// combineLatest reserves its output rows (one per distinct index), not the sum
// of the inputs' rows times the joined width: 32 aligned inputs of 2e5 rows
// reserved 1.6 GB of the WebAssembly heap and threw std::bad_alloc.
test("combineLatest of many aligned inputs fits the WebAssembly heap", async () => {
  await ready();
  const n = 32, T = 200_000;
  const index = Float64Array.from({ length: T }, (_, i) => i);
  const ins = Array.from({ length: n }, (_, j) => ({ values: Float64Array.from({ length: T }, (_, i) => i + j), index }));
  const r: any = combineLatest(ins);
  assert.equal(r.index.length, T);
  const v = r.values.data as Float64Array;
  assert.equal(v.length, T * n);
  assert.equal(v[(T - 1) * n + (n - 1)], T - 1 + n - 1);
});
