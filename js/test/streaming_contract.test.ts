import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { ready, current, BacktestPortfolioTarget, PortfolioReport } from "../src/index.ts";
import { wrapOp } from "../src/runtime.ts";
import * as ops from "../src/generated/ops.ts";
import type { NdArray } from "../src/index.ts";

// The streaming contract, over every generated operator that builds with its
// defaults: a sync iterator reads nothing before the first next(), and sync
// and async streams give exactly the batch result, for scalar events and
// (K,) asset events, with one source per input or one source of packed events.
// Mirrors the streaming section of tests/test_shape_contract.py.

await ready();

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(HERE, "..", "src", "generated", "ops.ts"), "utf8");
const kindOf = (name: string) =>
  new RegExp(`export function ${name}\\(`).test(SRC)
    ? (SRC.split(`export function ${name}(`).at(-1)!.match(/return (?:_params\()?(wrapOp|wrapRowOp|wrapReducerOp)\(/)?.[1] ?? "")
    : "";

const T = 40, K = 3;

// Bounded-positive values, as in the Python contract tests, but a different
// series per input and per asset, so a wrong column or field shows.
function rng(seed: number) {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return 0.1 + 0.8 * (s / 2 ** 32); };
}
const series = (n: number, seed: number) => { const r = rng(seed); return Float64Array.from({ length: n }, r); };
const scalars = (nIn: number) => Array.from({ length: nIn }, (_, i) => series(T, 11 + i));
const panel = (nIn: number): NdArray[] => Array.from({ length: nIn }, (_, i) => ({ data: series(T * K, 101 + i), shape: [T, K] }));
const rowsOf = (x: NdArray): Float64Array[] => {
  const w = x.data.length / x.shape[0];
  return Array.from({ length: x.shape[0] }, (_, t) => x.data.slice(t * w, (t + 1) * w));
};
const packedRows = (p: NdArray[]): NdArray[] =>
  Array.from({ length: T }, (_, t) => {
    const n = p.length, k0 = p[0].shape[1], data = new Float64Array(k0 * n);
    for (let k = 0; k < k0; k++) for (let i = 0; i < n; i++) data[k * n + i] = p[i].data[t * k0 + k];
    return { data, shape: [k0, n] };
  });

// Every result as one flat array: a number, an array of numbers, a typed array,
// an NdArray, or a list of any of these.
function flat(x: any): number[] {
  if (typeof x === "number") return [x];
  if (x instanceof Float64Array) return Array.from(x);
  if (x && x.data instanceof Float64Array) return Array.from(x.data);
  if (Array.isArray(x)) return x.flatMap(flat);
  throw new Error(`unexpected result ${typeof x}`);
}
function same(got: any, want: any, what: string) {
  const g = flat(got), w = flat(want);
  assert.equal(g.length, w.length, `${what}: length`);
  for (let i = 0; i < w.length; i++)
    if (!(g[i] === w[i] || (Number.isNaN(g[i]) && Number.isNaN(w[i]))))
      assert.fail(`${what}[${i}]: got ${g[i]}, expected ${w[i]}`);
}

function* iter<V>(xs: Iterable<V>) { yield* xs; }
async function* aiter<V>(xs: Iterable<V>) { for (const x of xs) yield x; }
async function drain(it: AsyncIterable<any>) { const out: any[] = []; for await (const v of it) out.push(v); return out; }
const zipRows = <V>(cols: ArrayLike<V>[]): V[][] => Array.from({ length: cols[0].length }, (_, t) => cols.map((c) => c[t]));

type Factory = () => any;
const factories: Array<{ name: string; make: Factory; kind: string }> = [];
for (const [name, f] of Object.entries(ops)) {
  if (typeof f !== "function" || !/^[A-Z]/.test(name)) continue;
  try { (f as Factory)().dispose(); } catch { continue; }      // needs arguments
  factories.push({ name, make: f as Factory, kind: kindOf(name) });
}

test("the contract covers the generated operators", () => {
  assert.ok(factories.filter((f) => f.kind === "wrapOp").length > 200);
  assert.ok(factories.some((f) => f.kind === "wrapRowOp"));
});

for (const { name, make, kind } of factories.filter((f) => f.kind === "wrapOp")) {
  const probe = make();
  const nIn: number = probe.numInputs, nOut: number = probe.numOutputs;
  probe.dispose();
  const xs = scalars(nIn), p = panel(nIn);
  const batch = (...args: any[]) => make()(...args);

  test(`${name}: numInputs and numOutputs`, () => {
    assert.ok(Number.isInteger(nIn) && nIn >= 1 && Number.isInteger(nOut) && nOut >= 1);
  });

  test(`${name}: a sync stream reads nothing before the first next()`, () => {
    const pulls = new Array(nIn).fill(0);
    const counted = (i: number) => (function* () { for (const v of xs[i]) { pulls[i]++; yield v; } })();
    const it = make()(...xs.map((_, i) => counted(i))) as Iterator<any>;
    assert.deepEqual(pulls, new Array(nIn).fill(0));
    for (const step of [1, 2, 3]) {
      it.next();
      assert.deepEqual(pulls, new Array(nIn).fill(step));
    }
  });

  test(`${name}: sync and async scalar streams equal batch`, async () => {
    const want = batch(...xs);
    same([...make()(...xs.map((x) => iter(x)))], want, "sync");
    same(await drain(make()(...xs.map((x) => aiter(x)))), want, "async");
  });

  test(`${name}: sync and async asset-event streams equal batch`, async () => {
    const want = batch(...p);
    same([...make()(...p.map((x) => iter(rowsOf(x))))], want, "sync");
    same(await drain(make()(...p.map((x) => aiter(rowsOf(x))))), want, "async");
    // (K, 1) events read as (K,).
    const unit = p.map((x) => rowsOf(x).map((r) => ({ data: r, shape: [K, 1] })));
    same([...make()(...unit.map((x) => iter(x)))], want, "sync (K, 1)");
  });

  if (nIn > 1) {
    test(`${name}: one source of packed events equals batch`, async () => {
      const want = batch(...xs);
      same([...make()(iter(zipRows(xs)))], want, "sync tuples");
      same(await drain(make()(aiter(zipRows(xs)))), want, "async tuples");
      const wantP = batch(...p);
      same([...make()(iter(packedRows(p)))], wantP, "sync (K, nIn)");
      same(await drain(make()(aiter(packedRows(p)))), wantP, "async (K, nIn)");
      same([...make()(iter(zipRows(p.map(rowsOf))))], wantP, "sync tuples of rows");
      same(await drain(make()(aiter(zipRows(p.map(rowsOf))))), wantP, "async tuples of rows");
    });

    test(`${name}: a scalar source broadcasts over asset events`, async () => {
      const want = batch(xs[0], ...p.slice(1));
      same([...make()(iter(xs[0]), ...p.slice(1).map((x) => iter(rowsOf(x))))], want, "sync");
      same(await drain(make()(aiter(xs[0]), ...p.slice(1).map((x) => aiter(rowsOf(x))))), want, "async");
    });

    test(`${name}: mixing sync and async sources raises`, () => {
      assert.throws(() => make()(aiter(xs[0]), ...xs.slice(1).map((x) => iter(x))), /async/);
    });
  }
}

for (const { name, make } of factories.filter((f) => f.kind === "wrapRowOp")) {
  const fi: number = make().rowOp.fieldsIn;
  const blocks: NdArray[] = Array.from({ length: fi }, (_, i) => ({
    data: series(12 * 4, 301 + i).map((v) => v + 0.4), shape: [12, 4],
  }));
  const rows = blocks.map(rowsOf);

  test(`${name}: a row stream reads nothing before the first next()`, () => {
    const pulls = new Array(fi).fill(0);
    const counted = (i: number) => (function* () { for (const r of rows[i]) { pulls[i]++; yield r; } })();
    const it = make()(...rows.map((_, i) => counted(i))) as Iterator<any>;
    assert.deepEqual(pulls, new Array(fi).fill(0));
    for (const step of [1, 2, 3]) {
      it.next();
      assert.deepEqual(pulls, new Array(fi).fill(step));
    }
  });

  test(`${name}: sync and async row streams equal batch`, async () => {
    const want = make()(...blocks);
    same([...make()(...rows.map((r) => iter(r)))], want, "sync");
    same(await drain(make()(...rows.map((r) => aiter(r)))), want, "async");
    const unit = rows.map((rs) => rs.map((r) => ({ data: r, shape: [4, 1] })));
    same([...make()(...unit.map((r) => iter(r)))], want, "sync (K, 1)");
    same(await drain(make()(...unit.map((r) => aiter(r)))), want, "async (K, 1)");
    if (fi > 1) {
      same([...make()(iter(zipRows(rows)))], want, "sync tuples");
      same(await drain(make()(aiter(zipRows(rows)))), want, "async tuples");
    }
  });

  if (fi > 1) {
    test(`${name}: mixing row sources raises`, () => {
      assert.throws(() => make()(aiter(rows[0]), ...rows.slice(1).map((r) => iter(r))), /async/);
      assert.throws(() => make()(iter(rows[0]), ...blocks.slice(1)), /cannot be mixed/);
    });
  }
}

test("PortfolioReport: lazy, and sync and async streams equal batch", async () => {
  const w: NdArray = { data: series(12 * 4, 7), shape: [12, 4] };
  const px: NdArray = { data: series(12 * 4, 8).map((v) => v + 1), shape: [12, 4] };
  const engine = BacktestPortfolioTarget()(w, px) as NdArray;          // (T, K, 4)
  const events = Array.from({ length: 12 }, (_, t): NdArray => ({
    data: engine.data.slice(t * 16, (t + 1) * 16), shape: [4, 4],
  }));
  const want = PortfolioReport()(engine);
  let pulls = 0;
  const it = PortfolioReport()((function* () { for (const e of events) { pulls++; yield e; } })()) as Iterator<any>;
  assert.equal(pulls, 0);
  const first = it.next().value;
  assert.equal(pulls, 1);
  const rest: any[] = [];
  for (let r = it.next(); !r.done; r = it.next()) rest.push(r.value);
  same([first, ...rest], want, "sync");
  same(await drain(PortfolioReport()(aiter(events))), want, "async");
});

test("a source that ends early ends the stream and closes the other sources", () => {
  let closed = false;
  const long = (function* () { try { for (let i = 0; i < 10; i++) yield i; } finally { closed = true; } })();
  const out = [...(ops.Add()(iter([1, 2, 3]), long) as Iterable<number>)];
  assert.deepEqual(out, [1, 3, 5]);
  assert.ok(closed);
});

test("an iterator mixed with an array raises", () => {
  assert.throws(() => ops.Add()(iter([1, 2]), new Float64Array([1, 2])), /cannot be mixed/);
  assert.throws(() => ops.Add()(iter([1]), iter([1]), iter([1])), RangeError);
});

test("a one-input op takes (K, F) events like a (T, K, F) batch", async () => {
  const x: NdArray = { data: series(T * 6, 5), shape: [T, 2, 3] };
  const want = ops.RollingMean(3)(x) as NdArray;
  const events = Array.from({ length: T }, (_, t): NdArray => ({ data: x.data.slice(t * 6, (t + 1) * 6), shape: [2, 3] }));
  const got = [...(ops.RollingMean(3)(iter(events)) as Iterable<NdArray>)];
  assert.deepEqual(got[0].shape, [2, 3]);
  same(got, want, "sync");
  same(await drain(ops.RollingMean(3)(aiter(events))), want, "async");
});

test("multi-output asset events are (K, nOut)", () => {
  const p = panel(1);
  const got = [...(ops.MACD()(iter(rowsOf(p[0]))) as Iterable<NdArray>)];
  assert.deepEqual(got[0].shape, [K, 3]);
  same(got, ops.MACD()(p[0]), "MACD");
});

test("a failed copy ends the stream and closes its source; the next stream works", () => {
  const M = current();
  let calls = 0;
  const op = wrapOp(M, () => {
    calls++;
    if (calls === 2) throw new Error("no copy");
    return new M.RollingMean(2, "strict");
  });
  const x = Float64Array.from({ length: 12 }, (_, i) => i + 1);
  const rows = [0, 1, 2].map((t) => x.slice(t * 4, t * 4 + 4));
  let closed = false;
  const src = (function* () { try { yield* rows; } finally { closed = true; } })();
  const it = op(src) as Iterator<Float64Array>;
  assert.throws(() => it.next(), /no copy/);
  assert.ok(closed);
  assert.equal(it.next().done, true);
  const want = ops.RollingMean(2, "strict")({ data: x, shape: [3, 4] });
  same([...(op(iter(rows)) as Iterable<Float64Array>)], want, "next stream");
});

test("an event shape change in a multi-input stream raises", () => {
  const it = ops.Add()(iter([new Float64Array(3), new Float64Array(3)]), iter([new Float64Array(3), new Float64Array(4)])) as Iterator<any>;
  it.next();
  assert.throws(() => it.next(), /same shape/);
  const it2 = ops.Add()(iter([1, new Float64Array(2)]), iter([1, 2])) as Iterator<any>;
  assert.equal(it2.next().value, 2);
  assert.throws(() => it2.next(), /^RangeError: Add: event 1 of input 0 has shape \(2,\), expected a number$/);
});

test("assets that do not broadcast raise from the stream", () => {
  const it = ops.Add()(iter([new Float64Array(3)]), iter([new Float64Array(4)])) as Iterator<any>;
  assert.throws(() => it.next(), /do not broadcast/);
});

test("using a multi-input stream after dispose raises", () => {
  const op = ops.Add();
  const it = op(iter([new Float64Array(3), new Float64Array(3)]), iter([new Float64Array(3), new Float64Array(3)])) as Iterator<any>;
  it.next();
  op.dispose();
  assert.throws(() => it.next(), /used after dispose/);
});

test("overlapping next() calls of an async stream run in order", async () => {
  const it = ops.Add()(aiter([1, 2]), aiter([3, 4])) as AsyncIterator<number>;
  const [a, b, c] = await Promise.all([it.next(), it.next(), it.next()]);
  assert.deepEqual([a, b, c], [{ done: false, value: 4 }, { done: false, value: 6 }, { done: true, value: undefined }]);
});

// A source that records whether it was closed.
function tracked<V>(xs: Iterable<V>, log: string[], name: string) {
  return (function* () { try { yield* xs; } finally { log.push(name); } })();
}
function atracked<V>(xs: Iterable<V>, log: string[], name: string) {
  return (async function* () { try { for (const x of xs) yield x; } finally { log.push(name); } })();
}

test("an error in an event closes every source and ends the stream", () => {
  const log: string[] = [];
  const it = ops.Add()(
    tracked([new Float64Array(3), new Float64Array(3), new Float64Array(3)], log, "a"),
    tracked([new Float64Array(3), new Float64Array(4), new Float64Array(3)], log, "b"),
  ) as Iterator<any>;
  it.next();
  assert.throws(() => it.next(), /same shape/);
  assert.deepEqual(log.sort(), ["a", "b"]);
  assert.equal(it.next().done, true);
  // A for...of loop sees the same: the error escapes and the sources are closed.
  const log2: string[] = [];
  assert.throws(() => { for (const _ of ops.RollingMean(2)(tracked([new Float64Array(2), new Float64Array(3)], log2, "x"))) { /* */ } }, /same shape/);
  assert.deepEqual(log2, ["x"]);
});

test("an error in an async event closes every source and ends the stream", async () => {
  const log: string[] = [];
  const it = ops.Add()(
    atracked([new Float64Array(3), new Float64Array(3)], log, "a"),
    atracked([new Float64Array(3), new Float64Array(4)], log, "b"),
  ) as AsyncIterator<any>;
  await it.next();
  await assert.rejects(it.next() as Promise<any>, /same shape/);
  assert.deepEqual(log.sort(), ["a", "b"]);
  assert.equal((await it.next()).done, true);
});

test("throw() on a stream closes the sources and rethrows", async () => {
  const log: string[] = [];
  const it = ops.Add()(tracked([1, 2, 3], log, "a"), tracked([1, 2, 3], log, "b")) as Generator<number>;
  it.next();
  assert.throws(() => it.throw(new Error("stop")), /stop/);
  assert.deepEqual(log.sort(), ["a", "b"]);
  assert.equal(it.next().done, true);
  const alog: string[] = [];
  const ait = ops.Add()(atracked([1, 2], alog, "a"), atracked([1, 2], alog, "b")) as AsyncGenerator<number>;
  await ait.next();
  await assert.rejects(ait.throw(new Error("stop")), /stop/);
  assert.deepEqual(alog.sort(), ["a", "b"]);
});

test("closing an async stream closes every source even when one close rejects", async () => {
  const log: string[] = [];
  const bad: AsyncIterableIterator<number> = {
    [Symbol.asyncIterator]() { return this; },
    async next() { return { done: false, value: 1 }; },
    async return() { log.push("bad"); throw new Error("close failed"); },
  };
  const it = ops.Add()(bad, atracked([1, 2, 3], log, "good")) as AsyncGenerator<number>;
  await it.next();
  await assert.rejects(it.return(undefined), /close failed/);
  assert.deepEqual(log.sort(), ["bad", "good"]);
});

test("streams have the built-in iterator helpers", async () => {
  const want = ops.RollingMean(3)(scalars(1)[0]) as Float64Array;
  const s = ops.RollingMean(3)(iter(scalars(1)[0])) as any;
  assert.equal(typeof s.throw, "function");
  const G = (globalThis as any).Iterator;
  if (typeof G === "function") {
    assert.ok(s instanceof G);
    same(s.take(5).toArray(), want.slice(0, 5), "take");
    same((ops.RollingMean(3)(iter(scalars(1)[0])) as any).map((v: number) => v * 2).toArray(), Array.from(want, (v) => v * 2), "map");
  }
  const asyncProto = Object.getPrototypeOf(Object.getPrototypeOf((async function* () { /* */ })()));
  const a = ops.RollingMean(3)(aiter(scalars(1)[0])) as any;
  assert.ok(Object.getPrototypeOf(asyncProto) && Object.prototype.isPrototypeOf.call(Object.getPrototypeOf(asyncProto), a));
  assert.equal(typeof a.throw, "function");
  if (typeof a.take === "function") same(await a.take(5).toArray(), want.slice(0, 5), "async take");
});

test("an async stream reads nothing before the first next()", async () => {
  const xs = scalars(2);
  const pulls = [0, 0];
  const counted = (i: number) => (async function* () { for (const v of xs[i]) { pulls[i]++; yield v; } })();
  const it = ops.Add()(counted(0), counted(1)) as AsyncIterator<number>;
  await Promise.resolve();
  assert.deepEqual(pulls, [0, 0]);
  for (const step of [1, 2, 3]) {
    await it.next();
    assert.deepEqual(pulls, [step, step]);
  }
  const rows = [new Float64Array([1, 2]), new Float64Array([3, 4])];
  let rp = 0;
  const rit = ops.CrossDemean()((async function* () { for (const r of rows) { rp++; yield r; } })()) as AsyncIterator<any>;
  assert.equal(rp, 0);
  await rit.next();
  assert.equal(rp, 1);
});

test("interleaved streams of one op keep their own event shapes", () => {
  const op = ops.Add();
  const p3 = panel(2);
  const p2: NdArray[] = [0, 1].map((i) => ({ data: series(T * 2, 501 + i), shape: [T, 2] }));
  const s1 = op(...p3.map((x) => iter(rowsOf(x)))) as Iterator<any>;
  const s2 = op(iter(zipRows(scalars(2)))) as Iterator<any>;
  const s3 = op(iter(packedRows(p2))) as Iterator<any>;
  const r1 = rowsOf(ops.Add()(...p3) as NdArray), r3 = rowsOf(ops.Add()(...p2) as NdArray);
  const xs = scalars(2);
  for (let t = 0; t < 5; t++) {
    same(s1.next().value, r1[t], `assets ${t}`);
    assert.equal(s2.next().value, xs[0][t] + xs[1][t]);
    same(s3.next().value, r3[t], `packed ${t}`);
  }
});

test("every result has its own shape array", () => {
  const p = panel(1);
  const got = [...(ops.MACD()(iter(rowsOf(p[0]))) as Iterable<NdArray>)];
  assert.notEqual(got[0].shape, got[1].shape);
  got[0].shape.push(99);
  assert.deepEqual(got[1].shape, [K, 3]);
});

// ---- Independent streams (decision D8) -------------------------------------
//
// op(number) is one step of the op's running series. Every other call (array,
// iterator, async iterator) is an independent run: it starts clean, runs on its
// own copies, and resets the running series when it opens. Mirrors the D8
// section of tests/test_shape_contract.py.

const A = [1, 2, 3, 4, 5, 6], B = [100, 200, 300, 400, 500, 600];
const mean = () => ops.RollingMean(3, "expanding");
const interleave = (i1: Iterator<any>, i2: Iterator<any>, n: number) => {
  const g1: any[] = [], g2: any[] = [];
  for (let t = 0; t < n; t++) { g1.push(i1.next().value); g2.push(i2.next().value); }
  return [g1, g2];
};

test("D8: interleaved streams of one op are independent", () => {
  const op = mean();
  const [g1, g2] = interleave(op(iter(A)), op(iter(B)), A.length);
  same(g1, mean()(A), "first");
  same(g2, mean()(B), "second");
});

test("D8: streams one after the other are independent", () => {
  const op = mean();
  same([...op(iter(A))], mean()(A), "first");
  same([...op(iter(B))], mean()(B), "second");
});

test("D8: a stream on a used op starts clean", () => {
  const op = mean();
  op(1); op(2);
  same([...op(iter(A))], mean()(A), "stream");
});

test("D8: opening a stream resets the running series", () => {
  const op = mean();
  op(1); op(2);
  const it = op(iter(B)) as Iterator<number>;
  assert.equal(op(3), 3);                               // a first event
  same([...(it as any)], mean()(B), "stream");          // the live calls did not touch it
});

test("D8: live calls and reset() do not touch a running stream", () => {
  const op = mean();
  const it = op(iter(A)) as Iterator<number>;
  const got = [it.next().value];
  op(50); op.reset();
  for (let r = it.next(); !r.done; r = it.next()) got.push(r.value);
  same(got, mean()(A), "stream");
});

test("D8: asset streams are independent", () => {
  const x: NdArray = { data: Float64Array.from({ length: 24 }, (_, i) => i + 1), shape: [6, 4] };
  const y: NdArray = { data: x.data.map((v) => v * 10).reverse(), shape: [6, 4] };
  const op = mean();
  op(1);
  const [g1, g2] = interleave(op(iter(rowsOf(x))), op(iter(rowsOf(y))), 6);
  same(g1, mean()(x), "first");
  same(g2, mean()(y), "second");
  assert.equal(op(3), 3);
});

test("D8: multi-input streams are independent", () => {
  const cov = () => ops.RollingCov(3, "expanding");
  const op = cov();
  op(1, 2); op(3, 1);
  const it1 = op(iter(A), iter(B)), it2 = op(iter(zipRows([B, A])));
  const [g1, g2] = interleave(it1, it2, A.length);
  same(g1, cov()(A, B), "fields");
  same(g2, cov()(B, A), "packed");
});

test("D8: row op streams are independent", () => {
  const bt = () => BacktestPortfolioTarget();
  const w: NdArray = { data: series(12 * 4, 7), shape: [12, 4] };
  const px: NdArray = { data: series(12 * 4, 8).map((v) => v + 1), shape: [12, 4] };
  const w2: NdArray = { data: series(12 * 4, 17), shape: [12, 4] };
  const px2: NdArray = { data: series(12 * 4, 18).map((v) => v + 1), shape: [12, 4] };
  const op = bt();
  const [r0, p0] = [rowsOf(w)[0], rowsOf(px)[0]];
  op(r0, p0);
  const it1 = op(iter(rowsOf(w)), iter(rowsOf(px)));
  const it2 = op(iter(zipRows([rowsOf(w2), rowsOf(px2)])));
  const [g1, g2] = interleave(it1, it2, 12);
  same(g1, bt()(w, px), "first book");
  same(g2, bt()(w2, px2), "second book");
  same(op(r0, p0), bt()(r0, p0), "first event after open");
});

test("D8: PortfolioReport streams are independent", () => {
  const book = (s1: number, s2: number) => {
    const w: NdArray = { data: series(12 * 4, s1), shape: [12, 4] };
    const px: NdArray = { data: series(12 * 4, s2).map((v) => v + 1), shape: [12, 4] };
    const e = BacktestPortfolioTarget()(w, px) as NdArray;
    return { e, events: Array.from({ length: 12 }, (_, t): NdArray => ({ data: e.data.slice(t * 16, (t + 1) * 16), shape: [4, 4] })) };
  };
  const b1 = book(7, 8), b2 = book(17, 18);
  const op = PortfolioReport();
  op(b1.events[0]);
  const [g1, g2] = interleave(op(iter(b1.events)), op(iter(b2.events)), 12);
  same(g1, PortfolioReport()(b1.e), "first");
  same(g2, PortfolioReport()(b2.e), "second");
  same(op(b1.events[0]), PortfolioReport()(b1.events[0]), "first event after open");
});

test("D8: async streams are independent", async () => {
  const op = mean();
  op(1); op(2);
  const it1 = op(aiter(A)) as AsyncIterator<number>, it2 = op(aiter(B)) as AsyncIterator<number>;
  const g1: number[] = [], g2: number[] = [];
  for (let t = 0; t < A.length; t++) { g1.push((await it1.next()).value); g2.push((await it2.next()).value); }
  same(g1, mean()(A), "first");
  same(g2, mean()(B), "second");
  assert.equal(op(3), 3);
});

test("D8: streaming an op without a factory raises", () => {
  const M = current();
  const op = wrapOp(M, new M.RollingMean(3, "expanding"));
  assert.throws(() => op(iter(A)), /factory/);
  op.dispose();
});

function probe(failAt: number | null) {
  const log = { returns: 0 };
  let i = 0;
  const it: IterableIterator<number> = {
    [Symbol.iterator]() { return this; },
    next() {
      if (failAt !== null && i === failAt) throw new Error("source broke");
      return { done: false, value: i++ };
    },
    return() { log.returns++; return { done: true, value: undefined }; },
  };
  return { it, log };
}

test("G4: a source whose next() threw is not closed; the others are", () => {
  const a = probe(null), b = probe(2);
  assert.throws(() => [...(ops.Add()(a.it, b.it) as Iterable<number>)], /source broke/);
  assert.equal(b.log.returns, 0);
  assert.equal(a.log.returns, 1);
});

test("G4: same for async sources", async () => {
  const mk = (failAt: number | null) => {
    const log = { returns: 0 };
    let i = 0;
    const it = {
      [Symbol.asyncIterator]() { return this; },
      async next() {
        if (failAt !== null && i === failAt) throw new Error("source broke");
        return { done: false, value: i++ };
      },
      async return() { log.returns++; return { done: true, value: undefined }; },
    };
    return { it, log };
  };
  const a = mk(null), b = mk(2);
  await assert.rejects(async () => { for await (const _ of ops.Add()(a.it, b.it) as AsyncIterable<number>) { void _; } }, /source broke/);
  assert.equal(b.log.returns, 0);
  assert.equal(a.log.returns, 1);
});
