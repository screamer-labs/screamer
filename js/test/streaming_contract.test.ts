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
    ? (SRC.split(`export function ${name}(`)[1].match(/return (wrapOp|wrapRowOp|wrapReducerOp)\(/)?.[1] ?? "")
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
    const n = p.length, data = new Float64Array(K * n);
    for (let k = 0; k < K; k++) for (let i = 0; i < n; i++) data[k * n + i] = p[i].data[t * K + k];
    return { data, shape: [K, n] };
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
  assert.throws(() => ops.Add()(iter([1]), iter([1]), iter([1])), TypeError);
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

test("a failed copy leaves the stream retryable", () => {
  const M = current();
  let calls = 0;
  const op = wrapOp(M, () => {
    calls++;
    if (calls === 2) throw new Error("no copy");
    return new M.RollingMean(2, "strict");
  });
  const x = Float64Array.from({ length: 12 }, (_, i) => i + 1);
  const rows = [0, 1, 2].map((t) => x.slice(t * 4, t * 4 + 4));
  const it = op(iter(rows)) as Iterator<Float64Array>;
  assert.throws(() => it.next(), /no copy/);
  const want = ops.RollingMean(2, "strict")({ data: x.slice(4), shape: [2, 4] });
  same([it.next().value, it.next().value], want, "retry");
});

test("an event shape change in a multi-input stream raises", () => {
  const it = ops.Add()(iter([new Float64Array(3), new Float64Array(3)]), iter([new Float64Array(3), new Float64Array(4)])) as Iterator<any>;
  it.next();
  assert.throws(() => it.next(), /same shape/);
  const it2 = ops.Add()(iter([1, new Float64Array(2)]), iter([1, 2])) as Iterator<any>;
  assert.equal(it2.next().value, 2);
  assert.throws(() => it2.next(), /same shape/);
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

test("a second next() of an async stream while one runs rejects", async () => {
  const it = ops.Add()(aiter([1, 2]), aiter([3, 4])) as AsyncIterator<number>;
  const first = it.next();
  await assert.rejects(it.next() as Promise<any>, /still running/);
  assert.deepEqual(await first, { done: false, value: 4 });
  assert.deepEqual(await it.next(), { done: false, value: 6 });
  assert.deepEqual((await it.next()).done, true);
});

test("a failed event leaves an async stream usable", async () => {
  const it = ops.RollingMean(2)(aiter([new Float64Array(2), new Float64Array(3), new Float64Array([1, 2])])) as AsyncIterator<any>;
  await it.next();
  await assert.rejects(it.next() as Promise<any>, /same shape/);
  same((await it.next()).value, [0.5, 1], "after the error");
});
