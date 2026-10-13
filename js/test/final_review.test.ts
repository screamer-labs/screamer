// The fixes of the 3.0.0 final review, JavaScript side: each case names the
// behaviour Python has (the expected values and messages are Python's).
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ready, current, Input, Pipeline, PortfolioReport, RollingMean, BollingerBands, RollingMinMax,
  resample, dropna, select, combineLatest, merge, split, strategyTest, positionTest, backtestReport, MARKET,
  BacktestPriceTarget,
} from "../src/index.ts";

const gc = (globalThis as any).gc as (() => void) | undefined;
const nanEq = (a: ArrayLike<number>, b: ArrayLike<number>) =>
  assert.deepEqual(Array.from(a, String), Array.from(b, String));
const settle = async () => {
  for (let i = 0; i < 6; i++) { gc?.(); await new Promise((r) => setTimeout(r, 0)); }
};

test("1: a reducer inside a Pipeline is kept and saved as in Python", async () => {
  await ready();
  const ins = ["a", "b", "c", "d"].map((n) => Input(n));
  const p = new Pipeline(ins, [PortfolioReport()(...ins)]);
  await settle();
  for (let i = 0; i < 200; i++) RollingMean(5);   // reuse any freed slot
  // Python's to_json(None) of the same Pipeline, text for text.
  assert.equal(p.toJSONText(null),
    '{"screamer_dag": 1, "inputs": ["a", "b", "c", "d"], "align_outputs": true, "max_pending": 1000000, ' +
    '"nodes": [{"id": 0, "kind": "input", "name": "a"}, {"id": 1, "kind": "input", "name": "b"}, ' +
    '{"id": 2, "kind": "input", "name": "c"}, {"id": 3, "kind": "input", "name": "d"}, ' +
    '{"id": 4, "kind": "functor", "cls": "PortfolioReport", "params": {}, "in": [0, 1, 2, 3]}], "outputs": [4]}');
  // A run raises the engine's width error in both languages (each input is
  // its own stream), after the collector as before it: no freed op is read.
  const feed = { a: [100, 101, 102], b: [0, 1, 1], c: [1, 1, 2], d: [0, 0.1, 0.1] };
  const msg = "an operator takes 4 column(s), but its input 0 has 1; each input of an operator is a separate " +
    "stream of 4-column events. To join streams into one row, use CombineLatest";
  assert.throws(() => p(feed), (e: any) => e instanceof RangeError && e.message === msg);
  const q = Pipeline.fromJSON(JSON.stringify(p));
  assert.throws(() => q(feed), (e: any) => e instanceof RangeError && e.message === msg);
  // One Node is refused as in Python.
  assert.throws(() => PortfolioReport()(Input("x")),
    (e: any) => e instanceof TypeError && e.message === "PortfolioReport DAG composition expects four portfolio columns.");
});

test("2: resample settings follow the shared rule (Python's messages)", async () => {
  await ready();
  const x = Float64Array.from([0, 1, 2, 3, 4]);
  const n = Input("x");
  const range = (f: () => unknown, msg: string) =>
    assert.throws(f, (e: any) => e instanceof RangeError && e.message === msg);
  range(() => resample(x, { every: 0 }), "resample: every must be positive");
  range(() => resample(x, { every: NaN }), "resample: every must be positive");
  range(() => resample(x, { every: -2 }), "resample: every must be positive");
  range(() => resample(n, { every: 0 }), "resample: every must be positive");
  range(() => resample(x, { count: 0 }), "resample: count must be >= 1");
  range(() => resample(x, { count: -1 }), "resample: count must be >= 1");
  range(() => resample(x, { threshold: 0 }), "resample: threshold must be > 0");
  range(() => resample(x, { threshold: -1 }), "resample: threshold must be > 0");
  range(() => resample(x, { every: 2, fill: "bogus" as any }), "resample: fill must be one of ('skip', 'nan', 'carry')");
  range(() => resample(x, { every: 2, label: "center" as any }), 'resample: label must be "left" or "right"');
  range(() => resample(x, { every: 2, maxAge: 3 }), "resample: max_age requires clock=True, agg='last', and fill='carry'");
  range(() => resample(x, { every: 2, count: 2 }), "resample: pass exactly one of freq=, every=, count=, threshold=, or clock=");
  range(() => resample(n, { clock: true, reducer: RollingMean(2)(Input("_")) as any, driver: Input("c") }),
    "resample clock= mode requires a builtin string agg (first/last/min/max/sum/count/mean/ohlc); " +
    "functor reducers are not supported with clock=. Apply your functor to the resampled output instead.");
  assert.throws(() => resample(n, { threshold: 2, agg: "ohlcv" }), (e: any) => e instanceof RangeError && /threshold= mode supports single-column/.test(e.message));
  assert.throws(() => resample(x, { every: 2, agg: "bogus" }), (e: any) => e instanceof RangeError && /unknown agg string 'bogus'/.test(e.message));
  assert.throws(() => resample(x, { evry: 2 } as any), (e: any) => e instanceof TypeError && e.message === "resample: unknown option 'evry'");
  range(() => resample(n, { clock: true }), "resample: clock: true requires a clock Node as the driver");
  range(() => dropna(n, { how: "some" as any }), 'dropna: how must be "any" or "all"');
  // A description with every 0 is refused before any run (no trap).
  const d = new Pipeline([n], [resample(n, { every: 2 })]).toJSON() as any;
  d.nodes[1].params.every = 0;
  range(() => Pipeline.fromJSON(d), "resample: every must be positive");
  // A refused setting leaves nothing on the WASM heap.
  const M = current();
  for (let i = 0; i < 50; i++) try { resample(x, { every: 0 }); } catch { /* refused */ }
  const h0 = M.heapInUse();
  for (let i = 0; i < 200; i++) try { resample(x, { every: 0 }); } catch { /* refused */ }
  assert.ok(M.heapInUse() - h0 < 4096);
  // Aliases and synonyms, as Python resolves them.
  nanEq(resample(x, { every: 2, agg: "high" }).values as Float64Array, [1, 3, 4]);
  nanEq(resample(x, { every: 2, agg: "std" }).values as Float64Array, [Math.SQRT1_2, Math.SQRT1_2, NaN]);
});

test("3: a failed heap allocation is the out-of-memory RangeError", async () => {
  await ready();
  const M = current();
  const oom = (e: any) => e instanceof RangeError && e.message === "out of memory: the call needs more memory than is available";
  assert.throws(() => M.allocF64(2 ** 29), oom);
  assert.throws(() => M.allocF64(2 ** 32 + 2), oom);   // never wrapped to 2 doubles
  assert.throws(() => M.allocF64(2 ** 28 - 1 + 2 ** 27), oom);   // 3 GB: past the 2 GB heap
});

test("5: a one-input multi-output op gives per-event lists for number[]", async () => {
  await ready();
  const bb = BollingerBands(2)([1, 2, 3]) as number[][];
  assert.ok(Array.isArray(bb) && bb.length === 3 && Array.isArray(bb[0]) && bb[0].length === 3);
  nanEq(bb[2], [1.0857864376269049, 2.5, 3.914213562373095]);
  const mm = RollingMinMax(2)([1, 2, 3]) as number[][];
  assert.deepEqual(mm.slice(1), [[1, 2], [2, 3]]);
  // A Float64Array keeps the NdArray.
  assert.deepEqual((BollingerBands(2)(Float64Array.from([1, 2, 3])) as any).shape, [3, 3]);
});

test("8: the per-event call matches the batch call", async () => {
  await ready();
  const op = RollingMean(3), bb = BollingerBands(3);
  const xs = [1, 4, 2, 8, 5, 7];
  nanEq(xs.map((v) => op(v) as number), RollingMean(3)(Float64Array.from(xs)) as Float64Array);
  const rows = xs.map((v) => bb(v) as number[]);
  nanEq(rows.flat(), (BollingerBands(3)(Float64Array.from(xs)) as any).data);
});

test("10: a valueOf that grows memory never writes through a stale heap view", async () => {
  await ready();
  const grow = { valueOf() { RollingMean(2)(new Float64Array(4e6)); return 7; } };
  const x = Input("x");
  const p = new Pipeline([x], [select(x, [0, 1, 2])]);
  const live = p.live();
  live.push("x", 0, [1, grow, 3] as any);
  nanEq(((live.result() as any).values).data, [1, 7, 3]);
  live.dispose();
  const parts = split([1, grow, 3] as any, [0, 1, 0]);
  nanEq(parts[0].values as Float64Array, [1, 3]);
  nanEq(parts[1].values as Float64Array, [7]);
  const rows = [...combineLatest([[{ value: [1, 2], index: 0 }, { value: [3, grow], index: 1 }].values()] as any)] as any[];
  assert.deepEqual(rows.map((r) => Array.from(r.value)), [[1, 2], [3, 7]]);
});

test("11: a Proxy layout is copied before C++ reads it", async () => {
  await ready();
  const layout = new Proxy(["T", "T"], { get(t, k) { if (k === "0") { try { RollingMean(0); } catch { /* */ } } return (t as any)[k]; } });
  const r = combineLatest([Float64Array.from([1, 2]), Float64Array.from([3, 4])], { layout: layout as any });
  nanEq(r.values.data, [1, 3, 2, 4]);
});

test("12: strategyTest and positionTest refuse a text setting of another kind", async () => {
  await ready();
  const px = Float64Array.from({ length: 30 }, (_, i) => 100 + i);
  for (const k of ["method", "mode", "volumeMode"])
    assert.throws(() => strategyTest((c: any) => c, [px], { [k]: 5 } as any),
      (e: any) => e instanceof TypeError && e.message === `strategyTest: ${k} must be a string, got a number`);
  assert.throws(() => positionTest(BacktestPriceTarget(), px, [px], { method: 5 as any }),
    (e: any) => e instanceof TypeError && e.message === "positionTest: method must be a string, got a number");
});

test("13: Pipeline construction errors are RangeError / TypeError as in Python", async () => {
  await ready();
  const x = Input("x"), y = Input("y");
  const op = RollingMean(2);
  assert.throws(() => new Pipeline([op as any], [x]), RangeError);
  assert.throws(() => new Pipeline([x], [1 as any]), RangeError);
  assert.throws(() => new Pipeline([x], [RollingMean(2)(y)]), RangeError);
  assert.throws(() => new Pipeline([x, y], [RollingMean(2)(x)]), RangeError);
  const a = op(x);
  assert.throws(() => new Pipeline([x], [a, op(a)]), RangeError);
  const p = new Pipeline([x, y], [RollingMean(2)(x), RollingMean(2)(y)]);
  assert.throws(() => p({ x: [1] } as any), TypeError);
  assert.throws(() => p.live().push("z", 0, 1), RangeError);
  assert.throws(() => merge([x] as any), RangeError);
});

test("14: importing the package with --dump writes nothing", async () => {
  const { execFileSync } = await import("node:child_process");
  const out = execFileSync(process.execPath, ["--import", "tsx", "-e", 'await import("./src/index.ts")', "--", "--dump"],
    { cwd: new URL("..", import.meta.url), encoding: "utf8" });
  assert.equal(out, "");
});

test("15: a dimension past 2**32 - 1 is a RangeError", async () => {
  await ready();
  for (const shape of [[2 ** 40, 0], [0, 1e300]])
    assert.throws(() => RollingMean(2)({ data: new Float64Array(0), shape }), (e: any) => e instanceof RangeError && /above 2\*\*32 - 1/.test(e.message));
  assert.throws(() => PortfolioReport()({ data: new Float64Array(0), shape: [1e300, 0, 4] }), RangeError);
});

test("16: backtestReport checks the index length; MARKET is +inf", async () => {
  await ready();
  const rows = Array.from({ length: 5 }, () => [0, 0, 0, 0]);
  assert.throws(() => backtestReport(rows, { index: [0, 1] }),
    (e: any) => e instanceof RangeError && e.message === "backtestReport: the stream has 5 rows of values but 2 index values");
  assert.equal(backtestReport(rows, { index: [0, 1, 2, 3, 4] }).running.index.length, 5);
  assert.equal(MARKET, Infinity);
});

test("N5: stream function index errors name the function", async () => {
  await ready();
  const ev = (v: number, i: number) => ({ value: v, index: i });
  assert.throws(() => [...merge([[ev(1, 1), ev(2, 0)].values()])], (e: any) => e instanceof RangeError && e.message.startsWith("merge: "));
  assert.throws(() => [...merge([[ev(1, 0), { value: 2, index: "x" }].values()] as any)], (e: any) => e instanceof TypeError && e.message.startsWith("merge: "));
  assert.throws(() => [...combineLatest([[ev(1, 1), ev(2, 0)].values()])], (e: any) => e instanceof RangeError && e.message.startsWith("combineLatest: "));
  assert.throws(() => [...combineLatest([[ev(1, 0), ev(2, NaN)].values()])],
    (e: any) => e instanceof RangeError && e.message === "combineLatest: the index of input 0 is NaN at event 1; an index must be a number");
  assert.throws(() => combineLatest([{ values: [1, 2], index: [1, 0] }]), (e: any) => e instanceof RangeError && e.message.startsWith("combineLatest: "));
});
