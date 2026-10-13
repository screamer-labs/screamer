import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ready, circularShift, segmentShuffle, positionTest,
  BacktestOHLCTarget, BacktestPriceTarget, BacktestReport,
} from "../src/index.ts";

await ready();

const NaN_ = Number.NaN;
const X = [NaN_, NaN_, 1, 1, 1, 0, 0, -1, -1, -1, -1, 2, 0, 0, 1];

function col(a: { data: Float64Array; shape: number[] }, j: number): number[] {
  const [rows, cols] = a.shape;
  const out: number[] = [];
  for (let r = 0; r < rows; r++) out.push(a.data[r * cols + j]);
  return out;
}

// Same golden values as tests/test_validation.py::test_golden_values: the
// seeded surrogates must be identical in Python and JS/WASM.
test("validation: golden surrogates match Python", () => {
  assert.deepEqual(col(circularShift(X, 2, { minShift: 2, seed: 42 }), 0),
    [NaN_, NaN_, 2, 0, 0, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1]);
  assert.deepEqual(col(segmentShuffle(X, 2, { seed: 42 }), 1),
    [NaN_, NaN_, 0, 0, 1, 1, 1, -1, -1, -1, -1, 2, 0, 0, 1]);
});

test("validation: argument checks raise RangeError", () => {
  assert.throws(() => circularShift(X, 3, { minShift: 0 }), RangeError);
  assert.throws(() => circularShift(X, 3, { minShift: 7 }), RangeError);
  assert.throws(() => circularShift([NaN_, 1], 3), RangeError);
  assert.throws(() => segmentShuffle(X, 3, { seed: -1 }), RangeError);
});

// A deterministic market and a target that holds each side for 5 bars.
function market(T: number) {
  const o: number[] = [], h: number[] = [], l: number[] = [], c: number[] = [];
  let p = 100;
  for (let t = 0; t < T; t++) {
    const prev = p;
    p *= Math.exp(0.01 * Math.sin(1.7 * t) * Math.cos(0.31 * t * t));
    o.push(prev); c.push(p); h.push(Math.max(prev, p) * 1.001); l.push(Math.min(prev, p) * 0.999);
  }
  return { o, h, l, c };
}

test("validation: positionTest observed metric matches BacktestReport", () => {
  const { o, h, l, c } = market(300);
  const tgt = Array.from({ length: 300 }, (_, t) => (Math.floor(t / 5) % 3) - 1);
  const eng = BacktestOHLCTarget(2e-4);
  const res = positionTest(eng, tgt, [o, h, l, c], { n: 50, seed: 3 });
  const out = eng(Float64Array.from(tgt), o, h, l, c) as { data: Float64Array; shape: number[] };
  const rep = BacktestReport()(Float64Array.from(col(out, 0)), col(out, 1), col(out, 2), col(out, 3)) as
    { data: Float64Array; shape: number[] };
  const sharpe = col(rep, 5).filter(Number.isFinite).at(-1);
  assert.equal(res.observed, sharpe);
  assert.equal(res.null.length, 50);
  assert.ok(res.pValue >= 1 / 51 && res.pValue <= 1);
  assert.ok((res.minShift as number) >= 1);
  const again = positionTest(eng, tgt, [o, h, l, c], { n: 50, seed: 3 });
  assert.deepEqual(Array.from(again.null), Array.from(res.null));
});

test("validation: positionTest detects a planted edge", () => {
  const { o, h, l, c } = market(800);
  // Knows the next bar's direction: the engine executes the bar-t target on bar t+1.
  const tgt = c.map((_, t) => (t + 1 < c.length ? Math.sign(c[t + 1] - o[t + 1]) : 0));
  const res = positionTest(BacktestOHLCTarget(), tgt, [o, h, l, c],
    { n: 200, metric: "total_pnl", method: "segments" });
  assert.ok(res.pValue < 0.01, `p = ${res.pValue}`);
});

// Same golden values as tests/test_validation.py::test_permutation_golden_values.
test("validation: golden permutations match Python", async () => {
  const { permuteBars, permuteReturns } = await import("../src/index.ts");
  const o = [100, 101, 99, 102, 103, 101];
  const h = o.map((v) => v * 1.01), l = o.map((v) => v * 0.98);
  const c = [100.5, 99.5, 101, 102.5, 101.5, 102];
  const close = (a: number[], b: number[]) =>
    a.every((v, i) => (Number.isNaN(v) && Number.isNaN(b[i])) || Math.abs(v - b[i]) <= 1e-12 * Math.abs(b[i]));
  const bars = permuteBars(o, h, l, c, 2, { seed: 7 });
  assert.ok(close(col(bars[0], 1), [100.0, 101.00000000000003, 98.99999999999999,
    100.50246305418726, 101.99507389162568, 103.50738916256157]));
  assert.ok(close(col(bars[3], 1), [100.5, 99.49999999999997, 101.00000000000003,
    100.99512218680589, 103.00492610837439, 101.99999999999996]));
  const r = permuteReturns([NaN_, 10, 11, 10.5, 12, 11.5, 13], 2, { seed: 7 }) as
    { data: Float64Array; shape: number[] };
  assert.ok(close(col(r, 0), [NaN_, 10.0, 11.30434782608696, 10.833333333333336,
    12.380952380952385, 11.81818181818182, 13.0]));
});

test("validation: strategyTest separates an informative feature from noise", async () => {
  const { strategyTest, BacktestPriceTarget } = await import("../src/index.ts");
  // AR(1) returns with a deterministic pseudo-random driver.
  let seed = 12345;
  const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 - 0.5; };
  const T = 1200, price: number[] = [], feat: number[] = [], noise: number[] = [];
  let r = 0, p = 100, ew = 0;
  for (let t = 0; t < T; t++) {
    r = 0.3 * r + 0.02 * rnd();
    p *= Math.exp(r);
    price.push(p);
    ew = ew + (2 / 11) * (r - ew);
    feat.push(ew);
    noise.push(rnd());
  }
  const strat = (f: Float64Array, px: Float64Array) =>
    BacktestPriceTarget()(Array.from(f, Math.sign), px);
  const good = strategyTest(strat, [feat, price], { n: 200, method: "shift", columns: [0] });
  const bad = strategyTest(strat, [noise, price], { n: 200, method: "shift", columns: [0] });
  assert.ok(good.pValue < 0.05, `informative p = ${good.pValue}`);
  assert.ok(bad.pValue > 0.05, `noise p = ${bad.pValue}`);
  const bars = strategyTest((px: Float64Array) =>
      BacktestPriceTarget()(Array.from(px, () => 1), px), [price],
    { n: 20, method: "returns" });
  assert.equal(bars.null.length, 20);
});

test("validation: positionTest family matches the single-strategy null for K = 1", () => {
  const { o, h, l, c } = market(300);
  const tgt = Array.from({ length: 300 }, (_, t) => (Math.floor(t / 5) % 3) - 1);
  const eng = BacktestOHLCTarget();
  const one = positionTest(eng, tgt, [o, h, l, c], { n: 40, seed: 2 });
  const fam = positionTest(eng, [tgt], [o, h, l, c], { n: 40, seed: 2 });
  assert.deepEqual(Array.from(fam.null.data), Array.from(one.null));
  assert.equal(fam.familyPValue[0], one.pValue);
  const tgt2 = tgt.map((v) => -v);
  const two = positionTest(eng, [tgt, tgt2], [o, h, l, c], { n: 40, seed: 2 });
  for (let k = 0; k < 2; k++) assert.ok(two.familyPValue[k] >= two.pValue[k]);
});

// Same values as Python: position_test(..., n=50, seed=3, periods_per_year=252).
test("validation: per-run statistics match Python and line up with null", async () => {
  const { strategyTest } = await import("../src/index.ts");
  const { o, h, l, c } = market(300);
  const tgt = Array.from({ length: 300 }, (_, t) => (Math.floor(t / 5) % 3) - 1);
  const eng = BacktestOHLCTarget(2e-4);
  const res = positionTest(eng, tgt, [o, h, l, c], { n: 50, seed: 3, periodsPerYear: 252 });
  const near = (a: number, b: number) => Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(b));
  assert.ok(near(res.observedStats.win_rate, 0.4358974358974359));
  assert.ok(near(res.observedStats.sortino, -0.10425691757084403));
  assert.ok(near(res.nullStats.calmar[0], -0.21490573138656252));
  assert.ok(near(res.nullStats.win_rate[1], 0.5384615384615384));
  assert.deepEqual(Array.from(res.nullStats.round_trips.slice(0, 3)), [40, 39, 39]);
  assert.equal(res.observed, res.observedStats.sharpe);
  assert.deepEqual(Array.from(res.nullStats.sharpe).filter(Number.isFinite), Array.from(res.null));
  assert.equal(Object.keys(res.nullStats).length, 15);

  const wr = positionTest(eng, tgt, [o, h, l, c], { n: 20, metric: "win_rate" });
  assert.equal(wr.observed, wr.observedStats.win_rate);
  assert.throws(() => positionTest(eng, tgt, [o, h, l, c], { n: 5, metric: "turnover" }), RangeError);
  positionTest(eng, tgt, [o, h, l, c], { n: 5, metric: "turnover", largerIsBetter: false });

  const fam = positionTest(eng, [tgt, tgt.map((v) => -v)], [o, h, l, c], { n: 10 });
  assert.deepEqual(fam.nullStats.sortino.shape, [10, 2]);
  assert.equal(fam.observedStats.sortino.length, 2);
  assert.deepEqual(Array.from(fam.nullStats.sharpe.data), Array.from(fam.null.data));

  const st = strategyTest((...x: Float64Array[]) => BacktestOHLCTarget()(tgt, ...x), [o, h, l, c],
    { n: 8, metric: "calmar" });
  assert.equal(st.nullStats.calmar.length, 8);
  assert.equal(st.observed, st.observedStats.calmar);
});

// Same values as Python (screamer.probabilistic_sharpe / deflated_sharpe).
test("validation: probabilistic and deflated Sharpe match Python", async () => {
  const { probabilisticSharpe, deflatedSharpe } = await import("../src/index.ts");
  const x = [0.01, -0.02, 0.015, 0.003, -0.004, 0.02, 0.011, -0.006, 0.004, 0.009];
  const near = (a: number, b: number) => Math.abs(a - b) <= 1e-12 * Math.abs(b);
  assert.ok(near(probabilisticSharpe(x), 0.8256797972532145));
  assert.ok(near(probabilisticSharpe(x, 0.1), 0.7506739171464867));
  const d = deflatedSharpe(x, [0.1, 0.3, -0.2, 0.05]);
  assert.ok(near(d.probability, 0.6455505503887139) && near(d.benchmark, 0.21636840436103524));
  assert.throws(() => probabilisticSharpe([0.1, 0.2]), RangeError);
  assert.throws(() => deflatedSharpe(x, [0.1]), RangeError);
});


// Same golden values as tests/test_validation.py::test_mode_golden_values.
test("validation: permutation modes and notional volume match Python", async () => {
  const { permuteBars } = await import("../src/index.ts");
  const o = [100, 101, 99, 102, 103, 101];
  const h = o.map((v) => v * 1.01), l = o.map((v) => v * 0.98);
  const c = [100.5, 99.5, 101, 102.5, 101.5, 102], v = [5, 3, 8, 2, 6, 4];
  const near = (a: number[], b: number[]) => a.every((x, i) => Math.abs(x - b[i]) <= 1e-12 * Math.max(1, Math.abs(b[i])));
  const m100 = (a: number[]) => a.map((x) => x - 100);
  const d = permuteBars(m100(o), m100(h), m100(l), m100(c), 2, { mode: "diff", seed: 7 });
  assert.ok(near(col(d[3], 1), [0.5, -0.5, 1.0, 1.0, 3.0, 2.0]));
  const lv = permuteBars(o, h, l, c, 2, { mode: "level", seed: 7 });
  assert.deepEqual(col(lv[3], 0), [101.5, 102.5, 101.0, 99.5, 100.5, 102.0]);
  const nv = permuteBars(o, h, l, c, 2, { seed: 7, volume: v, volumeMode: "notional" });
  assert.equal(nv.length, 5);
  assert.ok(near(col(nv[4], 1), [5.0, 3.0, 8.0, 2.029800999901968, 3.9609756097560966, 5.97058823529412]));
  assert.throws(() => permuteBars(m100(o), m100(h), m100(l), m100(c), 2), RangeError);
});

test("validation: positionTest calls the engine once per chunk with a (T, k) block", () => {
  const engine = BacktestPriceTarget();
  let calls = 0;
  const counted: any = (...a: any[]) => { calls++; return (engine as any)(...a); };
  counted.reset = () => engine.reset();
  const T = 300;
  const price = Float64Array.from({ length: T }, (_, t) => 100 + Math.sin(t / 9) * 5 + t / 50);
  const target = Float64Array.from({ length: T }, (_, t) => (Math.sin(t / 13) > 0 ? 1 : -1));
  const r = positionTest(counted, target, [price], { n: 50, seed: 1 });
  assert.equal(r.null.length > 0, true);
  assert.equal(calls, 2);        // the observed run and one chunk of 50
});

// C2: strategyTest on (T, K) panels. Same values as Python:
// strategy_test(book, c, method="returns", n=5, seed=1) and the "bars" run with
// mode="diff", where book is an EwMean(span=4) sign book on BacktestPortfolioTarget.
test("validation: strategyTest on (T, K) panels matches Python", async () => {
  const { strategyTest, EwMean, LogReturn, BacktestPortfolioTarget } = await import("../src/index.ts");
  const T = 120, K = 3;
  const rows = (f: (t: number, k: number) => number) =>
    Array.from({ length: T }, (_, t) => Array.from({ length: K }, (_, k) => f(t, k)));
  const c = rows((t, k) => 100 * Math.exp(0.02 * Math.sin(0.21 * t * (k + 1)) + 0.001 * t * (k - 1)));
  const o = rows((t, k) => c[Math.max(0, t - 1)][k]);
  const h = rows((t, k) => Math.max(o[t][k], c[t][k]) * 1.001);
  const l = rows((t, k) => Math.min(o[t][k], c[t][k]) * 0.999);
  const shapes: number[][] = [];
  const book = (px: any) => {
    shapes.push(px.shape);
    const sig = EwMean(Number.NaN, 4)(LogReturn()(px)) as any;
    const w = Float64Array.from(sig.data as Float64Array, (v) => (Number.isNaN(v) ? 0 : Math.sign(v)) / K);
    return BacktestPortfolioTarget()({ data: w, shape: [T, K] }, px);
  };
  const near = (a: ArrayLike<number>, b: number[]) =>
    a.length === b.length && b.every((v, i) => Math.abs(a[i] - v) <= 1e-12 * Math.max(1, Math.abs(v)));
  const r = strategyTest(book, [c], { method: "returns", n: 5, seed: 1 });
  assert.ok(near([r.observed], [0.9256864022681778]), `${r.observed}`);
  assert.equal(r.pValue, 0.16666666666666666);
  assert.ok(near(r.null, [0.09176636863855668, 0.05996838391218732, 0.08515585398159334,
    0.12065525318962493, -0.05150083773850076]), `${Array.from(r.null)}`);
  assert.ok(shapes.every((s) => s[0] === T && s[1] === K));
  const b = strategyTest((_o: any, _h: any, _l: any, px: any) => book(px), [o, h, l, c],
    { n: 5, seed: 1, mode: "diff" });
  assert.ok(near(b.null, [0.08795893015155552, 0.045383667637995555, 0.09102912413542846,
    0.11555392420276887, -0.05560480824609657]), `${Array.from(b.null)}`);
});

test("validation: strategyTest applies one permutation to every asset", async () => {
  const { strategyTest, BacktestPortfolioTarget } = await import("../src/index.ts");
  const T = 80;
  const x = Array.from({ length: T }, (_, t) => 100 * Math.exp(0.01 * Math.sin(0.3 * t) + 0.002 * t));
  const panel = { data: Float64Array.from(x.flatMap((v) => [v, v])), shape: [T, 2] };
  const feature = Float64Array.from({ length: T }, (_, t) => t);
  for (const mode of ["log", "diff", "level"] as const) {
    const seen: any[] = [];
    strategyTest((px: any, f: Float64Array) => {
      seen.push([px, f]);
      return BacktestPortfolioTarget()({ data: new Float64Array(2 * T).fill(0.5), shape: [T, 2] }, px);
    }, [panel, feature], { method: "returns", mode, n: 4, seed: 2 });
    for (const [px, f] of seen.slice(1)) {
      for (let t = 0; t < T; t++) assert.equal(px.data[2 * t], px.data[2 * t + 1]);
      assert.ok(f instanceof Float64Array && f.length === T);
    }
    assert.notDeepEqual(Array.from(seen[1][0].data), Array.from(panel.data));
  }
  const book = (...xs: any[]) =>
    BacktestPortfolioTarget()({ data: new Float64Array(2 * T).fill(0.5), shape: [T, 2] }, xs[0]);
  assert.throws(() => strategyTest(book, [panel, x.slice(1)], { method: "returns", n: 2 }), RangeError);
  assert.throws(() => strategyTest(book, [{ data: panel.data, shape: [T, 2, 1] }], { method: "returns", n: 2 }), RangeError);
  assert.throws(() => strategyTest(book, ["abc" as any], { method: "returns", n: 2 }), TypeError);
  assert.throws(() => strategyTest(book, [panel, panel, x, panel], { n: 2 }), RangeError);
});

test("validation: permute errors come from C++ as RangeError; panels are rejected", async () => {
  const { strategyTest, permuteReturns, permuteBars } = await import("../src/index.ts");
  const T = 30;
  const x = Array.from({ length: T }, (_, t) => 100 + t);
  // errors raised by the C++ kernels, with their text
  assert.throws(() => permuteReturns([1, 2, NaN_, 4], 2),
    { name: "RangeError", message: "permute_returns needs finite values after the NaN warmup." });
  assert.throws(() => permuteReturns([1, -2, 3], 2),
    { name: "RangeError", message: 'permute_returns with mode="log" needs positive values; use mode="diff".' });
  assert.throws(() => permuteReturns([NaN_, 1], 2),
    { name: "RangeError", message: "permute_returns needs at least two values after the NaN warmup." });
  assert.throws(() => permuteBars([1], [1], [1], [1], 2),
    { name: "RangeError", message: "permute_bars needs at least two bars." });
  assert.throws(() => permuteBars([1, NaN_], [1, 1], [1, 1], [1, 1], 2),
    { name: "RangeError", message: "permute_bars needs finite open, high, low, and close prices." });
  // K = 0 panels
  const book = () => ({ data: new Float64Array(T * 4), shape: [T, 4] });
  const empty = { data: new Float64Array(0), shape: [T, 0] };
  assert.throws(() => strategyTest(book, [empty], { method: "returns", n: 2 }),
    { name: "RangeError", message: "permute_returns needs at least one asset." });
  assert.throws(() => strategyTest(book, [empty, empty, empty, empty], { n: 2 }),
    { name: "RangeError", message: "permute_bars needs at least one asset." });
  // many caught errors in a row leave the module usable
  for (let i = 0; i < 2000; i++) assert.throws(() => permuteReturns([1, 2, NaN_, 4], 1), RangeError);
  assert.equal(permuteReturns(x, 2).shape[1], 2);
  // the public permute functions take one series, as in Python
  const panel = { data: Float64Array.from(x.flatMap((v) => [v, v])), shape: [T, 2] };
  const rows = x.map((v) => [v, v]);
  assert.throws(() => permuteReturns(panel, 2), { name: "RangeError", message: "x must be a 1-D series" });
  assert.throws(() => permuteReturns(rows as any, 2), { name: "RangeError", message: "x must be a 1-D series" });
  assert.throws(() => permuteBars(panel, panel, panel, panel, 2),
    { name: "RangeError", message: "open must be a 1-D series" });
  assert.throws(() => permuteBars(rows as any, x, x, x, 2),
    { name: "RangeError", message: "open must be a 1-D series" });
  assert.throws(() => permuteReturns(x, 2, { volume: rows as any }),
    { name: "RangeError", message: "volume must be a 1-D series" });
  // a rank-1 NdArray is a series
  const one = { data: Float64Array.from(x), shape: [T] };
  assert.deepEqual(Array.from((permuteReturns(one, 2, { seed: 1 }) as any).data),
                   Array.from((permuteReturns(x, 2, { seed: 1 }) as any).data));
});

// B5: strategyTest runs in the shared C++ driver; the strategy is a callback.
test("validation: strategyTest callback errors reach the caller unchanged", async () => {
  const { strategyTest, BacktestPriceTarget, EwMean } = await import("../src/index.ts");
  const T = 60;
  const price = Float64Array.from({ length: T }, (_, t) => 100 + Math.sin(t / 5) + t / 10);
  const hold = (p: Float64Array) => BacktestPriceTarget()(new Float64Array(T).fill(1), p);
  class StrategyFailed extends Error {}
  const thrown = new StrategyFailed("from the strategy");
  let calls = 0;
  const failsLate = (p: Float64Array) => { if (++calls === 3) throw thrown; return hold(p); };
  assert.throws(() => strategyTest(failsLate, [price], { method: "returns", n: 5 }), (e) => e === thrown);
  assert.equal(calls, 3);
  const notAnError = { code: 7 };
  assert.throws(() => strategyTest(() => { throw notAnError; }, [price], { method: "returns", n: 2 }),
    (e) => e === notAnError);
  assert.throws(() => strategyTest(hold, [price], { method: "returns", n: 2,
    metric: () => { throw new EvalError("from the metric"); } }), { name: "EvalError", message: "from the metric" });
  // a C++ error inside the strategy (an invalid operator) is a RangeError as usual
  assert.throws(() => strategyTest((p: Float64Array) => EwMean(NaN, -1)(p), [price], { method: "returns", n: 2 }),
    RangeError);

  // A strategy that catches an error from WASM on every run: the driver's own
  // frames survive, so the result equals the run without the caught errors.
  const ref = strategyTest(hold, [price], { method: "returns", n: 20, seed: 4 });
  const catching = (p: Float64Array) => {
    assert.throws(() => EwMean(NaN, -1)(p), RangeError);
    return hold(p);
  };
  for (let i = 0; i < 3; i++) {
    const got = strategyTest(catching, [price], { method: "returns", n: 20, seed: 4 });
    assert.deepEqual(Array.from(got.null), Array.from(ref.null));
    assert.deepEqual(Array.from(got.nullStats.sharpe), Array.from(ref.nullStats.sharpe));
  }
  // many failed tests in a row leave the module usable
  for (let i = 0; i < 2000; i++) {
    let k = 0;
    assert.throws(() => strategyTest((p: Float64Array) => { if (++k === 2) throw thrown; return hold(p); },
      [price], { method: "returns", n: 3 }), (e) => e === thrown);
  }
  const again = strategyTest(hold, [price], { method: "returns", n: 20, seed: 4 });
  assert.deepEqual(Array.from(again.null), Array.from(ref.null));
});

test("validation: strategyTest gives each run new input arrays", async () => {
  const { strategyTest, BacktestPriceTarget } = await import("../src/index.ts");
  const T = 40;
  const price = Float64Array.from({ length: T }, (_, t) => 100 + t + Math.sin(t));
  const feature = Float64Array.from({ length: T }, (_, t) => t);
  const seen: Float64Array[][] = [];
  strategyTest((p: Float64Array, f: Float64Array) => {
    seen.push([p, f]);
    return BacktestPriceTarget()(new Float64Array(T).fill(1), p);
  }, [price, feature], { method: "returns", n: 3, seed: 1 });
  assert.deepEqual(Array.from(seen[0][1]), Array.from(feature));   // the real run
  for (const i of [0, 1]) {
    assert.notEqual(seen[1][i].buffer, seen[2][i].buffer);
    assert.notDeepEqual(Array.from(seen[1][i]), Array.from(seen[2][i]));
  }
});

// B5 fix round 1: optional options given as null, the range of n, the
// statistic table from C++, and the only path for C++ to call JavaScript.
test("validation: strategyTest treats null options as not given", async () => {
  const { strategyTest, BacktestPriceTarget } = await import("../src/index.ts");
  const T = 50;
  const c = Float64Array.from({ length: T }, (_, t) => 100 + t + Math.sin(t));
  const o = c.map((x) => x * 0.999), h = c.map((x) => x * 1.01), l = c.map((x) => x * 0.98);
  const hold = (p: Float64Array) => BacktestPriceTarget()(new Float64Array(T).fill(1), p);
  const bars = (_o: any, _h: any, _l: any, p: Float64Array) => hold(p);
  const same = (a: any, b: any) => assert.deepEqual(Array.from(a.null), Array.from(b.null));
  same(strategyTest(bars, [o, h, l, c], { n: 6, volume: null as any }), strategyTest(bars, [o, h, l, c], { n: 6 }));
  const shift = (_f: any, p: Float64Array) => hold(p);
  same(strategyTest(shift, [c, c], { method: "shift", columns: [0], n: 6, minShift: null as any }),
       strategyTest(shift, [c, c], { method: "shift", columns: [0], n: 6 }));
  same(strategyTest(hold, [c], { method: "returns", n: null as any, seed: null as any, mode: null as any,
                                 price: null as any, fixed: null as any, periodsPerYear: null as any }),
       strategyTest(hold, [c], { method: "returns", n: 500 }));
});

test("validation: strategyTest checks the range of n before allocating", async () => {
  const { strategyTest, BacktestPriceTarget } = await import("../src/index.ts");
  const T = 30;
  const c = Float64Array.from({ length: T }, (_, t) => 100 + t);
  const hold = (p: Float64Array) => BacktestPriceTarget()(new Float64Array(T).fill(1), p);
  assert.throws(() => strategyTest(hold, [c], { method: "returns", n: 0 }),
    { name: "RangeError", message: "n must be at least 1" });
  // The results stream into JS arrays, so the heap does not bound n: each
  // result array holds n values, at most 2^31 - 1.
  const { current } = await import("../src/index.ts");
  const max = (current() as any).strategyTestMaxRuns() as number;
  assert.equal(max, 2 ** 31 - 1);
  for (const n of [max + 1, 2 ** 32 + 3, 2 ** 60, Number.MAX_VALUE, Infinity])
    assert.throws(() => strategyTest(hold, [c], { method: "returns", n }),
      { name: "RangeError", message: `n is too large: at most ${max} runs` });
  assert.equal(strategyTest(hold, [c], { method: "returns", n: 3 }).null.length, 3);
});

test("validation: positionTest overloads type each result (tsc)", async () => {
  const ts = (await import("typescript")).default;
  const path = await import("node:path");
  const file = path.resolve(import.meta.dirname, "types/position_test.types.ts");
  const program = ts.createProgram([file], {
    strict: true, noEmit: true, target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext,
    moduleResolution: ts.ModuleResolutionKind.Bundler, allowImportingTsExtensions: true,
    types: ["node"], lib: ["lib.es2022.d.ts", "lib.esnext.disposable.d.ts"],
  });
  const errors = ts.getPreEmitDiagnostics(program)
    .map((d) => ts.flattenDiagnosticMessageText(d.messageText, "\n"));
  assert.deepEqual(errors, []);
});

test("validation: the statistic table comes from C++", async () => {
  const { current, strategyTest } = await import("../src/index.ts");
  const rows = (current() as any).strategyStats() as [string, string, number, number][];
  assert.deepEqual(rows.map((r) => r[0]), ["total_pnl", "max_drawdown", "total_cost", "turnover",
    "num_trades", "sharpe", "exposure", "round_trips", "win_rate", "profit_factor", "avg_win",
    "avg_loss", "avg_holding", "sortino", "calmar"]);
  assert.deepEqual(rows[5], ["sharpe", "report", 5, 1]);
  assert.deepEqual(rows[3], ["turnover", "report", 2, -1]);
  assert.deepEqual(rows[2], ["total_cost", "report", 1, 0]);
  const res = strategyTest((p: Float64Array) => BacktestPriceTarget()(Array.from(p, () => 1), p),
    [[1, 2, 3, 4, 5]], { method: "returns", n: 2 });
  assert.deepEqual(Object.keys(res.nullStats), rows.map((r) => r[0]));
  // an inherited property name is not a statistic; C++ checks the name
  for (const metric of ["toString", "constructor", "__proto__"])
    assert.throws(() => strategyTest(() => null, [[1, 2, 3]], { metric: metric as any }),
      (e: any) => e instanceof RangeError && e.message.startsWith("metric must be one of ['avg_holding'"));
});

test("validation: C++ calls JavaScript only through wasmCallbacks", async () => {
  const fs = await import("node:fs");
  const path = await import("node:path");
  const root = path.resolve(import.meta.dirname, "../..");
  const cpp = fs.readdirSync(path.join(root, "wasm"))
    .filter((f) => /\.(cpp|h)$/.test(f))
    .map((f) => [f, fs.readFileSync(path.join(root, "wasm", f), "utf8")] as const);
  // Calls of a JS method from C++: the one in JsCallbacks::call, which only
  // receives a wasmCallbacks object, and Array.push on a result array.
  const allowed = new Set(["validation_bindings.cpp: cb_.call<int>(name", 'shape_bindings.cpp: a.call<void>("push"']);
  for (const [f, src] of cpp)
    for (const m of src.matchAll(/(\w+)\.call<(\w+)>\(([^,)]*)/g))
      assert.ok(allowed.has(`${f}: ${m[1]}.call<${m[2]}>(${m[3]}`), `${f}: C++ calls JavaScript: ${m[0]}`);
  // Every function (or constructor) that takes an emscripten::val, whatever
  // its return type or parameter name, is classified here, so a new one fails
  // this test until it is listed. Callback takers are exported and must get
  // their object from wasmCallbacks(...).call; holders keep it inside C++.
  const takers = ["strategyTestRun", "positionTestRun", "walkForwardRun"];
  const holders = ["JsCallbacks", "JsStrategyHost", "JsEngineHost"];
  const values = ["eventPlan", "layoutPlan", "permuteBarsInto", "permuteReturnsInto", "positions",
    "read_heads", "read_layouts", "combineLatestBatch", "JoinStream"];
  for (const [f, src] of cpp)
    assert.ok(!/using\s+(namespace\s+emscripten|emscripten::val)\b/.test(src), `${f}: val must be spelled emscripten::val`);
  const found = cpp.flatMap(([, src]) => [...src.matchAll(
    /(\w+)\s*\(([^()]*\bemscripten::val\b[^()]*)\)\s*(?:const\s*)?(?::[^{;]*)?\{/g)].map((m) => m[1]));
  assert.deepEqual([...found].sort(), [...takers, ...holders, ...values].sort());
  const js = fs.readdirSync(path.join(root, "js/src")).filter((f) => f.endsWith(".ts"))
    .map((f) => fs.readFileSync(path.join(root, "js/src", f), "utf8")).join("\n");
  for (const name of takers) {
    const uses = [...js.matchAll(new RegExp(`(.{0,30})M\\.${name}\\(`, "g"))];
    assert.ok(uses.length > 0);
    for (const u of uses) assert.match(u[1], /\.call\(\(cb\) => $/, `M.${name} must be called inside wasmCallbacks(...).call`);
  }
  assert.ok(!/export function callbackScope/.test(js), "callbackScope must stay private to errors.ts");
});

// B6: positionTest and walkForward run in the shared C++ driver.
test("validation: positionTest and walkForward return the user's errors unchanged", async () => {
  const { walkForward, EwMean, current } = await import("../src/index.ts");
  const T = 120;
  const { c } = market(T);
  const tgt = Float64Array.from({ length: T }, (_, t) => (Math.floor(t / 7) % 2 ? 1 : -1));
  const neg = Array.from(tgt, (v) => -v);
  const engineThatFails = (failOn: number, err: unknown) => {
    const eng = BacktestPriceTarget(1e-4);
    let k = 0;
    const f: any = (...a: any[]) => { if (++k === failOn) throw err; return (eng as any)(...a); };
    f.reset = () => eng.reset();
    return f;
  };
  class EngineFailed extends Error {}
  for (const failOn of [1, 2]) {
    const err = new EngineFailed("engine");
    assert.throws(() => positionTest(engineThatFails(failOn, err), tgt, [c], { n: 5 }), (e) => e === err);
    assert.throws(() => walkForward(engineThatFails(failOn, err), [tgt, neg], [c], { train: 50, test: 10 }),
      (e) => e === err);
  }
  const notAnError = { code: 7 };
  assert.throws(() => positionTest(engineThatFails(2, notAnError), [tgt, neg], [c], { n: 3 }), (e) => e === notAnError);
  let calls = 0;
  const failing = (out: any) => { if (++calls === 2) throw new EvalError("from the metric"); return out.data[0]; };
  assert.throws(() => positionTest(BacktestPriceTarget(), tgt, [c], { n: 3, metric: failing }),
    { name: "EvalError", message: "from the metric" });
  assert.throws(() => walkForward(BacktestPriceTarget(), [tgt, neg], [c], { train: 50, test: 10,
    metric: () => { throw notAnError; } }), (e) => e === notAnError);
  // an engine that catches an error from WASM on every run leaves the driver intact
  const ref = positionTest(BacktestPriceTarget(), tgt, [c], { n: 20, seed: 4 });
  const eng = BacktestPriceTarget();
  const catching: any = (...a: any[]) => { assert.throws(() => EwMean(NaN, -1)(c), RangeError); return (eng as any)(...a); };
  catching.reset = () => eng.reset();
  const got = positionTest(catching, tgt, [c], { n: 20, seed: 4 });
  assert.deepEqual(Array.from(got.null), Array.from(ref.null));
  // many failed tests in a row leave the stack and the module as they were
  const M = current() as any;
  const sp = M.stackSave();
  for (let i = 0; i < 2000; i++) {
    const err = new EngineFailed("again");
    assert.throws(() => positionTest(engineThatFails(2, err), tgt, [c], { n: 3 }), (e) => e === err);
  }
  assert.equal(M.stackSave(), sp);
  assert.deepEqual(Array.from(positionTest(BacktestPriceTarget(), tgt, [c], { n: 20, seed: 4 }).null),
    Array.from(ref.null));
});

test("validation: positionTest checks the range of n in C++ before allocating", async () => {
  const { current } = await import("../src/index.ts");
  const max = (current() as any).strategyTestMaxRuns() as number;
  const T = 60;
  const { c } = market(T);
  const tgt = Array.from({ length: T }, (_, t) => (Math.floor(t / 5) % 2 ? 1 : -1));
  for (const n of [0, -3, 2.5, NaN, -Infinity])
    assert.throws(() => positionTest(BacktestPriceTarget(), tgt, [c], { n }),
      { name: "RangeError", message: "n must be at least 1" });
  for (const n of [max + 1, 2 ** 32 + 3, 2 ** 60, Number.MAX_VALUE, Infinity])
    assert.throws(() => positionTest(BacktestPriceTarget(), tgt, [c], { n }),
      { name: "RangeError", message: `n is too large: at most ${max} runs` });
  // K strategies keep K values per run
  assert.throws(() => positionTest(BacktestPriceTarget(), [tgt, tgt], [c], { n: max }),
    { name: "RangeError", message: `n is too large: at most ${Math.floor(max / 2)} runs` });
  assert.equal(positionTest(BacktestPriceTarget(), tgt, [c], { n: 3 }).nullStats.sharpe.length, 3);
});

test("validation: positionTest and walkForward treat null options as not given", async () => {
  const { walkForward } = await import("../src/index.ts");
  const T = 80;
  const { c } = market(T);
  const tgt = Array.from({ length: T }, (_, t) => (Math.floor(t / 5) % 2 ? 1 : -1));
  const nulls: any = { n: 30, method: null, metric: null, largerIsBetter: null, minShift: null, seed: null,
    periodsPerYear: null };
  const a = positionTest(BacktestPriceTarget(), tgt, [c], nulls);
  const b = positionTest(BacktestPriceTarget(), tgt, [c], { n: 30 });
  assert.deepEqual(a, b);
  const X = [tgt, tgt.map((v) => -v)];
  const w = walkForward(BacktestPriceTarget(), X, [c], { train: 30, test: 10, metric: null,
    largerIsBetter: null, anchored: null, periodsPerYear: null } as any);
  assert.deepEqual(w, walkForward(BacktestPriceTarget(), X, [c], { train: 30, test: 10 }));
});

test("validation: positionTest and walkForward errors use the Python text", async () => {
  const { walkForward, BacktestPortfolioTarget } = await import("../src/index.ts");
  const T = 60;
  const { c } = market(T);
  const tgt = Array.from({ length: T }, (_, t) => (Math.floor(t / 5) % 2 ? 1 : -1));
  const pan = Array.from({ length: T }, () => [1, 1]);
  const msg = (f: () => unknown, message: string) => assert.throws(f, { name: "RangeError", message });
  msg(() => positionTest(BacktestPriceTarget(), tgt, [c], { seed: -1 }), "seed must be an integer in [0, 2**53)");
  msg(() => positionTest(BacktestPriceTarget(), [tgt, tgt], [c], { method: "segments" }),
    'several strategies need method="shift", which shifts them jointly');
  msg(() => positionTest(BacktestPortfolioTarget(), pan, [pan], { method: "segments" }),
    'a portfolio is tested with method="shift", which moves all assets together');
  msg(() => positionTest(BacktestPortfolioTarget(), pan, [c]), "target and market inputs must all have shape (T, K)");
  msg(() => positionTest(BacktestPriceTarget(), tgt, [pan]), "market input must be a 1-D series");
  msg(() => positionTest(BacktestPriceTarget(), tgt, [c], { metric: "turnover" }),
    'metric "turnover" has no natural direction; pass larger_is_better');
  msg(() => walkForward(BacktestPortfolioTarget(), pan, [pan], { train: 10, test: 10 }),
    "walk_forward needs a single-asset engine; portfolio engines are not supported");
  const bad: any = () => ({ data: new Float64Array(3 * T), shape: [T, 3] });
  bad.reset = () => {};
  msg(() => positionTest(bad, tgt, [c], { n: 2 }),
    "engine must return a backtest output of shape (T, 4) for a (T,) target, or (T, k, 4) for a (T, k) target");
});

test("validation: positionTest takes NdArray targets like arrays", () => {
  const T = 90;
  const { c } = market(T);
  const a = Array.from({ length: T }, (_, t) => (Math.floor(t / 5) % 2 ? 1 : -1));
  const b = a.map((v, t) => (t % 7 ? v : -v));
  const one = positionTest(BacktestPriceTarget(), a, [c], { n: 12 });
  assert.deepEqual(positionTest(BacktestPriceTarget(), { data: Float64Array.from(a), shape: [T] }, [c], { n: 12 }), one);
  const fam = positionTest(BacktestPriceTarget(), [a, b], [c], { n: 12 });
  const block = Float64Array.from({ length: 2 * T }, (_, i) => (i % 2 ? b : a)[i >> 1]);
  assert.deepEqual(positionTest(BacktestPriceTarget(), { data: block, shape: [T, 2] } as any, [c], { n: 12 }), fam);
});

test("validation: a custom metric must give one number per strategy", async () => {
  const { walkForward, strategyTest, BacktestPortfolioTarget } = await import("../src/index.ts");
  const T = 80;
  const { c } = market(T);
  const tgt = Array.from({ length: T }, (_, t) => (Math.floor(t / 5) % 2 ? 1 : -1));
  const hold = (p: Float64Array) => BacktestPriceTarget()(new Float64Array(T).fill(1), p);
  const pan = c.map((v) => [v, v]), w = tgt.map((v) => [v / 2, -v / 2]);
  const wrong = /^metric must return one value per strategy in its input: 1 expected, got (2|0)$/;
  const calls: ((m: any) => unknown)[] = [
    (m) => positionTest(BacktestPriceTarget(), tgt, [c], { n: 3, metric: m }),
    (m) => positionTest(BacktestPriceTarget(), [tgt, tgt], [c], { n: 3, metric: m }),
    (m) => positionTest(BacktestPortfolioTarget(), w, [pan], { n: 3, metric: m }),
    (m) => walkForward(BacktestPriceTarget(), [tgt, tgt], [c], { train: 30, test: 10, metric: m }),
    (m) => strategyTest(hold, [c], { method: "returns", n: 3, metric: m }),
  ];
  for (const run of calls) {
    for (const bad of [[1, 2], [], new Float64Array(2)])
      assert.throws(() => run(() => bad), (e: any) => e instanceof RangeError && wrong.test(e.message));
    // Python accepts and rejects the same kinds (test_validation.py)
    for (const bad of [undefined, null, "1", { x: 1 }, [1, "2"], true, [true], ["1"], [[0.5]],
                       { data: Float64Array.of(1), shape: [1] }, new BigInt64Array(1)])
      assert.throws(() => run(() => bad),
        { name: "TypeError", message: "metric must return a number or a 1-D array of numbers" });
    for (const good of [0.5, [0.5], Float64Array.of(0.5), Int32Array.of(1)]) run(() => good);
    const one = run(() => [0.5]) as any, num = run(() => 0.5) as any;
    assert.deepEqual(one, num);
  }
});

test("validation: the family p-value is the max statistic over the null", () => {
  const T = 200;
  const { c } = market(T);
  const a = Array.from({ length: T }, (_, t) => (Math.floor(t / 6) % 2 ? 1 : -1));
  const b = a.map((v, t) => (t % 5 ? v : -v));
  const flat = new Array(T).fill(0), none = new Array(T).fill(NaN);
  none[0] = 1;   // a column whose metric is NaN in every round
  const cases: [number[][], any][] = [
    [[a, b, flat], { metric: "total_pnl" }],
    [[a, b, flat, none], { metric: "sharpe", largerIsBetter: false }],
    // profit_factor is +inf for a run without a losing trade: such rounds count
    [[a, b, flat], { metric: "profit_factor" }],
    [[a, b], { metric: (out: any) => (out.data[(out.shape[0] - 1) * 4] > 0 ? Infinity : -Infinity) }],
  ];
  for (const [targets, opts] of cases) {
    const res = positionTest(BacktestPriceTarget(1e-4), targets, [c], { n: 120, seed: 3, ...opts });
    const K = targets.length, n = res.null.shape[0];
    const sign = (opts.largerIsBetter ?? true) ? 1 : -1;
    const best: number[] = [];
    for (let r = 0; r < n; r++) {
      const row = Array.from(res.null.data.subarray(r * K, (r + 1) * K), (v) => sign * v).filter((v) => !Number.isNaN(v));
      if (row.length) best.push(Math.max(...row));
    }
    for (let k = 0; k < K; k++) {
      const o = res.observed[k];
      const want = Number.isFinite(o) ? (1 + best.filter((v) => v >= sign * o).length) / (1 + best.length) : NaN;
      assert.ok(Object.is(res.familyPValue[k], want), `${k}: ${res.familyPValue[k]} vs ${want}`);
    }
  }
});

test("validation: per-strategy p-values keep infinite metrics", async () => {
  const { strategyTest, BacktestPortfolioTarget } = await import("../src/index.ts");
  const T = 200;
  const { c } = market(T);
  const a = Array.from({ length: T }, (_, t) => (Math.floor(t / 6) % 2 ? 1 : -1));
  const b = a.map((v, t) => (t % 5 ? v : -v));
  const eng = () => BacktestPriceTarget(1e-4);
  const real = eng()(Float64Array.from(b), c) as any;
  const thr = real.data[(T - 1) * 4] + 0.1;      // +inf above the real run, so it stays finite
  const pnl = (out: any) => out.data[(out.shape[0] - 1) * 4];
  const inf = (out: any) => { const v = pnl(out); return v > thr ? Infinity : v === 0 ? NaN : v; };
  // recomputed from the null: NaN left out, infinite metrics kept
  const nullP = (nul: ArrayLike<number>, o: number, larger: boolean) => {
    if (!Number.isFinite(o)) return NaN;
    const kept = Array.from(nul).filter((v) => !Number.isNaN(v));
    return (1 + kept.filter((v) => (larger ? v >= o : v <= o)).length) / (1 + kept.length);
  };
  let seen = 0;
  for (const [metric, larger] of [["profit_factor", undefined], ["total_pnl", false], [inf, undefined], [inf, false]] as const) {
    const L = larger ?? true;
    const one = positionTest(eng(), b, [c], { n: 150, seed: 4, metric: metric as any, largerIsBetter: larger });
    if (typeof metric === "string") {
      const full = one.nullStats[metric as "profit_factor"];
      assert.deepEqual(Array.from(one.null), Array.from(full).filter((v) => !Number.isNaN(v)));
    }
    seen += Array.from(one.null).filter((v) => !Number.isFinite(v)).length * (Number.isFinite(one.observed) ? 1 : 0);
    assert.ok(Object.is(one.pValue, nullP(one.null, one.observed, L)));
    const fam = positionTest(eng(), [a, b, new Array(T).fill(0)], [c], { n: 150, seed: 4, metric: metric as any, largerIsBetter: larger });
    for (let k = 0; k < 3; k++) {
      const col = Array.from({ length: 150 }, (_, r) => fam.null.data[r * 3 + k]);
      assert.ok(Object.is(fam.pValue[k], nullP(col, fam.observed[k], L)), `${k}`);
    }
    const pan = c.map((v) => [v, v]), w = a.map((v, t) => [v / 2, b[t] / 2]);
    const book = positionTest(BacktestPortfolioTarget(1e-4, 0, "fraction", 0, 1, 0, "units", 0), w, [pan], { n: 60, seed: 4, metric: metric as any, largerIsBetter: larger });
    assert.ok(Object.is(book.pValue, nullP(book.null, book.observed, L)));
    const st = strategyTest((p: Float64Array) => eng()(Float64Array.from(b), p), [c],
      { method: "returns", n: 60, seed: 4, metric: metric as any, largerIsBetter: larger });
    assert.ok(Object.is(st.pValue, nullP(st.null, st.observed, L)));
  }
  assert.ok(seen > 0);
});

test("validation: positionTest takes NdArray series markets", () => {
  const T = 90;
  const { c } = market(T);
  const cs = Float64Array.from(c), nc = { data: cs, shape: [T] };
  const a = Array.from({ length: T }, (_, t) => (Math.floor(t / 5) % 2 ? 1 : -1));
  const b = a.map((v, t) => (t % 7 ? v : -v));
  const block = Float64Array.from({ length: 2 * T }, (_, i) => (i % 2 ? b : a)[i >> 1]);
  assert.deepEqual(positionTest(BacktestPriceTarget(), Float64Array.from(a), [nc], { n: 9 }),
    positionTest(BacktestPriceTarget(), a, [c], { n: 9 }));
  const fam = positionTest(BacktestPriceTarget(), [a, b], [c], { n: 9 });
  assert.deepEqual(positionTest(BacktestPriceTarget(), [a, b], [nc], { n: 9 }), fam);
  assert.deepEqual(positionTest(BacktestPriceTarget(), { data: block, shape: [T, 2] }, [nc], { n: 9 }), fam);
});

test("validation: the first bad metric result raises before the metric runs again", async () => {
  const { walkForward } = await import("../src/index.ts");
  const T = 80;
  const { c } = market(T);
  const tgt = Array.from({ length: T }, (_, t) => (Math.floor(t / 5) % 2 ? 1 : -1));
  const X = [tgt, tgt, tgt, tgt, tgt];
  class Late extends Error {}
  for (const [bad, kind] of [[true, TypeError], [[1, 2], RangeError]] as const) {
    // the real run's call sees strategy 1 to 5; call 2 is bad, call 4 throws
    const make = () => {
      let k = 0;
      return (out: any) => { k++; if (k === 2) return bad; if (k === 4) throw new Late(); return out.data[0]; };
    };
    assert.throws(() => positionTest(BacktestPriceTarget(), X, [c], { n: 3, metric: make() }), kind);
    assert.throws(() => walkForward(BacktestPriceTarget(), X, [c], { train: 30, test: 10, metric: make() }), kind);
    let calls = 0;
    const counting = make();
    assert.throws(() => positionTest(BacktestPriceTarget(), X, [c], { n: 3,
      metric: (out: any) => { calls++; return counting(out); } }), kind);
    assert.equal(calls, 2);      // the batch stopped at the bad result
  }
});
