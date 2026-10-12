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
