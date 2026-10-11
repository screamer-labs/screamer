import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ready, BacktestPortfolioTarget, BacktestPortfolioOHLCTarget, BacktestPriceTarget,
} from "../src/index.ts";

await ready();

// The same deterministic market as the Python reference run.
const T = 12, K = 3;
const grid = (f: (t: number, k: number) => number) =>
  Array.from({ length: T }, (_, t) => Array.from({ length: K }, (_, k) => f(t, k)));
const close = grid((t, k) => 100 + 5 * Math.sin(0.7 * t + k) + t * (k - 1) * 0.3);
const open = grid((t, k) => close[t][k] - 0.4 * Math.cos(1.3 * t + k));
const high = grid((t, k) => Math.max(open[t][k], close[t][k]) + 0.5);
const low = grid((t, k) => Math.min(open[t][k], close[t][k]) - 0.5);
const N = Number.NaN;
const w = grid(() => N);
w[0] = [0.5, 0.3, -0.2];
w[6] = [0.2, -0.4, 0.4];

const near = (a: ArrayLike<number>, b: number[]) =>
  a.length === b.length && Array.from(a).every((v, i) => Math.abs(v - b[i]) <= 1e-8 * Math.max(1, Math.abs(b[i])));
// An array of rows as a (T, K) NdArray: engines return an array of rows for an array of rows (A1).
const mat = (rows: ArrayLike<number>[]) =>
  ({ data: Float64Array.from(rows.flatMap((r) => Array.from(r))), shape: [rows.length, rows[0].length] });
const lastRow = (out: { data: Float64Array; shape: number[] }) =>
  out.data.slice((T - 1) * K * 4, T * K * 4);

test("portfolio: price engine matches Python", () => {
  const out = BacktestPortfolioTarget([1e-3, 2e-3, 0], 1e-3, "fraction", 0, 1, 0, "weight", 1000, 0, "drifted", 0.9)(mat(w), mat(close));
  assert.deepEqual(out.shape, [T, K, 4]);
  assert.ok(near(lastRow(out), [-14.848773927210585, 2.479841617924665, 1.8289156022515223, 0.0,
    -52.696147942260524, 5.8606744881548165, -3.5912204094741877, 0.0,
    6.630791765020081, -10.56136701579796, 3.38570944868474, 0.0]), `${Array.from(lastRow(out))}`);
});

test("portfolio: OHLC engine matches Python, row by row too", () => {
  const make = () => BacktestPortfolioOHLCTarget(5e-4, 0, "fraction", 0, 1, 0, "weight", 1000, 0.02, "all");
  const out = make()(mat(w), mat(open), mat(high), mat(low), mat(close));
  assert.ok(near(lastRow(out), [-30.25379519300619, 2.579317680865061, 1.9022804987220079, 0.0,
    -38.35574725809002, 5.806450377889529, -3.5579937336258953, 0.0,
    -14.707953559570182, -10.768468619923866, 3.45210103008495, 0.0]));
  const op = make();
  const rows = w.map((_, t) => op(w[t], open[t], high[t], low[t], close[t]));
  assert.deepEqual(rows[0].shape, [K, 4]);
  assert.ok(near(rows[T - 1].data, Array.from(lastRow(out))));
});

test("portfolio: one asset in units matches BacktestPriceTarget", () => {
  const tgt = close.map((_, t) => (t % 4) - 1.5);
  const px = close.map((r) => r[0]);
  const a = BacktestPortfolioTarget(1e-3, 5e-4, "fraction", 0, 1, 0, "units", 0)(
    mat(tgt.map((v) => [v])), mat(px.map((v) => [v])));
  const b = BacktestPriceTarget(0, 1e-3, -Infinity, Infinity, 1, 0, 5e-4)(tgt, px) as { data: Float64Array };
  assert.ok(near(a.data, Array.from(b.data)));
});

test("portfolio: argument checks", () => {
  assert.throws(() => BacktestPortfolioTarget(0, 0, "fraction", 0, 1, 0, "shares"));
  assert.throws(() => BacktestPortfolioTarget()(w));
});

// Same values as Python: position_test(BacktestPortfolioOHLCTarget(fee=1e-4), w, o, h, l, c, n=20, seed=1).
test("portfolio: positionTest on a portfolio engine matches Python", async () => {
  const { positionTest, CrossRank, CrossNormalize, strategyTest } = await import("../src/index.ts");
  const T2 = 200, K2 = 3;
  const c2 = Array.from({ length: T2 }, (_, t) =>
    Array.from({ length: K2 }, (_, k) => 100 * Math.exp(0.02 * Math.sin(0.21 * t * (k + 1)) + 0.001 * t * (k - 1))));
  const o2 = c2.map((_, t) => c2[Math.max(0, t - 1)]);
  const h2 = c2.map((r, t) => r.map((v, k) => Math.max(o2[t][k], v) * 1.001));
  const l2 = c2.map((r, t) => r.map((v, k) => Math.min(o2[t][k], v) * 0.999));
  const r2 = c2.map((r, t) => r.map((v, k) => (t === 0 ? 0 : Math.log(v / c2[t - 1][k]))));
  const w2 = CrossNormalize(1.0, 0.0)(CrossRank()(r2)) as Float64Array[];
  const res = positionTest(BacktestPortfolioOHLCTarget(1e-4), w2, [o2, h2, l2, c2], { n: 20, seed: 1 });
  assert.ok(near([res.observed], [1.452972122723607], 1e-7), `${res.observed}`);
  assert.equal(res.pValue, 0.047619047619047616);
  assert.equal(res.minShift, 7);
  assert.ok(near(res.null.slice(0, 3), [-0.5139950836552667, 0.04105857280105748, 0.5015761441403116], 1e-7));
  assert.ok(near(res.nullStats.turnover.slice(0, 3), [0.8993364407916147, 1.0821060107674296, 1.3473586309642132], 1e-7));
  // strategyTest accepts a strategy that returns a (T, K, 4) portfolio output.
  const st = strategyTest((...bars: Float64Array[]) => {
    const rows = (f: number) => Array.from({ length: T2 }, (_, t) => [0, 1, 2].map((k) => bars[k * 4 + f][t]));
    return BacktestPortfolioOHLCTarget(1e-4)(mat(w2), mat(rows(0)), mat(rows(1)), mat(rows(2)), mat(rows(3)));
  }, [0, 1, 2].flatMap((k) => [o2, h2, l2, c2].map((x) => x.map((r) => r[k]))),
  { n: 5, method: "shift", columns: [0] });
  assert.equal(st.nullStats.sharpe.length, 5);
});

test("A1: an NdArray target with an array-of-rows market returns an NdArray", () => {
  const all = BacktestPortfolioTarget()(mat(w), mat(close)) as any;
  const mixed = BacktestPortfolioTarget()(mat(w), close) as any;
  assert.ok(!Array.isArray(mixed));
  assert.deepEqual(mixed.shape, all.shape);
  assert.deepEqual(Array.from(mixed.data), Array.from(all.data));
});
