import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ready, BacktestPortfolioTarget, BacktestPortfolioOHLCTarget, BacktestPriceTarget, PortfolioReport,
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
  const b = BacktestPriceTarget(0, 1e-3, -Infinity, Infinity, 1, 0, 5e-4)(Float64Array.from(tgt), px) as { data: Float64Array };
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

// C1: PortfolioReport()(out, prices) reports turnover as traded notional.
// Positions [[0, 0], [1, 2], [1, 0]] at prices [[10, 1], [10, 1], [20, 1]]:
// step 1 trades |1|*10 + |2|*1 = 12, step 2 trades 0*20 + |-2|*1 = 2.
const pricedRows = [[0, 0], [1, 2], [1, 0]].map((pos) => pos.map((p) => [0, 0, p, 0]));
const pricedOut = { data: Float64Array.from(pricedRows.flat(2)), shape: [3, 2, 4] };
const priceRows = [[10, 1], [10, 1], [20, 1]];
const pricedPrices = { data: Float64Array.from(priceRows.flat()), shape: [3, 2] };
const col = (rows: ArrayLike<number>[], j: number) => rows.map((r) => r[j]);
const asRows = (x: any): ArrayLike<number>[] =>
  Array.from({ length: x.shape[0] }, (_, t) => x.data.slice(t * 6, (t + 1) * 6));

test("C1: turnover with prices is traded notional", () => {
  const rep = PortfolioReport()(pricedOut, pricedPrices) as any;
  assert.deepEqual(col(asRows(rep), 2), [0, 12, 14]);
  assert.deepEqual(col(asRows(rep), 3), [0, 2, 3]);
  assert.deepEqual(col(asRows(PortfolioReport()(pricedOut) as any), 2), [0, 3, 5]);
});

test("C1: priced streams, live events, and containers equal the batch", async () => {
  const batch = asRows(PortfolioReport()(pricedOut, pricedPrices) as any).map((r) => Array.from(r));
  const same = (got: ArrayLike<number>[]) =>
    assert.deepEqual(got.map((r) => Array.from(r).map((v) => (Number.isNaN(v) ? "nan" : v))),
      batch.map((r) => r.map((v) => (Number.isNaN(v) ? "nan" : v))));
  same([...PortfolioReport()(pricedRows.values(), priceRows.values()) as any]);
  same([...PortfolioReport()(pricedRows.map((r, t) => [r, priceRows[t]]).values()) as any]);
  async function* agen(x: any[]) { for (const v of x) yield v; }
  const aio: any[] = [];
  for await (const r of PortfolioReport()(agen(pricedRows), agen(priceRows)) as any) aio.push(r);
  same(aio);
  const op = PortfolioReport();
  same(pricedRows.map((r, t) => op(r, priceRows[t]) as any));
  const list = PortfolioReport()(pricedRows, priceRows) as any;
  assert.ok(Array.isArray(list));
  same(list);
  for (const [o, p] of [[pricedRows, pricedPrices], [pricedOut, priceRows]] as const) {
    const got = PortfolioReport()(o, p) as any;
    assert.ok(!Array.isArray(got) && got.shape, "A8: any NdArray gives an NdArray");
    same(asRows(got));
  }
});

test("C1: prices of one match no prices; mismatched shapes and kinds raise", () => {
  const ones = { data: new Float64Array(6).fill(1), shape: [3, 2] };
  assert.deepEqual(PortfolioReport()(pricedOut, ones), PortfolioReport()(pricedOut));
  assert.throws(() => PortfolioReport()(pricedOut, { data: new Float64Array(4), shape: [2, 2] }), RangeError);
  assert.throws(() => PortfolioReport()(pricedOut, { data: new Float64Array(3), shape: [3, 1] }), RangeError);
  assert.throws(() => PortfolioReport()(pricedRows[1], [10]), RangeError);
  assert.throws(() => PortfolioReport()(pricedOut, "prices"), TypeError);
  assert.throws(() => PortfolioReport()(pricedRows.values(), pricedPrices), TypeError);
});

test("C1: a NaN price is read only where the position changed", () => {
  const p = priceRows.map((r) => r.slice());
  p[2][0] = Number.NaN;                       // asset 0 holds 1 unit at step 2
  p[0] = [Number.NaN, Number.NaN];            // nothing trades at step 0
  assert.deepEqual(col(PortfolioReport()(pricedRows, p) as any, 2), [0, 12, 14]);
  p[2][0] = 20;
  p[1][0] = Number.NaN;                       // asset 0 trades at step 1: the row is skipped
  const rep = PortfolioReport()(pricedRows, p) as any;
  assert.ok(Array.from(rep[1]).every((v) => Number.isNaN(v)));
  assert.equal(rep[2][2], 20);
});

test("C1: the prices mode is locked until reset, live and in a stream", () => {
  const op = PortfolioReport();
  op(pricedRows[0], priceRows[0]);
  assert.throws(() => op(pricedRows[1]), { name: "RangeError", message: /without prices after rows with prices/ });
  op.reset();
  op(pricedRows[0]);
  assert.throws(() => op(pricedRows[1], priceRows[1]), { name: "RangeError", message: /prices after rows without prices/ });
  op.reset();
  assert.deepEqual(op(pricedRows[0], priceRows[0]), [0, 0, 0, 0, 0, Number.NaN]);
  assert.throws(() => [...PortfolioReport()([[pricedRows[0], priceRows[0]], pricedRows[1]].values()) as any],
    { name: "RangeError", message: /without prices/ });
  assert.throws(() => [...PortfolioReport()([pricedRows[0], [pricedRows[1], priceRows[1]]].values()) as any],
    { name: "RangeError", message: /prices after rows without/ });
});

test("C1: async pairs, [T, K, 1] prices, malformed pairs", async () => {
  const batch = PortfolioReport()(pricedRows, priceRows) as any;
  async function* pairs() { for (let t = 0; t < 3; t++) yield [pricedRows[t], priceRows[t]]; }
  const got: any[] = [];
  for await (const r of PortfolioReport()(pairs()) as any) got.push(Array.from(r));
  assert.deepEqual(got, batch.map((r: any) => Array.from(r)));
  const unit = { data: pricedPrices.data, shape: [3, 2, 1] };
  assert.deepEqual(PortfolioReport()(pricedOut, unit), PortfolioReport()(pricedOut, pricedPrices));
  assert.throws(() => [...PortfolioReport()([[new Float64Array(8), [1, 1, 1]]].values()) as any],
    { name: "RangeError", message: /got two parts of 8 and 3 values/ });
  const one = [[[0, 0, 0, 0]], [[0, 0, 1, 0]]];
  assert.throws(() => [...PortfolioReport()([[one[0][0], [1]], [one[1][0], [1, 1, 1, 1]]].values()) as any], RangeError);
  assert.throws(() => PortfolioReport()(pricedRows.map((r, t) => [r, priceRows[t]])),
    { name: "TypeError", message: /two arguments/ });
});

// G33 (a): one packed [T, K, F] NdArray, F last in argument order, gives the
// separate arrays' results bit for bit.
test("portfolio: a packed [T, K, F] array equals the separate arrays", () => {
  const pack = (fields: number[][][]) => {
    const F = fields.length, data = new Float64Array(T * K * F);
    for (let t = 0; t < T; t++) for (let k = 0; k < K; k++) for (let f = 0; f < F; f++)
      data[(t * K + k) * F + f] = fields[f][t][k];
    return { data, shape: [T, K, F] };
  };
  const bits = (a: { data: Float64Array }, b: { data: Float64Array }) =>
    assert.deepEqual(new Uint8Array(a.data.buffer, a.data.byteOffset, a.data.byteLength),
                     new Uint8Array(b.data.buffer, b.data.byteOffset, b.data.byteLength));
  const p = BacktestPortfolioTarget({ fee: [0.001] });
  bits(p(pack([w, close])) as any, p(mat(w), mat(close)) as any);
  const o = BacktestPortfolioOHLCTarget({ fee: [0.001], fillAt: "ohlc4", delay: 2 });
  const sep = o(mat(w), mat(open), mat(high), mat(low), mat(close)) as any;
  const packed = o(pack([w, open, high, low, close])) as any;
  assert.deepEqual(packed.shape, [T, K, 4]);
  bits(packed, sep);
  assert.throws(() => p(pack([w, close, close])), { name: "RangeError",
    message: "a packed input of BacktestPortfolioTarget has shape [T, K, 2], got [12,3,3]" });
  assert.throws(() => o(mat(w), mat(close)), { name: "RangeError",
    message: "BacktestPortfolioOHLCTarget expects 5 inputs, or one packed [T, K, 5] array, got 2" });
  p.dispose(); o.dispose();
});
