// The research-toolkit operators and backtest helpers against the Python
// binding. Expected values come from wasm/smoke/oracle_research.json
// (devtools/wasm/gen_research_oracle.py).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { ready, Hold, CusumFilter, LiquidationDistance, RollingMultiRegression, ExpandingMultiRegression,
         EwMultiRegression, backtestReport, walkForward, BacktestOHLCTarget,
         BacktestPortfolioTarget } from "../src/index.ts";
import * as api from "../src/index.ts";

await ready();

type Num = number | null | "inf" | "-inf";
type Case = { name: string; settings: Record<string, unknown>; input: string | string[]; shape: number[];
              out: Num[]; tol?: number };
const oracle: { inputs: Record<string, Num[]>; cases: Case[]; helpers: Record<string, any> } =
  JSON.parse(readFileSync(new URL("../../wasm/smoke/oracle_research.json", import.meta.url), "utf8"));
const num = (v: Num): number => (v === null ? NaN : v === "inf" ? Infinity : v === "-inf" ? -Infinity : v);
const series = (k: string) => Float64Array.from(oracle.inputs[k], num);
const camel = (k: string) => k.replace(/_([a-z])/g, (_m, c) => c.toUpperCase());
// An operator setting, {"functor", "params"}, is built by its factory.
const setting = (v: any): unknown => (v !== null && typeof v === "object" && "functor" in v
  ? (api as any)[v.functor](options(v.params)) : v);
const options = (s: Record<string, unknown>): Record<string, unknown> =>
  Object.fromEntries(Object.entries(s).map(([k, v]) => [camel(k), setting(v)]));

export const close = (got: ArrayLike<number>, want: Num[], what: string, tol = 1e-9) => {
  assert.equal(got.length, want.length, `${what}: length`);
  for (let i = 0; i < want.length; i++) {
    const w = num(want[i]);
    if (Number.isNaN(w)) assert.ok(Number.isNaN(got[i]), `${what} [${i}]: ${got[i]} is not NaN`);
    else if (!Number.isFinite(w)) assert.equal(got[i], w, `${what} [${i}]`);
    else assert.ok(Math.abs(got[i] - w) <= tol * Math.max(1, Math.abs(w)), `${what} [${i}]: ${got[i]} vs ${w}`);
  }
};

const dataOf = (out: any): ArrayLike<number> => (ArrayBuffer.isView(out) ? out as Float64Array : out.data);

for (const c of oracle.cases) {
  test(`research: ${c.name}(${JSON.stringify(c.settings)}) matches Python`, () => {
    const make = (api as any)[c.name];
    const inputs = Array.isArray(c.input) ? c.input.map(series) : [series(c.input)];
    const out = make(options(c.settings))(...inputs);
    close(dataOf(out), c.out, c.name, c.tol ?? 1e-9);
  });
}

test("research: Hold cooldown worked example", () => {
  const out = Hold({ n: 2, cooldown: 2 })(Float64Array.from([5, 0, 7, 3, 0, 4, 0])) as Float64Array;
  assert.deepEqual(Array.from(out), [5, 5, 0, 0, 0, 4, 4]);
});

test("research: CusumFilter hand example", () => {
  const out = CusumFilter({ threshold: 1 })(Float64Array.from([0.5, 0.6, -0.2, 0.3, -1.5, 0.1, NaN, 0.4])) as Float64Array;
  assert.deepEqual(Array.from(out.subarray(0, 6)), [0, 1, 0, 0, -1, 0]);
  assert.ok(Number.isNaN(out[6]));
  assert.equal(out[7], 0);
});

test("research: RollingUpsideDeviation hand example", () => {
  const out = (api as any).RollingUpsideDeviation({ windowSize: 3 })(Float64Array.from([1, -2, 2, 0])) as Float64Array;
  assert.ok(Number.isNaN(out[0]) && Number.isNaN(out[1]));
  assert.ok(Math.abs(out[2] - Math.sqrt(5 / 3)) < 1e-12 && Math.abs(out[3] - Math.sqrt(4 / 3)) < 1e-12);
});

test("research: RollingSeasonalMean NaN advances the slot clock", () => {
  const out = (api as any).RollingSeasonalMean({ period: 2, windowSize: 2, startPolicy: "expanding" })(
    Float64Array.from([1, 10, NaN, 20, 5, 30])) as Float64Array;
  assert.ok(Number.isNaN(out[2]));
  assert.deepEqual([out[0], out[1], out[3], out[4], out[5]], [1, 10, 15, 3, 25]);
});

test("research: RollingJumpShare hand example", () => {
  const out = (api as any).RollingJumpShare({ windowSize: 4 })(Float64Array.from([1, 1, 1, 10])) as Float64Array;
  assert.ok(Math.abs(out[3] - (1 - 8 * Math.PI / 103)) < 1e-12);
});

test("research: RollingVarianceRatio hand example", () => {
  const x = Float64Array.from([1, -1, 1, -1, 1]);
  const ratio = (api as any).RollingVarianceRatio({ windowSize: 5, lag: 2 })(x) as Float64Array;
  const z = (api as any).RollingVarianceRatio({ windowSize: 5, lag: 2, output: "zscore" })(x) as Float64Array;
  assert.ok(Math.abs(ratio[4] - 1 / 9) < 1e-12 && Math.abs(z[4] + 20 / 9) < 1e-12);
});

test("research: RollingPermutationEntropy of a monotone series is exactly 0", () => {
  const x = Float64Array.from({ length: 50 }, (_, i) => i);
  const out = (api as any).RollingPermutationEntropy({ windowSize: 10, order: 3 })(x) as Float64Array;
  assert.ok(Array.from(out.subarray(11)).every((v) => v === 0));
});

test("research: BayesianChangepoint first value", () => {
  const out = (api as any).BayesianChangepoint()(Float64Array.from([0.5])) as { data: Float64Array; shape: number[] };
  assert.deepEqual(out.shape, [1, 3]);
  assert.ok(out.data[0] === 0 && Math.abs(out.data[1] - 0.25) < 1e-12 && Math.abs(out.data[2] - Math.sqrt(3.1875)) < 1e-12);
});

test("research: RollingHMM is NaN until its first fit", () => {
  const x = Float64Array.from({ length: 30 }, (_, i) => Math.sin(i) * (i % 7 === 0 ? 3 : 1));
  const out = (api as any).RollingHMM({ windowSize: 20, refitEvery: 5 })(x) as { data: Float64Array; shape: number[] };
  assert.deepEqual(out.shape, [30, 3]);
  assert.ok(Array.from(out.data.subarray(0, 19 * 3)).every(Number.isNaN));
  assert.ok(Array.from(out.data.subarray(19 * 3)).every(Number.isFinite));
});

test("research: backtestTrades matches Python", () => {
  const h = oracle.helpers.backtest_trades;
  const values = { data: Float64Array.from(h.values, num), shape: [h.T, 4] };
  const tr = (api as any).backtestTrades(values);
  for (const k of ["entry", "exit", "side", "bars", "pnl", "cost"]) close(tr[k], h.result[k], `backtestTrades ${k}`);
  close(Float64Array.from(tr.closed), h.result.closed, "backtestTrades closed");
});

test("research: backtestTrades rejects a portfolio output", () => {
  assert.throws(() => (api as any).backtestTrades({ data: new Float64Array(24), shape: [2, 3, 4] }),
                /once per asset/);
});

test("research: backtestReport by matches Python", () => {
  const hb = oracle.helpers.by, ht = oracle.helpers.backtest_trades;
  const values = { data: Float64Array.from(ht.values, num), shape: [ht.T, 4] };
  const { byGroup } = backtestReport(values, { periodsPerYear: hb.periods_per_year, by: hb.labels });
  assert.deepEqual([...byGroup!.keys()], hb.groups.map((g: any) => g[0]));
  for (const [label, stats] of hb.groups) {
    const got = byGroup!.get(label)!;
    for (const [k, v] of Object.entries(stats as Record<string, Num>)) close([got[k]], [v], `by ${label} ${k}`);
  }
});

test("research: backtestReport by with one string label equals the summary", () => {
  const ht = oracle.helpers.backtest_trades;
  const values = { data: Float64Array.from(ht.values, num), shape: [ht.T, 4] };
  const { summary, byGroup } = backtestReport(values, { by: Array(ht.T).fill("all") });
  const g = byGroup!.get("all")!;
  for (const k of ["total_cost", "turnover", "sharpe"]) assert.ok(Math.abs(g[k] - summary[k]) <= 1e-9 * Math.max(1, Math.abs(summary[k])), k);
});

test("research: walkForward with carry matches Python", () => {
  const h = oracle.helpers.walk_forward_carry;
  const T = h.carry.length;
  const market = h.market.map((x: Num[]) => Float64Array.from(x, num));
  const targets = { data: Float64Array.from(h.targets, num), shape: [T, h.K] };
  const res = walkForward(BacktestOHLCTarget({ takerFee: 2e-4 }), targets, market,
    { train: h.train, test: h.test, carry: Float64Array.from(h.carry, num), carryMultiplier: h.carry_multiplier });
  close(res.trainScores.data, h.train_scores, "trainScores");
  assert.deepEqual(Array.from(res.choice), h.choice);
  close(res.output.data, h.output, "output");
  for (const [k, v] of Object.entries(h.stats as Record<string, Num>)) close([(res.stats as any)[k]], [v], `stats ${k}`);
});

test("research: walkForward with a portfolio engine matches Python", () => {
  const h = oracle.helpers.walk_forward_portfolio;
  const prices = { data: Float64Array.from(h.prices, num), shape: [h.T, h.K] };
  const targets = { data: Float64Array.from(h.targets, num), shape: [h.T, h.K, h.C] };
  const carry = { data: Float64Array.from(h.carry, num), shape: [h.T, h.K] };
  const res = walkForward(BacktestPortfolioTarget({ fee: 1e-4 }), targets, [prices],
                          { train: h.train, test: h.test, carry });
  close(res.trainScores.data, h.train_scores, "trainScores");
  assert.deepEqual(Array.from(res.choice), h.choice);
  close((res.target as any).data, h.target, "target");
  close(res.output.data, h.output, "output");
  for (const [k, v] of Object.entries(h.stats as Record<string, Num>)) close([(res.stats as any)[k]], [v], `stats ${k}`);
});

test("research: walkForward restates held portfolio targets as Python does", () => {
  const p = oracle.helpers.walk_forward_portfolio, h = oracle.helpers.walk_forward_portfolio_hold;
  const prices = { data: Float64Array.from(p.prices, num), shape: [p.T, p.K] };
  const targets = { data: Float64Array.from(h.targets, num), shape: [p.T, p.K, p.C] };
  // the carry as T rows of K values
  const flat = Float64Array.from(p.carry, num);
  const carry = Array.from({ length: p.T }, (_, t) => Array.from(flat.subarray(t * p.K, (t + 1) * p.K)));
  const res = walkForward(BacktestPortfolioTarget({ fee: 1e-3 }), targets, [prices],
                          { train: h.train, test: h.test, carry });
  assert.deepEqual(Array.from(res.choice), h.choice);
  close(res.trainScores.data, h.train_scores, "trainScores");
  close((res.target as any).data, h.target, "target");
  close(res.output.data, h.output, "output");
  for (const [k, v] of Object.entries(h.stats as Record<string, Num>)) close([(res.stats as any)[k]], [v], `stats ${k}`);
});

test("research: engines declare their mark input for walkForward carry", () => {
  const marks: [string, number][] = [["BacktestPriceTarget", 0], ["BacktestOHLCTarget", 3],
    ["BacktestOHLCVTarget", 3], ["BacktestTradesTarget", 0], ["BacktestL1Target", 1],
    ["BacktestPortfolioTarget", 0], ["BacktestPortfolioOHLCTarget", 3]];
  for (const [name, mark] of marks) assert.equal(((api as any)[name]() as any).markInput, mark, name);
  assert.equal(((api as any).BacktestCarry() as any).markInput, undefined);
  const T = 40, c = Float64Array.from({ length: T }, (_, t) => 100 + Math.sin(t));
  const x = Float64Array.from({ length: T }, (_, t) => (t % 7 < 3 ? 1 : -1));
  assert.throws(() => walkForward((api as any).BacktestCarry(), x, [c, c, c, c, c, c],
                                  { train: 10, test: 10, carry: new Float64Array(T) }),
    { name: "RangeError", message: /carry needs an engine with a mark input.*BacktestCarry has none/ });
});

test("research: backtestReport by keeps the number 1 and the string '1' apart", () => {
  const ht = oracle.helpers.backtest_trades;
  const values = { data: Float64Array.from(ht.values, num).slice(0, 16), shape: [4, 4] };
  const { byGroup } = backtestReport(values, { by: [1, "1", 1, "1"] });
  assert.deepEqual([...byGroup!.keys()], [1, "1"]);
});

const liqRows = (out: any, n: number) =>
  Array.from({ length: n }, (_, i) => Array.from(dataOf(out)).slice(3 * i, 3 * i + 3));
const f64 = (xs: number[]) => Float64Array.from(xs);

test("research: LiquidationDistance hand example", () => {
  // Long 1 at 100 (liq 90.5), breached on bar 2; increased to 2 at 95 on bar 3
  // (entry 97.5, liq 88.2375, still breached); flipped short at 100 on bar 5
  // (liq 109.5, breach cleared); flat on bar 7.
  const out = LiquidationDistance({ leverage: 10 })(
    f64([0, 1, 1, 2, 2, -1, -1, 0]), f64([100, 100, 98, 95, 90, 100, 104, 100]),
    f64([102, 102, 100, 97, 92, 102, 110, 102]), f64([97, 97, 90, 92, 85, 97, 101, 97]));
  const rows = liqRows(out, 8);
  assert.ok(rows[0].every(Number.isNaN) && rows[7].every(Number.isNaN));
  close(rows[1], [(97 - 90.5) / 97, 90.5, 0], "long");
  close(rows[2], [(90 - 90.5) / 90, 90.5, 1], "long breach");
  close(rows[3], [(92 - 88.2375) / 92, 88.2375, 1], "blended increase keeps the breach");
  close(rows[4], [(85 - 88.2375) / 85, 88.2375, 1], "increase breach");
  close(rows[5], [(109.5 - 102) / 102, 109.5, 0], "short");
  close(rows[6], [(109.5 - 110) / 110, 109.5, 1], "short breach");
});

test("research: LiquidationDistance reduce keeps the entry", () => {
  const out = LiquidationDistance({ leverage: 10 })(
    f64([3, 3, 1, 2]), f64([100, 105, 110, 120]), f64([101, 106, 111, 121]), f64([99, 104, 109, 119]));
  const rows = liqRows(out, 4);
  close(rows.slice(0, 3).map((r) => r[1]), [90.5, 90.5, 90.5], "reduce");
  close([rows[3][1]], [(100 + 120) / 2 * 0.905], "increase after a reduce blends from the kept entry");
});

test("research: LiquidationDistance pending fills on NaN opens", () => {
  // Flat again before the pending fill: nothing leaks into the next entry.
  let rows = liqRows(LiquidationDistance({ leverage: 10 })(
    f64([0, 1, 0, 1]), f64([100, NaN, 100, 97]), f64([101, 101, 101, 98]), f64([99, 99, 99, 96])), 4);
  assert.ok(rows.slice(0, 3).every((r) => r.every(Number.isNaN)));
  close(rows[3], [(96 - 97 * 0.905) / 96, 97 * 0.905, 0], "fresh entry after a cancelled fill");
  // A flip on a NaN-open bar fills at the next finite open and clears the breach.
  rows = liqRows(LiquidationDistance({ leverage: 10 })(
    f64([1, 1, -1, -1]), f64([100, 92, NaN, 97]), f64([101, 93, 98, 98]), f64([99, 90, 96, 96])), 4);
  assert.equal(rows[1][2], 1);
  assert.ok(rows[2].every(Number.isNaN));
  close(rows[3], [(97 * 1.095 - 98) / 98, 97 * 1.095, 0], "flip on a NaN open");
});

test("research: LiquidationDistance on (T, K) columns matches Python", () => {
  const h = oracle.helpers.liquidation_2d;
  const nd = (xs: Num[]) => ({ data: Float64Array.from(xs, num), shape: [h.T, h.K] });
  const out = LiquidationDistance({ leverage: h.leverage })(...h.inputs.map(nd)) as any;
  assert.deepEqual(Array.from(out.shape), h.shape);
  close(dataOf(out), h.output, "LiquidationDistance 2-D");
});

test("research: LiquidationDistance setting checks", () => {
  assert.throws(() => LiquidationDistance({ leverage: 1 }), { name: "RangeError", message: /leverage must be more than 1/ });
  assert.throws(() => LiquidationDistance({ leverage: 10, maintenance: 0.1 }),
    { name: "RangeError", message: /maintenance must be 0 or more and less than 1 \/ leverage/ });
});

test("research: RollingMultiRegression on (T, K) columns matches Python", () => {
  const h = oracle.helpers.ols_2d;
  const nd = (xs: Num[]) => ({ data: Float64Array.from(xs, num), shape: [h.T, h.K] });
  const out = RollingMultiRegression({ windowSize: h.window_size, nRegressors: h.n_regressors })(...h.inputs.map(nd)) as any;
  assert.deepEqual(Array.from(out.shape), h.shape);
  close(dataOf(out), h.output, "RollingMultiRegression 2-D");
});

test("research: RollingMultiRegression one event at a time equals the batch", () => {
  const ins = ["ols_y", "ols_x1", "ols_x2"].map(series);
  const batch = dataOf(RollingMultiRegression(20, 2)(...ins));
  const op = RollingMultiRegression(20, 2);
  const rows: number[] = [];
  for (let t = 0; t < ins[0].length; t++) rows.push(...(op(ins[0][t], ins[1][t], ins[2][t]) as number[]));
  assert.equal(rows.length, batch.length);
  for (let i = 0; i < rows.length; i++) assert.ok(Object.is(rows[i], batch[i]) || rows[i] === batch[i], `[${i}]`);
});

test("research: RollingMultiRegression and ExpandingMultiRegression setting checks", () => {
  assert.throws(() => RollingMultiRegression({ windowSize: 3, nRegressors: 2 }),
    { name: "RangeError", message: /window_size must be at least n_regressors \+ 1/ });
  assert.throws(() => RollingMultiRegression(40, 17), { name: "RangeError", message: /n_regressors must be between 1 and 16/ });
  assert.throws(() => ExpandingMultiRegression(0), { name: "RangeError", message: /^ExpandingMultiRegression: nRegressors must be at least 1, got 0$/ });
  assert.throws(() => ExpandingMultiRegression(2, 1 as any), { name: "TypeError", message: /^ExpandingMultiRegression: intercept must be a bool, got number$/ });
  assert.throws(() => RollingMultiRegression(20, 2)(series("ols_y"), series("ols_x1")),
    { name: "RangeError", message: /^RollingMultiRegression expects 3 inputs, got 2$/ });
  assert.equal((ExpandingMultiRegression(2, false)(1, 2, 3) as number[]).length, 5);
});

test("research: EwMultiRegression on (T, K) columns matches Python", () => {
  const h = oracle.helpers.ew_2d;
  const nd = (xs: Num[]) => ({ data: Float64Array.from(xs, num), shape: [h.T, h.K] });
  const out = EwMultiRegression({ nRegressors: h.n_regressors, span: h.span })(...h.inputs.map(nd)) as any;
  assert.deepEqual(Array.from(out.shape), h.shape);
  close(dataOf(out), h.output, "EwMultiRegression 2-D");
});

test("research: EwMultiRegression one event at a time equals the batch", () => {
  const ins = ["ols_y", "ols_x1", "ols_x2"].map(series);
  const batch = dataOf(EwMultiRegression({ nRegressors: 2, alpha: 0.1 })(...ins));
  const op = EwMultiRegression({ nRegressors: 2, alpha: 0.1 });
  const rows: number[] = [];
  for (let t = 0; t < ins[0].length; t++) rows.push(...(op(ins[0][t], ins[1][t], ins[2][t]) as number[]));
  assert.equal(rows.length, batch.length);
  for (let i = 0; i < rows.length; i++) assert.ok(Object.is(rows[i], batch[i]) || rows[i] === batch[i], `[${i}]`);
});

test("research: EwMultiRegression setting checks", () => {
  assert.throws(() => EwMultiRegression(2), { name: "RangeError", message: /Exactly one of com, span, halflife, or alpha/ });
  assert.throws(() => EwMultiRegression({ nRegressors: 2, span: 10, alpha: 0.1 }),
    { name: "RangeError", message: /Exactly one of com, span, halflife, or alpha/ });
  assert.throws(() => EwMultiRegression({ nRegressors: 2, alpha: 1 }), { name: "RangeError", message: /Alpha must be a finite value/ });
  assert.throws(() => EwMultiRegression({ nRegressors: 0, alpha: 0.1 }),
    { name: "RangeError", message: /^EwMultiRegression: nRegressors must be at least 1, got 0$/ });
  assert.throws(() => EwMultiRegression(2, 1 as any, null, null, null, 0.1),
    { name: "TypeError", message: /^EwMultiRegression: intercept must be a bool, got number$/ });
  assert.equal((EwMultiRegression(2, false, null, 10)(1, 2, 3) as number[]).length, 5);
});
