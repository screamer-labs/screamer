import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { ready, backtestReport, BacktestPriceTarget, BacktestReport, BacktestStats } from "../src/index.ts";
import type { NdArray } from "../src/index.ts";

// backtestReport against Python's backtest_report (devtools/wasm/gen_oracle.py
// writes the fixture). Both label the same C++ report, so the keys and their
// order match exactly, and so does every value up to the last bits: a native
// clang build fuses a multiply and an add within one expression, which
// WebAssembly does not (CMakeLists.txt), so values agree within a relative
// 1e-12 and NaN and infinities sit in the same places. Within JavaScript the
// comparisons are exact.
const HERE = dirname(fileURLToPath(import.meta.url));
type Num = number | null | "inf" | "-inf";
const cases: Array<{
  name: string;
  shape: number[];
  values: Num[];
  periods_per_year: number;
  running: Record<string, Num[]>;
  summary: Record<string, Num>;
}> = JSON.parse(readFileSync(join(HERE, "..", "..", "wasm", "smoke", "oracle_backtest.json"), "utf8"));

const num = (v: Num) => (v === null ? NaN : v === "inf" ? Infinity : v === "-inf" ? -Infinity : v);
const f64 = (xs: Num[]) => Float64Array.from(xs, num);
const same = (a: number, b: number) => (Number.isNaN(b) ? Number.isNaN(a) : a === b);
const close = (a: number, b: number) =>
  same(a, b) || (Number.isFinite(b) && Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(b)));

function assertSame(got: ArrayLike<number>, want: Float64Array, what: string, eq = same) {
  assert.equal(got.length, want.length, `${what}: length`);
  for (let i = 0; i < want.length; i++)
    assert.ok(eq(got[i], want[i]), `${what}: index ${i}: ${got[i]} vs ${want[i]}`);
}

function input(c: (typeof cases)[number]): NdArray {
  return { data: f64(c.values), shape: c.shape };
}

test("backtestReport: the fixture covers NaN and infinite values, annualizing, one row and no rows", () => {
  assert.deepEqual(cases.map((c) => c.name),
    ["base", "annualized", "holes", "winners", "one row", "no rows"]);
});

test("backtestReport: running columns and summary match Python's, keys in Python's order", async () => {
  await ready();
  for (const c of cases) {
    const { running, summary } = backtestReport(input(c), { periodsPerYear: c.periods_per_year });
    assert.deepEqual(Object.keys(running), Object.keys(c.running), `${c.name}: running keys`);
    assert.deepEqual(Object.keys(summary), Object.keys(c.summary), `${c.name}: summary keys`);
    for (const [k, v] of Object.entries(c.running)) {
      assert.ok(running[k] instanceof Float64Array, `${c.name}: ${k} is a Float64Array`);
      assertSame(running[k], f64(v), `${c.name}: running.${k}`, close);
    }
    for (const [k, v] of Object.entries(c.summary))
      assert.ok(close(summary[k], num(v)), `${c.name}: summary.${k}: ${summary[k]} vs ${v}`);
  }
});

test("backtestReport: an array of rows gives the NdArray result; null is NaN", async () => {
  await ready();
  const c = cases.find((x) => x.name === "holes")!;
  const nd = input(c);
  const rows: Array<Array<number | null>> = [];
  for (let t = 0; t < c.shape[0]; t++)
    rows.push(Array.from(nd.data.subarray(t * 4, t * 4 + 4), (v) => (Number.isNaN(v) ? null : v)));
  const a = backtestReport(nd, { periodsPerYear: c.periods_per_year });
  const b = backtestReport(rows, { periodsPerYear: c.periods_per_year });
  for (const k of Object.keys(a.running)) assertSame(b.running[k], a.running[k], k);
  for (const k of Object.keys(a.summary)) assert.ok(same(b.summary[k], a.summary[k]), k);
});

test("backtestReport: the report columns are the BacktestReport and BacktestStats operators' columns", async () => {
  await ready();
  const c = cases.find((x) => x.name === "annualized")!;
  const { running } = backtestReport(input(c), { periodsPerYear: 252 });
  const T = c.shape[0];
  const col = (k: number) => Float64Array.from({ length: T }, (_, t) => input(c).data[t * 4 + k]);
  const cols = [col(0), col(1), col(2), col(3)];
  const rep = BacktestReport(252)(...cols) as NdArray;
  const st = BacktestStats(252)(...cols) as NdArray;
  ["drawdown", "cum_cost", "turnover", "trades", "max_drawdown", "sharpe"].forEach((k, j) =>
    assertSame(running[k], Float64Array.from({ length: T }, (_, t) => rep.data[t * 6 + j]), k));
  ["exposure", "round_trips", "win_rate", "profit_factor", "avg_win", "avg_loss", "avg_holding",
   "sortino", "calmar"].forEach((k, j) =>
    assertSame(running[k], Float64Array.from({ length: T }, (_, t) => st.data[t * 9 + j]), k));
});

test("backtestReport: an index is returned as running.index", async () => {
  await ready();
  const c = cases.find((x) => x.name === "base")!;
  const index = Array.from({ length: c.shape[0] }, (_, t) => 10 * t);
  const { running } = backtestReport(input(c), { index });
  assert.deepEqual(Array.from(running.index), index);
  assert.equal(Object.keys(running).at(-1), "index");
});

test("backtestReport: an engine's output reads directly", async () => {
  await ready();
  const price = Float64Array.from({ length: 50 }, (_, t) => 100 + Math.sin(t / 3) * 5);
  const target = Float64Array.from(price, (_, t) => (t % 10 < 5 ? 1 : -1));
  const out = BacktestPriceTarget()(target, price) as NdArray;
  const { running, summary } = backtestReport(out);
  assert.equal(running.equity.length, 50);
  assert.equal(summary.total_pnl, running.equity[49]);
});

test("backtestReport: errors match Python's", async () => {
  await ready();
  const msg = "values must be a (T, 4) array of [equity, pnl, position, cost]";
  const bad = (v: unknown) =>
    assert.throws(() => backtestReport(v as any), (e: any) => e instanceof RangeError && e.message === msg);
  bad({ data: new Float64Array(6), shape: [2, 3] });
  bad({ data: new Float64Array(8), shape: [1, 2, 4] });
  bad({ data: new Float64Array(4), shape: [4] });
  bad(new Float64Array(4));
  bad([1, 2, 3, 4]);
  bad([]);
  bad([[1, 2, 3], [4, 5, 6]]);
  const ok: NdArray = { data: new Float64Array(8), shape: [2, 4] };
  for (const p of [0, -1, NaN, Infinity])
    assert.throws(() => backtestReport(ok, { periodsPerYear: p }),
      (e: any) => e instanceof RangeError && e.message === "periods_per_year must be finite and positive.");
  for (const p of ["252", null])
    assert.throws(() => backtestReport(ok, { periodsPerYear: p as any }),
      (e: any) => e instanceof TypeError && /periodsPerYear is a/.test(e.message));
  assert.throws(() => backtestReport([[1, "2", 3, 4]] as any),
    (e: any) => e instanceof TypeError && /item \[0\]\[1\] of values is a string, expected a number/.test(e.message));
  assert.throws(() => backtestReport([[1, 2, 3, 4], [1, 2, 3]]), RangeError);
  assert.throws(() => backtestReport("abc" as any), TypeError);
});
