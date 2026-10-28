import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { ready, backtestReport, backtestTrades } from "../src/index.ts";
import type { NdArray } from "../src/index.ts";

// backtestReport on a portfolio engine's [T, K, 4] output against Python
// (devtools/wasm/gen_book_report_oracle.py writes the fixture;
// tests/test_book_report_oracle_fresh.py keeps it current). Both label the
// same C++ book report. Native clang fuses a multiply and an add in the
// running updates, which WebAssembly does not, so values compare with a
// relative tolerance of 1e-9; NaN and infinities sit in the same places.
const HERE = dirname(fileURLToPath(import.meta.url));
type Num = number | null | "inf" | "-inf";
type Stats = Record<string, Num>;
const cases: Array<{
  name: string;
  shape: number[];
  values: Num[];
  periods_per_year: number;
  running: Record<string, Num[]>;
  summary: Stats;
  per_asset: Stats[];
  by?: number[];
  by_group?: Array<[number, Stats]>;
}> = JSON.parse(
  readFileSync(join(HERE, "..", "..", "wasm", "smoke", "oracle_book_report.json"), "utf8"));

const num = (v: Num) => (v === null ? NaN : v === "inf" ? Infinity : v === "-inf" ? -Infinity : v);
function close(a: number, b: number): boolean {
  if (Number.isNaN(b)) return Number.isNaN(a);
  if (!Number.isFinite(b)) return a === b;
  return Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b));
}
function assertStats(got: Record<string, number>, want: Stats, what: string) {
  assert.deepEqual(Object.keys(got), Object.keys(want), `${what}: keys`);
  for (const [k, v] of Object.entries(want))
    assert.ok(close(got[k], num(v)), `${what}.${k}: ${got[k]} vs ${v}`);
}
const input = (c: (typeof cases)[number]): NdArray =>
  ({ data: Float64Array.from(c.values, num), shape: c.shape });

test("backtestReport book: the fixture covers a late asset, groups, one asset and no rows", () => {
  assert.deepEqual(cases.map((c) => c.name),
    ["base", "late asset", "grouped", "one asset", "no rows"]);
});

for (const c of cases) {
  test(`backtestReport book: ${c.name} matches Python`, async () => {
    await ready();
    const r = backtestReport(input(c), { periodsPerYear: c.periods_per_year, by: c.by });
    assert.deepEqual(Object.keys(r.running), Object.keys(c.running), "running keys");
    for (const [k, v] of Object.entries(c.running)) {
      assert.equal(r.running[k].length, v.length, `running.${k}: length`);
      v.forEach((w, t) => assert.ok(close(r.running[k][t], num(w)),
        `running.${k}[${t}]: ${r.running[k][t]} vs ${w}`));
    }
    assertStats(r.summary, c.summary, "summary");
    assert.equal(r.perAsset!.length, c.per_asset.length);
    c.per_asset.forEach((p, k) => assertStats(r.perAsset![k], p, `perAsset[${k}]`));
    if (c.by_group === undefined) {
      assert.equal(r.byGroup, undefined);
    } else {
      assert.deepEqual([...r.byGroup!.keys()], c.by_group.map(([label]) => label));
      for (const [label, g] of c.by_group) assertStats(r.byGroup!.get(label)!, g, `byGroup ${label}`);
    }
  });
}

test("backtestReport book: nested arrays read as the NdArray", async () => {
  await ready();
  const c = cases.find((x) => x.name === "late asset")!;
  const nd = input(c);
  const [T, K] = c.shape;
  const rows = Array.from({ length: T }, (_, t) => Array.from({ length: K }, (_, k) =>
    Array.from(nd.data.subarray((t * K + k) * 4, (t * K + k) * 4 + 4),
               (v) => (Number.isNaN(v) ? null : v))));
  const a = backtestReport(nd, { periodsPerYear: 365 });
  const b = backtestReport(rows, { periodsPerYear: 365 });
  for (const k of Object.keys(a.running))
    a.running[k].forEach((v, t) => assert.ok(Object.is(v, b.running[k][t]) ||
      (Number.isNaN(v) && Number.isNaN(b.running[k][t])), `${k}[${t}]`));
  assert.deepEqual(b.perAsset!.length, K);
});

test("backtestReport book: a single-asset result has no perAsset", async () => {
  await ready();
  const r = backtestReport({ data: new Float64Array(8), shape: [2, 4] });
  assert.equal(r.perAsset, undefined);
});

test("backtestReport book: wrong shapes throw naming the function", async () => {
  await ready();
  const msg = "backtestReport: values must be a (T, 4) or (T, K, 4) array of " +
    "[equity, pnl, position, cost]";
  for (const shape of [[2, 3, 3], [2, 1, 1, 4], [4]])
    assert.throws(
      () => backtestReport({ data: new Float64Array(shape.reduce((a, b) => a * b, 1)), shape }),
      (e: any) => e instanceof RangeError && e.message === msg);
  assert.throws(() => backtestReport({ data: new Float64Array(0), shape: [3, 0, 4] }),
    (e: any) => e instanceof RangeError && /backtestReport: .*at least one asset/.test(e.message));
  assert.throws(() => backtestReport([[[1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]]),
    (e: any) => e instanceof RangeError && /must hold 1 asset rows, got 2 at row 1/.test(e.message));
  assert.throws(() => backtestTrades({ data: new Float64Array(8), shape: [1, 2, 4] }),
    /once per asset/);
});
