import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ready, Input, Pipeline, resample, RollingMean, BollingerBands, SchmittTrigger, BacktestPriceTarget,
  probabilisticSharpe, circularShift, segmentShuffle, permuteReturns, positionTest, walkForward,
  backtestReport,
} from "../src/index.ts";

// Task 31 fix round 1. Mirrors tests/test_bool_settings.py and
// tests/test_setting_kinds.py.

const typeError = (message: string) => ({ name: "TypeError", message });

test("an integer setting refuses a fraction, as in Python", async () => {
  await ready();
  assert.throws(() => RollingMean(2.5), typeError("RollingMean: windowSize must be an integer, got 2.5"));
  assert.throws(() => RollingMean(NaN), typeError("RollingMean: windowSize must be an integer, got NaN"));
  RollingMean(3).dispose();
});

test("a function's number argument refuses a boolean (G25 a)", async () => {
  await ready();
  const pnl = Float64Array.from({ length: 50 }, (_, i) => Math.sin(i) * 0.01);
  const x = Float64Array.from(pnl, (v) => Math.sign(v));
  const p = Float64Array.from(pnl, (v, i) => 100 + i + v);
  const cases: [string, () => unknown][] = [
    ["probabilisticSharpe: benchmark must be a number", () => probabilisticSharpe(pnl, true as any)],
    ["circularShift: n must be an integer", () => circularShift(x, true as any)],
    ["circularShift: seed must be an integer", () => circularShift(x, 3, { seed: true as any })],
    ["segmentShuffle: n must be an integer", () => segmentShuffle(x, true as any)],
    ["permuteReturns: seed must be an integer", () => permuteReturns(p, 2, { seed: true as any })],
    ["positionTest: n must be an integer", () => positionTest(BacktestPriceTarget(), x, [p], { n: true as any })],
    ["walkForward: train must be an integer",
      () => walkForward(BacktestPriceTarget(), x, [p], { train: true as any, test: 5 })],
    ["backtestReport: periodsPerYear must be a number",
      () => backtestReport({ data: new Float64Array(8), shape: [2, 4] }, { periodsPerYear: true as any })],
  ];
  for (const [msg, call] of cases) assert.throws(call, typeError(`${msg}, got boolean`), msg);
});

test("a setting documented as finite refuses an infinity and NaN (b)", async () => {
  await ready();
  for (const bad of [Infinity, -Infinity, NaN])
    assert.throws(() => BollingerBands(20, bad), RangeError, `numStd ${bad}`);
  assert.throws(() => SchmittTrigger(-Infinity, 3), RangeError);
  assert.throws(() => SchmittTrigger(-3, Infinity), RangeError);
  BollingerBands(20, 0).dispose();
});

test("a Pipeline call takes one argument (c)", async () => {
  await ready();
  const x = Input("x");
  const p = new Pipeline([x], [RollingMean(2)(x)]);
  try {
    assert.throws(() => (p as any)([1, 2, 3], [0, 1, 2]),
      typeError("Pipeline: takes one argument, the feeds, got 2; pass an index as {values, index} " +
        "and several inputs as an object keyed by input name"));
    p({ values: [1, 2, 3], index: [0, 1, 2] });
  } finally { p.dispose(); }
});

test("fill applies to every= and clock= bars only (Q9)", async () => {
  await ready();
  const x = Input("x");
  for (const fill of ["nan", "carry"] as const)
    assert.throws(() => new Pipeline([x], [resample(x, { count: 2, fill })]),
      (e: any) => e instanceof RangeError && /a count= bar is never empty/.test(e.message));
  new Pipeline([x], [resample(x, { count: 2, fill: "skip" })]).dispose();
});

test("a setting that is not finite has one message, with the name passed (N1)", async () => {
  await ready();
  const { EwMean, Hold, BacktestPortfolioTarget } = await import("../src/index.ts");
  const rangeError = (message: string) => ({ name: "RangeError", message });
  assert.throws(() => BollingerBands(20, Infinity), rangeError("BollingerBands: numStd must be finite, got Infinity"));
  assert.throws(() => BollingerBands({ numStd: NaN }), rangeError("BollingerBands: numStd must be finite, got NaN"));
  assert.throws(() => EwMean({ span: Infinity }), rangeError("EwMean: span must be finite, got Infinity"));
  assert.throws(() => SchmittTrigger(-Infinity, 3), rangeError("SchmittTrigger: lower must be finite, got -Infinity"));
  assert.throws(() => Hold(3, Infinity), rangeError("Hold: release must be finite or NaN, got Infinity"));
  assert.throws(() => BacktestPortfolioTarget({ fee: [0.1, Infinity] }),
    rangeError("BacktestPortfolioTarget: fee must be finite, got Infinity"));
  // Another error keeps its message; NaN for an unset optional is not a value.
  assert.throws(() => EwMean(), (e: any) => e instanceof RangeError && !/must be finite/.test(e.message));
  Hold(3, NaN).dispose();
});

test("a trailing undefined is not an argument (N3)", async () => {
  await ready();
  const x = Input("x");
  const p = new Pipeline([x], [RollingMean(2)(x)]);
  try {
    (p as any)([1, 2, 3], undefined);
    (p as any)([1, 2, 3], undefined, undefined);
    assert.throws(() => (p as any)([1, 2, 3], undefined, 1), TypeError);
  } finally { p.dispose(); }
});

test("every float setting refused as Infinity or NaN has the one message (N1)", async () => {
  await ready();
  const { readFileSync } = await import("node:fs");
  const ops: any = await import("../src/generated/ops.ts");
  const oracle: { name: string; base: Record<string, number>; bool_message: Record<string, string>;
                  accepts_none: Record<string, boolean> }[] =
    JSON.parse(readFileSync(new URL("../../wasm/smoke/null_oracle.json", import.meta.url), "utf8"));
  const camel = (k: string) => k.replace(/_([a-z])/g, (_m, c) => c.toUpperCase());
  const ew = new Set(["com", "span", "halflife", "alpha"]);
  let refused = 0;
  for (const { name, base, bool_message, accepts_none } of oracle)
    for (const [k, msg] of Object.entries(bool_message)) {
      if (!/must be a number/.test(msg)) continue;            // the float settings
      const opts: Record<string, unknown> = {};
      for (const [b, v] of Object.entries(base)) if (!(ew.has(k) && ew.has(b))) opts[camel(b)] = v;
      for (const [bad, text] of [[Infinity, "Infinity"], [-Infinity, "-Infinity"], [NaN, "NaN"]] as const) {
        if (Number.isNaN(bad) && accepts_none[k]) continue;   // NaN is "not given" for an optional setting
        const value = /sequence/.test(msg) && name === "MovingAverage" ? [bad] : bad;
        try { ops[name]({ ...opts, [camel(k)]: value }).dispose(); continue; }
        catch (e) {
          if (!(e instanceof RangeError)) throw e;
          refused++;
          assert.match((e as Error).message, new RegExp(`^${name}: ${camel(k)} must be finite( or [-A-Za-z]+)*, got ${text}$`), `${name}(${k}=${text})`);
        }
      }
    }
  assert.ok(refused > 200, `${refused}`);
});
