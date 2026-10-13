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
