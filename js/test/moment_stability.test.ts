import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ready, RollingKurt, RollingSkew, RollingVar, ExpandingKurt, EwKurt, EwVar, RollingCorr, EwCorr,
  RollingPoly1, RollingPoly2, RollingTSF, ExpandingSlope, RollingOU, RollingADF, RollingZscore,
} from "../src/index.ts";

// The moment operators take their sums about an origin near the mean (C++
// core, shared with Python; tests/test_moment_stability.py has the exact
// references). Shifting the input by a large level leaves these statistics
// unchanged, so the result at level 1e6 must match the one at level 0.
const T = 600;
let seed = 12345;
const rand = () => {
  seed = (seed * 1103515245 + 12345) % 2147483648;
  return seed / 2147483648 - 0.5;
};
const e = Float64Array.from({ length: T }, rand);
const f = Float64Array.from({ length: T }, rand);
const shift = (x: Float64Array, level: number) => x.map((v) => v + level);

function maxDiff(a: Float64Array, b: Float64Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    if (Number.isNaN(a[i]) && Number.isNaN(b[i])) continue;
    m = Math.max(m, Math.abs(a[i] - b[i]));
  }
  return m;
}

test("moments: shift by 1e6 leaves skewness, kurtosis and variance unchanged", async () => {
  await ready();
  const ops: [string, () => (x: Float64Array) => unknown][] = [
    ["RollingKurt", () => RollingKurt(50)], ["RollingSkew", () => RollingSkew(50)],
    ["RollingVar", () => RollingVar(50)], ["ExpandingKurt", () => ExpandingKurt()],
    ["EwKurt", () => EwKurt(NaN, 50)], ["EwVar", () => EwVar(NaN, 50)],
  ];
  for (const [name, make] of ops) {
    const at0 = make()(e) as Float64Array;
    const at6 = make()(shift(e, 1e6)) as Float64Array;
    // Raw power sums give errors of order 1 or more here.
    assert.ok(maxDiff(at0, at6) < 1e-8, `${name}: ${maxDiff(at0, at6)}`);
  }
});

test("moments: shift by 1e6 leaves two-input correlations unchanged", async () => {
  await ready();
  for (const make of [() => RollingCorr(50), () => EwCorr(NaN, 50)]) {
    const at0 = make()(e, f) as Float64Array;
    const at6 = make()(shift(e, 1e6), shift(f, 1e6)) as Float64Array;
    assert.ok(maxDiff(at0, at6) < 1e-8, String(maxDiff(at0, at6)));
  }
});

// The regressions (tests/test_regression_stability.py has the exact
// references): slopes, curvatures and the OU rate, noise and ADF statistic do
// not move with the level; fitted values and the OU mean move by it.
const ar = (() => {
  const x = new Float64Array(T);
  for (let i = 1; i < T; i++) x[i] = 0.9 * x[i - 1] + e[i];
  return x;
})();
const unshift = (x: Float64Array, level: number) => x.map((v) => v - level);

test("regressions: shift by 1e6 leaves slopes unchanged and moves fitted values by the shift", async () => {
  await ready();
  const ops: [string, () => (x: Float64Array) => unknown, boolean][] = [
    ["RollingPoly1 slope", () => RollingPoly1(50, 1), false],
    ["RollingPoly1 endpoint", () => RollingPoly1(50, 0, "expanding"), true],
    ["RollingPoly2 curvature", () => RollingPoly2(50, 2), false],
    ["RollingPoly2 slope", () => RollingPoly2(50, 1, "expanding"), false],
    ["RollingPoly2 endpoint", () => RollingPoly2(50, 0), true],
    ["RollingTSF", () => RollingTSF(50), true],
    ["ExpandingSlope", () => ExpandingSlope(), false],
  ];
  for (const [name, make, moves] of ops) {
    const at0 = make()(e) as Float64Array;
    let at6 = make()(shift(e, 1e6)) as Float64Array;
    if (moves) at6 = unshift(at6, 1e6);
    // Raw sums gave up to 4e-7 here.
    assert.ok(maxDiff(at0, at6) < 1e-8, `${name}: ${maxDiff(at0, at6)}`);
  }
});

test("regressions: RollingOU and RollingADF follow a level shift", async () => {
  await ready();
  for (const output of ["mrr", "mean", "relmean", "std"]) {
    const at0 = RollingOU(50, output)(ar) as Float64Array;
    let at6 = RollingOU(50, output)(shift(ar, 1e6)) as Float64Array;
    if (output === "mean") at6 = unshift(at6, 1e6);
    // Before the fix the mean moved by 1e6 / window and the std by 1e5.
    assert.ok(maxDiff(at0, at6) < 1e-7, `RollingOU ${output}: ${maxDiff(at0, at6)}`);
  }
  const a0 = RollingADF(60, 1)(ar) as Float64Array;
  const a6 = RollingADF(60, 1)(shift(ar, 1e6)) as Float64Array;
  assert.ok(Number.isFinite(a6[T - 1]) && maxDiff(a0, a6) < 1e-8, String(maxDiff(a0, a6)));
});

test("RollingZscore: start_policy as RollingMean and RollingStd", async () => {
  await ready();
  const strict = RollingZscore(10)(e) as Float64Array;
  const expanding = RollingZscore(10, "expanding")(e) as Float64Array;
  const zero = RollingZscore(10, "zero")(e) as Float64Array;
  assert.ok(strict.slice(0, 9).every(Number.isNaN));
  assert.ok(Number.isNaN(expanding[0]) && expanding.slice(1).every(Number.isFinite));
  assert.ok(zero.every(Number.isFinite));
  assert.ok(maxDiff(strict.slice(9), expanding.slice(9)) === 0);
});
