import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ready, RollingKurt, RollingSkew, RollingVar, ExpandingKurt, EwKurt, EwVar, RollingCorr, EwCorr,
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
