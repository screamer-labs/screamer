// The research-toolkit operators and backtest helpers against the Python
// binding. Expected values come from wasm/smoke/oracle_research.json
// (devtools/wasm/gen_research_oracle.py).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { ready, Hold, CusumFilter } from "../src/index.ts";
import * as api from "../src/index.ts";

await ready();

type Num = number | null | "inf" | "-inf";
type Case = { name: string; settings: Record<string, unknown>; input: string; shape: number[];
              out: Num[]; tol?: number };
const oracle: { inputs: Record<string, Num[]>; cases: Case[]; helpers: Record<string, any> } =
  JSON.parse(readFileSync(new URL("../../wasm/smoke/oracle_research.json", import.meta.url), "utf8"));
const num = (v: Num): number => (v === null ? NaN : v === "inf" ? Infinity : v === "-inf" ? -Infinity : v);
const series = (k: string) => Float64Array.from(oracle.inputs[k], num);
const camel = (k: string) => k.replace(/_([a-z])/g, (_m, c) => c.toUpperCase());
const options = (s: Record<string, unknown>) =>
  Object.fromEntries(Object.entries(s).map(([k, v]) => [camel(k), v]));

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
    const out = make(options(c.settings))(series(c.input));
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
