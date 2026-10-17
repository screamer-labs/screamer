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
