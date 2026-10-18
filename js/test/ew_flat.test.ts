// The exponentially weighted ratios of a series that stops varying, against
// the Python binding. Expected values come from wasm/smoke/oracle_ew_flat.json
// (devtools/wasm/gen_ew_flat_oracle.py); tests/test_ew_flat.py has the exact
// references.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { ready } from "../src/index.ts";
import * as api from "../src/index.ts";

await ready();

type Case = { name: string; settings: Record<string, unknown>; inputs: string[]; out: (number | null)[] };
const oracle: { inputs: Record<string, (number | null)[]>; cases: Case[] } =
  JSON.parse(readFileSync(new URL("../../wasm/smoke/oracle_ew_flat.json", import.meta.url), "utf8"));
const series = (k: string) => Float64Array.from(oracle.inputs[k], (v) => (v === null ? NaN : v));
const run = (c: Case) => (api as any)[c.name](c.settings)(...c.inputs.map(series)) as Float64Array;

for (const c of oracle.cases) {
  test(`ew flat: ${c.name}(${c.inputs.join(", ")}) matches Python`, () => {
    const got = run(c);
    assert.equal(got.length, c.out.length);
    for (let i = 0; i < c.out.length; i++) {
      const w = c.out[i];
      if (w === null) assert.ok(Number.isNaN(got[i]), `${c.name} [${i}]: ${got[i]} is not NaN`);
      else assert.ok(Math.abs(got[i] - w) <= 1e-9 * Math.max(1, Math.abs(w)), `${c.name} [${i}]: ${got[i]} vs ${w}`);
    }
  });
}

test("ew flat: a regressor that stops varying gives NaN, then recovers", () => {
  // flat: varying, then 200 rows at 3.7 from row 100, then varying again.
  // At alpha = 0.3 the NaN starts some 130 rows after the series stops.
  const beta = (api as any).EwBeta({ alpha: 0.3 })(series("rand"), series("flat")) as Float64Array;
  const first = beta.findIndex((v, i) => i > 0 && Number.isNaN(v));
  assert.ok(first >= 200 && first < 260, String(first));
  for (let i = first; i < 300; i++) assert.ok(Number.isNaN(beta[i]), String(i));
  for (let i = 300; i < beta.length; i++) assert.ok(Number.isFinite(beta[i]), String(i));
});
