import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import * as S from "../src/index.ts";

// The Ehlers operators against Python (devtools/wasm/gen_ehlers_oracle.py
// writes the fixture; tests/test_ehlers_oracle_fresh.py keeps it current).
// Both run the one C++ kernel. Values that pass through sin, cos or atan can
// differ in the last bits between the two math libraries, so they compare
// with a relative tolerance of 1e-9; TrendMode and InstantaneousTrendline use
// none of those functions after the period and compare exactly. NaN must sit
// at the same positions.
const HERE = dirname(fileURLToPath(import.meta.url));
type Num = number | null;
interface Case {
  name: string;
  input: Num[];
  results: Record<string, { values: Num[]; shape: number[] }>;
}
const cases: Case[] = JSON.parse(
  readFileSync(join(HERE, "..", "..", "wasm", "smoke", "oracle_ehlers.json"), "utf8"));
const EXACT = new Set(["TrendMode", "InstantaneousTrendline"]);

test("ehlers: the fixture covers the five operators on two inputs", () => {
  assert.equal(cases.length, 2);
  for (const c of cases) {
    assert.deepEqual(Object.keys(c.results).sort(),
      ["CycleSine", "HilbertPhasor", "InstantaneousTrendline", "MAMA", "TrendMode"]);
  }
});

for (const c of cases) {
  for (const [op, ref] of Object.entries(c.results)) {
    test(`ehlers: ${op} matches Python on ${c.name}`, async () => {
      await S.ready();
      const x = Float64Array.from(c.input, (v) => (v === null ? NaN : v));
      const r: any = (S as any)[op]()(x);
      const got: number[] = Array.from(r.data ?? r);
      assert.deepEqual(r.shape ?? [got.length], ref.shape);
      assert.equal(got.length, ref.values.length);
      for (let i = 0; i < got.length; i++) {
        const want = ref.values[i];
        if (want === null) {
          assert.ok(Number.isNaN(got[i]), `${op}[${i}]: expected NaN, got ${got[i]}`);
        } else if (EXACT.has(op)) {
          assert.equal(got[i], want, `${op}[${i}]`);
        } else {
          const tol = 1e-9 * Math.max(1, Math.abs(want));
          assert.ok(Math.abs(got[i] - want) <= tol, `${op}[${i}]: ${got[i]} vs ${want}`);
        }
      }
    });
  }
}
