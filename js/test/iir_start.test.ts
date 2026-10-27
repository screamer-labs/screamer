import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import * as S from "../src/index.ts";

// startPolicy on the seven IIR filters against Python
// (devtools/wasm/gen_iir_start_oracle.py writes the fixture;
// tests/test_iir_start_oracle_fresh.py keeps it current). Both run the one
// C++ kernel; the Butterworth coefficients pass through sin, cos and exp, so
// values compare with a relative tolerance of 1e-9. NaN must sit at the same
// positions.
const HERE = dirname(fileURLToPath(import.meta.url));
type Num = number | null;
interface Fixture {
  input: Num[];
  cases: { op: string; options: Record<string, number | string>; values: Num[] }[];
}
const fx: Fixture = JSON.parse(
  readFileSync(join(HERE, "..", "..", "wasm", "smoke", "oracle_iir_start.json"), "utf8"));
const x = Float64Array.from(fx.input, (v) => (v === null ? NaN : v));

test("iir start: the fixture covers seven filters under both policies", () => {
  assert.equal(fx.cases.length, 14);
});

for (const c of fx.cases) {
  test(`iir start: ${c.op} startPolicy=${c.options.startPolicy} matches Python`, async () => {
    await S.ready();
    const got: number[] = Array.from((S as any)[c.op](c.options)(x));
    assert.equal(got.length, c.values.length);
    for (let i = 0; i < got.length; i++) {
      const want = c.values[i];
      if (want === null) {
        assert.ok(Number.isNaN(got[i]), `${c.op}[${i}]: expected NaN, got ${got[i]}`);
      } else {
        const tol = 1e-9 * Math.max(1, Math.abs(want));
        assert.ok(Math.abs(got[i] - want) <= tol, `${c.op}[${i}]: ${got[i]} vs ${want}`);
      }
    }
  });
}

test("iir start: the default is steady", async () => {
  await S.ready();
  const a = Array.from(S.SuperSmoother({ period: 10 })(x));
  const b = Array.from(S.SuperSmoother({ period: 10, startPolicy: "steady" })(x));
  assert.deepEqual(a, b);
});

test("iir start: an unknown policy throws naming the filter", async () => {
  await S.ready();
  assert.throws(() => S.Butter({ order: 2, cutoffFreq: 0.1, startPolicy: "bogus" }), /Butter/);
});
