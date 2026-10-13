import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { ready } from "../src/index.ts";
import * as api from "../src/index.ts";
import type { NdArray } from "../src/index.ts";

// The five microstructure operators that reuse another operator's kernel,
// against Python results (devtools/wasm/gen_oracle.py writes the fixture).
const HERE = dirname(fileURLToPath(import.meta.url));
type Nd = { shape: number[]; data: Array<number | null> };
const cases: Array<{
  name: string;
  args: Array<number | null>;
  series: Array<Array<number | null>>;
  packed: Nd;
  batch: Array<number | null>;
  packed_expect: Nd;
  stream: Array<number | null>;
}> = JSON.parse(readFileSync(join(HERE, "..", "..", "wasm", "smoke", "oracle_micro.json"), "utf8"));

const f64 = (xs: Array<number | null>) => Float64Array.from(xs, (v) => (v === null ? NaN : v));

function assertClose(got: ArrayLike<number>, want: Float64Array, what: string) {
  assert.equal(got.length, want.length, `${what}: length`);
  for (let i = 0; i < want.length; i++) {
    if (Number.isNaN(want[i])) assert.ok(Number.isNaN(got[i]), `${what}: index ${i} should be NaN`);
    else assert.ok(Math.abs(got[i] - want[i]) <= 1e-12, `${what}: index ${i}: ${got[i]} vs ${want[i]}`);
  }
}

test("microstructure: the fixture holds the five operators", () => {
  assert.deepEqual(cases.map((c) => c.name).sort(),
    ["EwKyleLambda", "QueueImbalance", "RollingKyleLambda", "RollingOrderImbalance", "SignedVolume"]);
});

for (const c of cases) {
  const make = () => (api as any)[c.name](...c.args.map((a) => (a === null ? NaN : a)));

  test(`microstructure: ${c.name} matches Python on separate inputs`, async () => {
    await ready();
    assertClose(make()(...c.series.map(f64)) as Float64Array, f64(c.batch), c.name);
  });

  test(`microstructure: ${c.name} matches Python on one packed input`, async () => {
    await ready();
    const packed: NdArray = { shape: c.packed.shape, data: f64(c.packed.data) };
    const got = make()(packed) as NdArray;
    assert.deepEqual(got.shape, c.packed_expect.shape);
    assertClose(got.data, f64(c.packed_expect.data), c.name);
  });

  test(`microstructure: ${c.name} matches Python on a stream of scalar events`, async () => {
    await ready();
    const op = make();
    const got = c.stream.map((_, t) => op(...c.series.map((s) => (s[t] === null ? NaN : s[t]))) as number);
    assertClose(got, f64(c.stream), c.name);
  });
}

test("microstructure: the Kyle lambdas are the beta kernels with the inputs swapped", async () => {
  await ready();
  const c = cases.find((x) => x.name === "RollingKyleLambda")!;
  const [flow, ret] = c.series.map(f64);
  assertClose(api.RollingKyleLambda(8)(flow, ret) as Float64Array,
    api.RollingBeta(8)(ret, flow) as Float64Array, "RollingKyleLambda");
  assertClose(api.EwKyleLambda(NaN, 6)(flow, ret) as Float64Array,
    api.EwBeta(NaN, 6)(ret, flow) as Float64Array, "EwKyleLambda");
});

test("microstructure: a string setting is a TypeError", async () => {
  await ready();
  assert.throws(() => (api as any).RollingKyleLambda("8"), TypeError);
  assert.throws(() => (api as any).EwKyleLambda(undefined, "6"), TypeError);
  assert.throws(() => (api as any).RollingOrderImbalance("5"), TypeError);
});
