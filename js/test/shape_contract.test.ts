import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { ready } from "../src/index.ts";
import * as api from "../src/index.ts";
import type { NdArray } from "../src/index.ts";

const HERE = dirname(fileURLToPath(import.meta.url));
type Nd = { shape: number[]; data: Array<number | null> };
const cases: Array<{ name: string; args: number[]; inputs: Nd[]; expect: Nd }> =
  JSON.parse(readFileSync(join(HERE, "..", "..", "wasm", "smoke", "shape_oracle.json"), "utf8"));

const toNd = (x: Nd): NdArray => ({ shape: x.shape, data: Float64Array.from(x.data, (v) => (v === null ? NaN : v)) });

for (const c of cases) {
  test(`shape contract: ${c.name} on ${c.inputs.map((i) => `[${i.shape}]`).join(", ")}`, async () => {
    await ready();
    const op = (api as any)[c.name](...c.args);
    const got = op(...c.inputs.map((i) => (i.shape.length === 1 ? toNd(i).data : toNd(i)))) as NdArray | Float64Array;
    const data = got instanceof Float64Array ? got : got.data;
    if (!(got instanceof Float64Array)) assert.deepEqual(got.shape, c.expect.shape);
    const want = toNd(c.expect).data;
    assert.equal(data.length, want.length);
    for (let i = 0; i < want.length; i++) {
      if (Number.isNaN(want[i])) assert.ok(Number.isNaN(data[i]), `index ${i}`);
      else assert.ok(Math.abs(data[i] - want[i]) <= 1e-9, `index ${i}: ${data[i]} vs ${want[i]}`);
    }
  });
}
