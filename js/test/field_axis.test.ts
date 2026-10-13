// A12: every multi-output operator puts the field axis last (rule 1). On
// (T, K) inputs a batch call gives (T, K, F), and a stream of (K,) events
// gives one (K, F) NdArray per event, the rows of the batch result. The
// Python results come from wasm/smoke/field_axis_oracle.json
// (devtools/wasm/gen_field_axis_oracle.py).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { ready } from "../src/index.ts";
import * as api from "../src/index.ts";
import type { NdArray } from "../src/index.ts";

type Case = { name: string; base: Record<string, number>; inputs: number; assets: number; fields: number; tail: Array<number | null> };
const oracle: { T: number; tail: number; inputs: Array<Array<number | null>>; cases: Case[] } =
  JSON.parse(readFileSync(new URL("../../wasm/smoke/field_axis_oracle.json", import.meta.url), "utf8"));
const { T, tail } = oracle;
const camel = (k: string) => k.replace(/_([a-z])/g, (_m, c) => c.toUpperCase());

// The first k columns of a (T, 3) oracle input, as a (T, k) NdArray.
const columns = (x: Array<number | null>, k: number): NdArray => {
  const data = new Float64Array(T * k);
  for (let t = 0; t < T; t++) for (let j = 0; j < k; j++) data[t * k + j] = x[t * 3 + j] ?? NaN;
  return { data, shape: [T, k] };
};

const close = (got: ArrayLike<number>, want: Array<number | null>, what: string) => {
  assert.equal(got.length, want.length, what);
  for (let i = 0; i < want.length; i++) {
    const w = want[i];
    if (w === null) assert.ok(Number.isNaN(got[i]), `${what} [${i}]: ${got[i]} is not NaN`);
    else assert.ok(Math.abs(got[i] - w) <= 1e-9 * Math.max(1, Math.abs(w)), `${what} [${i}]: ${got[i]} vs ${w}`);
  }
};

test("A12: the oracle covers the multi-output operators", () => {
  assert.ok(oracle.cases.length >= 30, `${oracle.cases.length} cases`);
});

for (const c of oracle.cases) {
  test(`A12: ${c.name} gives (T, K, F) in batch and (K, F) per event`, async () => {
    await ready();
    const opts: Record<string, number> = {};
    for (const [k, v] of Object.entries(c.base)) opts[camel(k)] = v;
    const make = () => (api as any)[c.name](opts);
    const xs = oracle.inputs.slice(0, c.inputs).map((x) => columns(x, c.assets));
    const per = c.assets * c.fields;

    const op = make();
    const batch = op(...xs) as NdArray;
    op.dispose();
    assert.deepEqual(batch.shape, [T, c.assets, c.fields]);
    close(batch.data.subarray((T - tail) * per), c.tail, `${c.name} batch`);

    const events = (x: NdArray) => (function* () {
      for (let t = 0; t < T; t++) yield x.data.slice(t * c.assets, (t + 1) * c.assets);
    })();
    const lazyOp = make();
    const rows = [...lazyOp(...xs.map(events))] as NdArray[];
    lazyOp.dispose();
    assert.equal(rows.length, T);
    for (let t = 0; t < T; t++) {
      assert.deepEqual(rows[t].shape, [c.assets, c.fields], `${c.name} event ${t}`);
      assert.deepEqual(Array.from(rows[t].data), Array.from(batch.data.subarray(t * per, (t + 1) * per)),
        `${c.name} event ${t} equals the batch row`);
    }
  });
}
