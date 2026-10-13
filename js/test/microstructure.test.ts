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

// The same C++ kernel runs in both bindings, so the results match exactly.
function assertSame(got: ArrayLike<number>, want: Float64Array, what: string) {
  assert.equal(got.length, want.length, `${what}: length`);
  for (let i = 0; i < want.length; i++) {
    if (Number.isNaN(want[i])) assert.ok(Number.isNaN(got[i]), `${what}: index ${i} should be NaN`);
    else assert.ok(got[i] === want[i], `${what}: index ${i}: ${got[i]} vs ${want[i]}`);
  }
}

function* iter<V>(xs: Iterable<V>) { yield* xs; }
async function* aiter<V>(xs: Iterable<V>) { for (const x of xs) yield x; }
async function drain(it: AsyncIterable<number>) { const out: number[] = []; for await (const v of it) out.push(v); return out; }

test("microstructure: the fixture holds the five operators, NaN inputs and every start policy", () => {
  assert.deepEqual([...new Set(cases.map((c) => c.name))].sort(),
    ["EwKyleLambda", "QueueImbalance", "RollingKyleLambda", "RollingOrderImbalance", "SignedVolume"]);
  for (const name of ["RollingKyleLambda", "RollingOrderImbalance"])
    assert.deepEqual(cases.filter((c) => c.name === name).map((c) => c.args[1]).sort(),
      ["expanding", "strict", "zero"]);
  assert.ok(cases.every((c) => c.series.some((s) => s.includes(null))), "every case has a NaN input");
});

for (const c of cases) {
  const make = () => (api as any)[c.name](...c.args.map((a) => (a === null ? NaN : a)));
  const label = `${c.name}(${c.args.map((a) => (a === null ? "NaN" : JSON.stringify(a))).join(", ")})`;

  test(`microstructure: ${label} matches Python on separate inputs`, async () => {
    await ready();
    assertSame(make()(...c.series.map(f64)) as Float64Array, f64(c.batch), c.name);
  });

  test(`microstructure: ${label} matches Python on one packed input`, async () => {
    await ready();
    const packed: NdArray = { shape: c.packed.shape, data: f64(c.packed.data) };
    const got = make()(packed) as NdArray;
    assert.deepEqual(got.shape, c.packed_expect.shape);
    assertSame(got.data, f64(c.packed_expect.data), c.name);
  });

  test(`microstructure: ${label} matches Python on a stream of scalar events`, async () => {
    await ready();
    const op = make();
    const got = c.stream.map((_, t) => op(...c.series.map((s) => (s[t] === null ? NaN : s[t]))) as number);
    assertSame(got, f64(c.stream), c.name);
  });

  test(`microstructure: ${label} matches Python on iterators and async iterators`, async () => {
    await ready();
    const cols = c.series.map(f64);
    const events = Array.from({ length: cols[0].length }, (_, t) => cols.map((col) => col[t]));
    const want = f64(c.stream);
    assertSame([...make()(...cols.map((col) => iter(col)))], want, `${c.name} iterators`);
    assertSame(await drain(make()(...cols.map((col) => aiter(col)))), want, `${c.name} async iterators`);
    if (cols.length > 1) {
      assertSame([...make()(iter(events))], want, `${c.name} iterator of events`);
      assertSame(await drain(make()(aiter(events))), want, `${c.name} async iterator of events`);
    }
  });
}

test("microstructure: the Kyle lambdas are the beta kernels with the inputs swapped", async () => {
  await ready();
  const c = cases.find((x) => x.name === "RollingKyleLambda")!;
  const [flow, ret] = c.series.map(f64);
  assertSame(api.RollingKyleLambda(8)(flow, ret) as Float64Array,
    api.RollingBeta(8)(ret, flow) as Float64Array, "RollingKyleLambda");
  assertSame(api.EwKyleLambda(NaN, 6)(flow, ret) as Float64Array,
    api.EwBeta(NaN, 6)(ret, flow) as Float64Array, "EwKyleLambda");
});

test("microstructure: a string setting is a TypeError", async () => {
  await ready();
  assert.throws(() => (api as any).RollingKyleLambda("8"), TypeError);
  assert.throws(() => (api as any).EwKyleLambda(undefined, "6"), TypeError);
  assert.throws(() => (api as any).RollingOrderImbalance("5"), TypeError);
});
