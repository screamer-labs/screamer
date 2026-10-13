import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { ready, split } from "../src/index.ts";

// split against Python's split (devtools/wasm/gen_oracle.py writes the
// fixture). Both call the one C++ partition, so every value matches exactly.
const HERE = dirname(fileURLToPath(import.meta.url));
type Num = number | null;
interface Case {
  name: string;
  values: Num[];
  sources: number[];
  index: Num[] | null;
  n: number | null;
  parts: Array<{ values: Num[]; index: Num[] | null }>;
}
const cases: Case[] = JSON.parse(
  readFileSync(join(HERE, "..", "..", "wasm", "smoke", "oracle_split.json"), "utf8"));

const f64 = (xs: Num[]) => Float64Array.from(xs, (v) => (v === null ? NaN : v));
const same = (a: number, b: number) => (Number.isNaN(b) ? Number.isNaN(a) : Object.is(a, b) || a === b);

function assertSame(got: ArrayLike<number>, want: Float64Array, what: string) {
  assert.equal(got.length, want.length, `${what}: length`);
  for (let i = 0; i < want.length; i++)
    assert.ok(same(got[i], want[i]), `${what}: item ${i}: ${got[i]} vs ${want[i]}`);
}

function run(c: Case) {
  const opts: { index?: Float64Array; n?: number } = {};
  if (c.index !== null) opts.index = f64(c.index);
  if (c.n !== null) opts.n = c.n;
  return split(f64(c.values), c.sources, opts);
}

test("split: the fixture covers positional, int and float indexes, n > max+1 and no events", () => {
  const names = cases.map((c) => c.name);
  for (const want of ["positional", "int index", "float index", "n larger than max(sources)+1", "no events"])
    assert.ok(names.includes(want), want);
  const larger = cases.find((c) => c.name === "n larger than max(sources)+1")!;
  assert.ok(larger.n! > Math.max(...larger.sources) + 1);
});

test("split: every fixture case matches Python exactly", async () => {
  await ready();
  for (const c of cases) {
    const parts = run(c);
    assert.equal(parts.length, c.parts.length, `${c.name}: number of outputs`);
    parts.forEach(([v, k], i) => {
      const want = c.parts[i];
      assert.ok(v instanceof Float64Array, `${c.name}: output ${i} values`);
      assertSame(v as Float64Array, f64(want.values), `${c.name}: output ${i} values`);
      if (want.index === null) assert.equal(k, null, `${c.name}: output ${i} index`);
      else assertSame(k!, f64(want.index), `${c.name}: output ${i} index`);
    });
  }
});

test("split: n larger than max(sources)+1 gives empty trailing outputs", async () => {
  await ready();
  const parts = split([1, 2, 3], [0, 1, 0], { n: 4 });
  assert.equal(parts.length, 4);
  assert.deepEqual(parts.map(([v]) => Array.from(v as Float64Array)), [[1, 3], [2], [], []]);
  assert.deepEqual(parts.map(([, k]) => k), [null, null, null, null]);
});

test("split: n too small is a RangeError with Python's message", async () => {
  await ready();
  assert.throws(() => split([1, 2, 3], [0, 3, 1], { n: 2 }), {
    name: "RangeError",
    message: "split: n=2 is too small for sources up to 3; events would be dropped",
  });
  assert.throws(() => split([1], [0], { n: -1 }), RangeError);
  assert.throws(() => split([], [], { n: -1 }), { name: "RangeError", message: "split: n must be non-negative" });
  assert.deepEqual(split([], [], { n: 0 }), []);
  assert.deepEqual(split([], []), []);
});

test("split: values take null as NaN, an NdArray of one dimension, and a typed array", async () => {
  await ready();
  const [[a], [b]] = split([1, null, 3], [0, 1, 0]);
  assert.deepEqual(Array.from(a as Float64Array), [1, 3]);
  assert.ok(Number.isNaN((b as Float64Array)[0]));
  const [[c]] = split({ data: new Float64Array([4, 5]), shape: [2] }, new Uint32Array([0, 0]));
  assert.deepEqual(Array.from(c as Float64Array), [4, 5]);
});

test("split: the outputs are copies, not views of the WASM heap", async () => {
  await ready();
  const [[v, k]] = split([1, 2], [0, 0], { index: [10, 20] });
  assert.equal((v as Float64Array).buffer.byteLength, (v as Float64Array).byteLength);
  assert.equal(k!.buffer.byteLength, k!.byteLength);
});

test("split: a wrong kind is a TypeError, a shape mismatch a RangeError", async () => {
  await ready();
  // values
  assert.throws(() => split(["a", 1] as any, [0, 1]), TypeError);
  assert.throws(() => split({ data: new Float64Array(4), shape: [2, 2] }, [0, 1]), RangeError);
  assert.throws(() => split([[1, 2], [3, 4]] as any, [0, 1]), RangeError);
  // sources: whole numbers from 0 to 2**32 - 1
  for (const bad of [[0.5, 0], [NaN, 0], [null, 0], [-1, 0], [2 ** 32, 0], ["0", 1]])
    assert.throws(() => split([1, 2], bad as any), TypeError, JSON.stringify(bad));
  assert.throws(() => split([1, 2], [0.5, 0]), {
    message: "split: a source must be a whole number from 0 to 4294967295, got 0.5",
  });
  // lengths
  assert.throws(() => split([1, 2], [0]), { name: "RangeError", message: "split: values/sources/index length differ" });
  assert.throws(() => split([1, 2], [0, 1], { index: [1] }), RangeError);
  // index
  assert.throws(() => split([1, 2], [0, 1], { index: ["a", "b"] as any }), TypeError);
  // options
  assert.throws(() => split([1], [0], { m: 1 } as any), { name: "TypeError", message: "split: unknown option 'm'" });
  assert.throws(() => split([1], [0], { n: "3" as any }), TypeError);
  assert.throws(() => split([1], [0], { n: 1.5 }), TypeError);
  assert.throws(() => split([1], [0], 3 as any), TypeError);
});
