import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { ready, split } from "../src/index.ts";
import type { NdArray } from "../src/index.ts";

// split against Python's split (devtools/wasm/gen_oracle.py writes the
// fixture). Both call the one C++ partition, so every value matches exactly.
const HERE = dirname(fileURLToPath(import.meta.url));
type Num = number | null | "inf" | "-inf";
interface Case {
  name: string;
  shape: number[];
  values: Num[];
  sources: number[];
  index: Num[] | null;
  n: number | null;
  parts: Array<{ shape: number[]; values: Num[]; index: Num[] | null }>;
}
const cases: Case[] = JSON.parse(
  readFileSync(join(HERE, "..", "..", "wasm", "smoke", "oracle_split.json"), "utf8"));

const num = (v: Num) => (v === null ? NaN : v === "inf" ? Infinity : v === "-inf" ? -Infinity : v);
const f64 = (xs: Num[]) => Float64Array.from(xs, num);
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
  const data = f64(c.values);
  return split(c.shape.length > 1 ? { data, shape: c.shape } : data, c.sources, opts);
}

test("split: the fixture covers indexes, n > max+1, no events and rows", () => {
  const names = cases.map((c) => c.name);
  for (const want of ["positional", "int index", "float index", "NaN and -inf index",
    "n larger than max(sources)+1", "no events", "(T, 3) rows", "(T, 1) rows", "(T, 2, 2) rows", "(0, 3) rows"])
    assert.ok(names.includes(want), want);
  const larger = cases.find((c) => c.name === "n larger than max(sources)+1")!;
  assert.ok(larger.n! > Math.max(...larger.sources) + 1);
});

test("split: every fixture case matches Python exactly", async () => {
  await ready();
  for (const c of cases) {
    const parts = run(c);
    assert.equal(parts.length, c.parts.length, `${c.name}: number of outputs`);
    parts.forEach(({ values: v, index: k }, i) => {
      const want = c.parts[i];
      if (c.shape.length > 1) {
        const nd = v as NdArray;
        assert.deepEqual(nd.shape, want.shape, `${c.name}: output ${i} shape`);
        assertSame(nd.data, f64(want.values), `${c.name}: output ${i} values`);
      } else {
        assert.ok(v instanceof Float64Array, `${c.name}: output ${i} values`);
        assertSame(v as Float64Array, f64(want.values), `${c.name}: output ${i} values`);
      }
      if (want.index === null) assert.equal(k, null, `${c.name}: output ${i} index`);
      else assertSame(k!, f64(want.index), `${c.name}: output ${i} index`);
    });
  }
});

test("split: n larger than max(sources)+1 gives empty trailing outputs", async () => {
  await ready();
  const parts = split([1, 2, 3], [0, 1, 0], { n: 4 });
  assert.equal(parts.length, 4);
  assert.deepEqual(parts.map(({ values: v }) => Array.from(v as Float64Array)), [[1, 3], [2], [], []]);
  assert.deepEqual(parts.map(({ index: k }) => k), [null, null, null, null]);
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
  const [{ values: a }, { values: b }] = split([1, null, 3], [0, 1, 0]);
  assert.deepEqual(Array.from(a as Float64Array), [1, 3]);
  assert.ok(Number.isNaN((b as Float64Array)[0]));
  const [{ values: c }] = split({ data: new Float64Array([4, 5]), shape: [2] }, new Uint32Array([0, 0]));
  assert.deepEqual(Array.from(c as Float64Array), [4, 5]);
});

test("split: the outputs are copies, not views of the WASM heap", async () => {
  await ready();
  const [{ values: v, index: k }] = split([1, 2], [0, 0], { index: [10, 20] });
  assert.equal((v as Float64Array).buffer.byteLength, (v as Float64Array).byteLength);
  assert.equal(k!.buffer.byteLength, k!.byteLength);
});

test("split: values given as rows split along time and keep the row shape", async () => {
  await ready();
  const parts = split([[1, 2], [3, null], [5, 6]], [1, 0, 1], { index: [10, 20, 30] });
  assert.equal(parts.length, 2);
  const { values: a, index: ka } = parts[0], { values: b, index: kb } = parts[1];
  assert.deepEqual((a as NdArray).shape, [1, 2]);
  assert.deepEqual(Array.from((a as NdArray).data), [3, NaN]);
  assert.deepEqual((b as NdArray).shape, [2, 2]);
  assert.deepEqual(Array.from((b as NdArray).data), [1, 2, 5, 6]);
  assert.deepEqual(Array.from(ka!), [20]);
  assert.deepEqual(Array.from(kb!), [10, 30]);
  const [{ values: e }] = split({ data: new Float64Array(0), shape: [0, 4] }, [], { n: 1 });
  assert.deepEqual((e as NdArray).shape, [0, 4]);
  assert.throws(() => split({ data: new Float64Array(6), shape: [3, 2] }, [0, 1]),
    { name: "RangeError", message: "split: values/sources/index length differ" });
});

test("split: a wrong kind is a TypeError, a shape mismatch a RangeError", async () => {
  await ready();
  // values
  assert.throws(() => split(["a", 1] as any, [0, 1]), TypeError);
  assert.throws(() => split(5 as any, [0]), TypeError);
  // sources: a whole number (TypeError) from 0 to 2147483646 (RangeError)
  const kinds: Array<[unknown[], string]> = [
    [[0.5, 0], "0.5"], [[NaN, 0], "NaN"], [[Infinity, 0], "inf"], [[null, 0], "null"],
    [[true, false], "a bool"], [[1, false], "a bool"], [["0", 1], "a string"],
  ];
  for (const [bad, text] of kinds)
    assert.throws(() => split([1, 2], bad as any), {
      name: "TypeError", message: `split: a source must be a whole number, got ${text}`,
    }, JSON.stringify(bad));
  for (const [bad, text] of [[[-1, 0], "-1"], [[2 ** 31 - 1, 0], "2147483647"], [[2 ** 32, 0], "4294967296"]] as const)
    assert.throws(() => split([1, 2], bad as any), {
      name: "RangeError", message: `split: a source must be from 0 to 2147483646, got ${text}`,
    });
  assert.throws(() => split([1, 2], new Float64Array([0, -1])), RangeError);
  assert.throws(() => split([1, 2], [[0], [1]] as any), RangeError);
  assert.throws(() => split([1], [2 ** 31 - 2], { n: 1 }), {
    name: "RangeError", message: "split: n=1 is too small for sources up to 2147483646; events would be dropped",
  });
  // lengths
  assert.throws(() => split([1, 2], [0]), { name: "RangeError", message: "split: values/sources/index length differ" });
  assert.throws(() => split([1, 2], [0, 1], { index: [1] }), RangeError);
  // index: a wrong kind, and rows (a shape error, as a 2-D NdArray index is)
  assert.throws(() => split([1, 2], [0, 1], { index: ["a", "b"] as any }), TypeError);
  assert.throws(() => split([1, 2], [0, 1], { index: [[1], [2]] as any }), {
    name: "RangeError", message: "split: index must be 1-D (one index per event), got 2 dimensions",
  });
  assert.throws(() => split([1, 2], [0, 1], { index: { data: new Float64Array(2), shape: [2, 1] } as any }),
    RangeError);
  // options
  assert.throws(() => split([1], [0], { m: 1 } as any), { name: "TypeError", message: "split: unknown option 'm'" });
  assert.throws(() => split([1], [0], { n: "3" as any }), TypeError);
  assert.throws(() => split([1], [0], { n: 1.5 }), { name: "TypeError", message: "split: n must be an integer, got 1.5" });
  assert.throws(() => split([1], [0], { n: 2 ** 31 }), RangeError);
  assert.throws(() => split([1], [0], 3 as any), TypeError);
  assert.throws(() => split([1], [0], { n: true as any }),
    { name: "TypeError", message: "split: n must be an integer, got boolean" });
});
