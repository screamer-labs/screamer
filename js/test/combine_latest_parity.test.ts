import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { ready, combineLatest } from "../src/index.ts";

// combineLatest against Python's combine_latest
// (devtools/wasm/gen_combine_oracle.py writes the fixture). Both call the one
// C++ join, so every value matches exactly. Python's (values, index) and
// (value, index) tuples are {values, index} and {value, index} objects here; a
// ValueError is a RangeError, and the messages name combineLatest.
const HERE = dirname(fileURLToPath(import.meta.url));
type Num = number | null;
type Result =
  | { error: string; message: string }
  | { values: Num[]; shape: number[]; index: Num[] | null }
  | { rows: Array<{ value: Num[]; shape: number[]; index: number | null }> };
interface Case {
  name: string;
  inputs: Array<{ values: Num[]; shape: number[] }>;
  index: number[][];
  layout: string | Array<string | null> | null;
  emit: "when_all" | "on_any";
  batch: Result; batch_indexed: Result; iter: Result; iter_indexed: Result;
}
const cases: Case[] = JSON.parse(
  readFileSync(join(HERE, "..", "..", "wasm", "smoke", "oracle_combine_latest.json"), "utf8"));

const f64 = (xs: Num[]) => Float64Array.from(xs, (v) => (v === null ? NaN : v));
const same = (a: number, b: Num) => (b === null ? Number.isNaN(a) : Object.is(a, b) || a === b);
const flat = (v: any): number[] =>
  typeof v === "number" ? [v] : Array.from(v instanceof Float64Array || Array.isArray(v) ? v : v.data);

function nd(x: { values: Num[]; shape: number[] }) { return { data: f64(x.values), shape: x.shape }; }

// Event t of an input: a number for a 1-D input, else an NdArray of the row.
function event(x: { values: Num[]; shape: number[] }, t: number) {
  const per = x.shape.slice(1).reduce((a, b) => a * b, 1);
  const data = f64(x.values.slice(t * per, (t + 1) * per));
  return x.shape.length === 1 ? data[0] : { data, shape: x.shape.slice(1) };
}

function run(f: () => any): any {
  try {
    const r = f();
    return r && typeof r.next === "function" ? { rows: [...r] } : r;
  } catch (e) {
    return { thrown: e };
  }
}

function check(name: string, want: Result, got: any) {
  if ("error" in want) {
    assert.ok(got.thrown, `${name}: expected ${want.error}, got a result`);
    const cls = want.error === "ValueError" ? RangeError : TypeError;
    assert.ok(got.thrown instanceof cls, `${name}: ${got.thrown} is not a ${cls.name}`);
    assert.equal(got.thrown.message, want.message.replace(/^combine_latest:/, "combineLatest:"), name);
    return;
  }
  assert.ok(!got.thrown, `${name}: ${got.thrown}`);
  if ("rows" in want) {
    assert.equal(got.rows.length, want.rows.length, `${name}: rows`);
    want.rows.forEach((w, r) => {
      const g = got.rows[r];
      const v = flat(g.value);
      assert.ok(v.length === w.value.length && v.every((x, j) => same(x, w.value[j])), `${name}: row ${r}`);
      if (w.shape.length > 1) assert.deepEqual(g.value.shape, w.shape, `${name}: row ${r} shape`);
      assert.equal(g.index, w.index, `${name}: row ${r} index`);
    });
    return;
  }
  assert.deepEqual(got.values.shape, want.shape, `${name}: shape`);
  assert.ok(got.values.data.length === want.values.length &&
    Array.from(got.values.data as Float64Array).every((x, j) => same(x, want.values[j])), `${name}: values`);
  if (want.index === null) assert.equal(got.index, null, `${name}: index`);
  else assert.ok(got.index.length === want.index.length &&
    Array.from(got.index as Float64Array).every((x, j) => same(x, want.index![j])), `${name}: index`);
}

test("combineLatest parity: the fixture covers two-value rows, layouts and errors", () => {
  const names = cases.map((c) => c.name);
  for (const want of ["two-value rows", "layout AT, T", "layout AT, AT", "layout TFA, T", "empty input",
    "lengths differ", "layouts do not broadcast", "layout count"])
    assert.ok(names.includes(want), want);
  assert.ok(cases.length > 80);
});

test("combineLatest parity: every fixture case matches Python exactly", async () => {
  await ready();
  for (const c of cases) {
    const opts: any = { emit: c.emit };
    if (c.layout !== null) opts.layout = c.layout;
    const ins = c.inputs.map(nd);
    const cl = (inputs: any[]) => combineLatest(inputs, opts) as any;
    check(`${c.name} batch`, c.batch, run(() => cl(ins)));
    check(`${c.name} batch indexed`, c.batch_indexed,
      run(() => cl(ins.map((x, k) => ({ values: x, index: c.index[k] })))));
    check(`${c.name} iter`, c.iter, run(() => cl(c.inputs.map((x) =>
      (function* () { for (let t = 0; t < x.shape[0]; t++) yield event(x, t); })()))));
    check(`${c.name} iter indexed`, c.iter_indexed, run(() => cl(c.inputs.map((x, k) =>
      (function* () { for (let t = 0; t < x.shape[0]; t++) yield { value: event(x, t), index: c.index[k][t] }; })()))));
  }
});
