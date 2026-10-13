import { test } from "node:test";
import assert from "node:assert/strict";
import { ready, Input, Pipeline, resample, type Output } from "../src/index.ts";

// G23: fill "carry" is per column. Each NaN cell of a bar takes its column's
// last non-NaN value, and an empty bar repeats those values, under every= and
// clock=, in batch and live. Mirrors tests/test_resample_fill.py.

const index = [0, 10, 25, 40, 41, 70];
const rows = [[1, 10], [NaN, 20], [3, NaN], [NaN, NaN], [NaN, 50], [6, NaN]];
const values = () => ({ data: Float64Array.from(rows.flat()), shape: [rows.length, 2] });
const flat = (o: Output) => Array.from(((o.values as any).data ?? o.values) as Float64Array);

test("G23: carry fills each NaN cell from its own column (every=)", async () => {
  await ready();
  const x = Input("x");
  const p = new Pipeline([x], [resample(x, { every: 10, fill: "carry" })]);
  try {
    const want = [[1, 10], [1, 20], [3, 20], [3, 20], [3, 50], [3, 50], [3, 50], [6, 50]];
    const out = p({ values: values(), index }) as Output;
    assert.deepEqual(Array.from(out.index), [0, 10, 20, 30, 40, 50, 60, 70]);
    assert.deepEqual(flat(out), want.flat());
    const s = p.live();
    rows.forEach((r, t) => s.push("x", index[t], r));
    s.flush();
    assert.deepEqual(flat(s.result() as Output), want.flat());
  } finally { p.dispose(); }
});

test("G23: carry fills each NaN cell from its own column (clock=)", async () => {
  await ready();
  const x = Input("x"), c = Input("c");
  const p = new Pipeline([x, c], [resample(x, { clock: true, driver: c, agg: "last", fill: "carry" })]);
  try {
    const out = p({ x: { values: values(), index }, c: { values: [0, 0, 0, 0, 0], index: [5, 15, 30, 45, 75] } }) as Output;
    assert.deepEqual(Array.from(out.index), [5, 15, 30, 45, 75]);
    assert.deepEqual(flat(out), [1, 10, 1, 20, 3, 20, 3, 50, 6, 50]);
  } finally { p.dispose(); }
});
