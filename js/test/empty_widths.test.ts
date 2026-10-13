import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ready, Input, Pipeline, select, resample, combineLatest, dropna,
  RollingMean, BollingerBands, type Output,
} from "../src/index.ts";

// G18: an empty result keeps the output's width when it is known, from the
// graph or from the input, in batch and in live drains; only a width-1
// output is (0,). Mirrors tests/test_empty_widths.py.

const shape = (o: Output): number[] => (o.values instanceof Float64Array ? [o.values.length] : (o.values as any).shape);
const empty = (w: number) => ({ data: new Float64Array(0), shape: [0, w] });

test("G18: an empty wide batch keeps its width", async () => {
  await ready();
  const a = Input("a"), b = Input("b");
  const cases: [Pipeline, any, number[]][] = [
    [new Pipeline([a], [a]), empty(3), [0, 3]],
    [new Pipeline([a], [a]), new Float64Array(0), [0]],
    [new Pipeline([a], [a]), empty(0), [0, 0]],
    [new Pipeline([a], [dropna(a)]), empty(3), [0, 3]],
    [new Pipeline([a], [resample(a, { count: 2, agg: "ohlc" })]), empty(3), [0, 12]],
    [new Pipeline([a], [resample(a, { count: 2, agg: "ohlc" })]), new Float64Array(0), [0, 4]],
  ];
  for (const [p, feed, want] of cases) {
    try { assert.deepEqual(shape(p(feed) as Output), want); } finally { p.dispose(); }
  }
  const p = new Pipeline([a, b], [combineLatest([a, b])]);
  try { assert.deepEqual(shape(p({ a: empty(3), b: new Float64Array(0) }) as Output), [0, 4]); } finally { p.dispose(); }
  for (const alignOutputs of [true, false]) {
    const q = new Pipeline([a, b], [a, RollingMean(2)(b)], { alignOutputs });
    try {
      const out = q({ a: empty(3), b: new Float64Array(0) }) as Output[];
      assert.deepEqual(out.map(shape), [[0, 3], [0]]);
    } finally { q.dispose(); }
  }
});

test("G18: graph widths are known before any event", async () => {
  await ready();
  const a = Input("a");
  for (const [make, w] of [[() => BollingerBands(3)(a), 3], [() => select(a, [0, 2]), 2]] as [() => any, number][]) {
    const p = new Pipeline([a], [make()]);
    try { assert.deepEqual(shape(p.live().result() as Output), [0, w]); } finally { p.dispose(); }
  }
  const p = new Pipeline([a], [a]);
  try { assert.deepEqual(shape(p.live().result() as Output), [0]); } finally { p.dispose(); }
});

for (const alignOutputs of [true, false]) {
  test(`G18: an empty live drain keeps the learned width (alignOutputs ${alignOutputs})`, async () => {
    await ready();
    const a = Input("a"), b = Input("b");
    const p = new Pipeline([a, b], [a, RollingMean(2)(b)], { alignOutputs });
    try {
      const s = p.live();
      s.push("a", 0, [1, 2, 3]); s.push("b", 0, 1);
      s.push("a", 1, [1, 2, 3]); s.push("b", 1, 2);
      s.result();
      assert.deepEqual((s.result() as Output[]).map(shape), [[0, 3], [0]]);
    } finally { p.dispose(); }
  });
}

test("G18: a known width reaches the aligned join before any row", async () => {
  await ready();
  const a = Input("a"), b = Input("b");
  const p = new Pipeline([a, b], [RollingMean(1)(a), resample(b, { count: 2 })]);
  try {
    const s = p.live();
    s.push("a", 0, 1);
    s.advance(5);
    assert.deepEqual(shape((s.result() as Output[])[1]), [0]);
    s.push("b", 6, [5, 6]);
    assert.deepEqual(shape((s.result() as Output[])[1]), [0, 2]);
    s.push("b", 7, [7, 8]);
    s.flush();
    const out = (s.result() as Output[])[1];
    assert.deepEqual(Array.from((out.values as any).data), [7, 8]);
    assert.deepEqual(Array.from(out.index), [6]);
  } finally { p.dispose(); }
});
