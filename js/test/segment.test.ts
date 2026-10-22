// Segment(op) in JavaScript: the same operator as Python's (the numbers are
// checked against Python in research.test.ts); here the call forms, the copy
// it holds, its lifetime and Pipeline.toJSON.
import { test } from "node:test";
import assert from "node:assert/strict";
import { ready, Segment, CumSum, RollingStd, BollingerBands, RollingCorr, Input, Pipeline } from "../src/index.ts";

await ready();

const label = Float64Array.from([NaN, 1, 1, NaN, 2, 2, 1, 1]);
const x = Float64Array.from([1, 2, 3, 4, 5, 6, 7, 8]);

test("Segment: resets op where the label changes; a NaN label keeps the segment", () => {
  const out = Segment(CumSum())(label, x) as Float64Array;
  assert.deepEqual(Array.from(out), [NaN, 2, 5, 9, 5, 11, 7, 15]);
});

test("Segment: one event at a time equals the batch, multi-output included", () => {
  for (const make of [() => Segment(RollingStd(2)), () => Segment(BollingerBands(2))]) {
    const batch = make()(label, x) as any;
    const op = make();
    const events = Array.from(label, (l, t) => op(l, x[t]));
    const flat = events.flatMap((e: any) => (typeof e === "number" ? [e] : e));
    assert.deepEqual(flat, Array.from(ArrayBuffer.isView(batch) ? batch : batch.data));
    op.dispose();
  }
});

test("Segment: keeps the arity of op", () => {
  const corr = Segment(RollingCorr(3));
  assert.equal(corr.numInputs, 3);
  assert.equal(corr.numOutputs, 1);
  const bands = Segment({ op: BollingerBands(3) });
  assert.equal(bands.numInputs, 2);
  assert.equal(bands.numOutputs, 3);
  corr.dispose();
  bands.dispose();
});

test("Segment: holds its own copy of op", () => {
  const inner = CumSum();
  const seg = Segment(inner);
  for (let i = 0; i < 5; i++) seg(1, 1);
  assert.equal(inner(1), 1);                 // the caller's op never ran
  assert.equal(Segment(inner)(1, 1), 1);     // nor is shared between wrappers
  assert.equal(seg(1, 1), 6);
  inner.dispose();                           // the wrapper's copy outlives it
  assert.equal(seg(1, 1), 7);
  seg.reset();
  assert.equal(seg(1, 1), 1);
  seg.dispose();
});

test("Segment: streams and Pipelines", async () => {
  const seg = Segment(CumSum());
  const streamed = Array.from(seg((function* () { for (let t = 0; t < 8; t++) yield [label[t], x[t]]; })()) as any);
  assert.deepEqual(streamed, [NaN, 2, 5, 9, 5, 11, 7, 15]);
  const row = Input("row");
  const p = new Pipeline([row], [Segment(RollingStd(2))(row)]);
  const again = Pipeline.fromJSON(p.toJSON());
  assert.deepEqual(again.toJSON(), p.toJSON());
  assert.deepEqual(p.toJSON().nodes[1].params, { op: { functor: "RollingStd", params: { window_size: 2 } } });
});

test("Segment: refuses what is not an operator", () => {
  assert.throws(() => Segment(3 as any), /Segment: op must be an operator from the screamer namespace, got number/);
  assert.throws(() => (Segment as any)(), /Segment: op is required/);
});
