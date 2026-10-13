// A wrong number of inputs is a RangeError naming the operator, with the text
// of the Python ValueError (rule A4). tests/test_input_kinds.py checks the
// same calls in Python.
import { before, test } from "node:test";
import assert from "node:assert/strict";
import * as api from "../src/index.ts";

const a = api as any;
const x = Float64Array.from([1, 2, 3]);
const range = (message: string) => ({ name: "RangeError", message });

before(async () => { await api.ready(); });

test("an N-input operator names itself and the count", () => {
  const corr = a.RollingCorr(3);
  assert.throws(() => corr(x), range("RollingCorr expects 2 inputs, got 1"));
  assert.throws(() => corr(x, x, x), range("RollingCorr expects 2 inputs, got 3"));
  assert.throws(() => corr(), range("RollingCorr expects 2 inputs, got 0"));
  assert.throws(() => corr(1.0), range("RollingCorr expects 2 inputs, got 1"));
  corr.dispose();
});

test("a one-input operator takes exactly one input", () => {
  const mean = a.RollingMean(3);
  assert.throws(() => mean(), range("RollingMean expects 1 input, got 0"));
  assert.throws(() => mean(1.0, 2.0), range("RollingMean expects 1 input, got 2"));
  mean.dispose();
});

test("a stream count names the operator", () => {
  const add = a.Add();
  const it = () => [1.0][Symbol.iterator]();
  assert.throws(() => add(it(), it(), it()),
    range("Add expects 2 inputs, or one stream of events holding every field, got 3"));
  add.dispose();
});

test("an empty list of events gives an empty list", () => {
  const add = a.Add();
  assert.deepEqual(add([]), []);
  add.dispose();
  const mean = a.RollingMean(2);
  assert.deepEqual(mean([]), []);
  mean.dispose();
});

test("PortfolioReport counts and four numbers", () => {
  const p = a.PortfolioReport();
  assert.throws(() => p(), range("PortfolioReport expects 4 inputs, or the engine output and optional prices, got 0"));
  assert.throws(() => p(1, 2, 3), range("PortfolioReport expects 4 inputs, or the engine output and optional prices, got 3"));
  const r = Array.from(p(1, 2, 3, 4) as ArrayLike<number>);
  assert.deepEqual(r.slice(0, 5), [0, 4, 3, 1, 0]);
  assert.ok(Number.isNaN(r[5]));
  p.dispose();
});

test("a NaN float index is a RangeError in merge and combineLatest", () => {
  const nan = /the index of input 0 is NaN at event 1; an index must be a number/;
  assert.throws(() => a.merge([{ values: [1, 2, 3], index: [0, NaN, 1] }]), (e: any) => e instanceof RangeError && nan.test(e.message));
  assert.throws(() => a.combineLatest([{ values: [1, 2, 3], index: [0, NaN, 1] }]), (e: any) => e instanceof RangeError && nan.test(e.message));
  const it = () => [{ value: 1, index: 0 }, { value: 2, index: NaN }][Symbol.iterator]();
  assert.throws(() => [...a.merge([it()])], (e: any) => e instanceof RangeError && nan.test(e.message));
});
