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

// ---- G2 / A4: one checked NdArray entry, RangeError for mismatches ----
{
  const bad = { data: new Float64Array(4), shape: [100, 50] };
  const kinds: [string, () => unknown][] = [
    ["elementwise", () => (api as any).RollingMean(3)(bad)],
    ["multi-input", () => (api as any).RollingCorr(3)(bad, bad)],
    ["row op", () => (api as any).CrossRank()(bad)],
    ["reducer", () => (api as any).PortfolioReport()({ data: new Float64Array(8), shape: [10, 5, 4] })],
    ["row stream", () => [...((api as any).CrossRank()([bad][Symbol.iterator]()) as Iterable<unknown>)]],
  ];
  for (const [name, call] of kinds)
    test(`G2: malformed NdArray raises RangeError on the ${name} path`, async () => {
      await ready();
      assert.throws(call, RangeError);
    });

  test("G2: a non-integer shape is a TypeError", async () => {
    await ready();
    assert.throws(() => (api as any).RollingMean(3)({ data: new Float64Array(4), shape: [2.5] }), TypeError);
  });

  test("A4: inputs of different lengths raise RangeError", async () => {
    await ready();
    assert.throws(() => (api as any).Add()(new Float64Array(5), new Float64Array(4)), RangeError);
  });

  test("A4: a wrong kind is still a TypeError", async () => {
    await ready();
    assert.throws(() => (api as any).RollingMean(3)("abc" as any), TypeError);
  });
}

test("G2: a malformed NdArray raises RangeError in positionTest (portfolio target) and walkForward targets", async () => {
  await ready();
  const bad = { data: new Float64Array(4), shape: [100, 50] };
  const px = new Float64Array(100).fill(1);
  const eng = (api as any).BacktestPortfolioTarget(0.001);
  assert.throws(() => (api as any).positionTest(eng, bad, [bad], { n: 2 }), RangeError);
  const single = (api as any).BacktestOHLCTarget(1e-4);
  assert.throws(() => (api as any).walkForward(single, bad, [px, px, px, px], { train: 10, test: 5 }), RangeError);
  assert.throws(() => (api as any).probabilisticSharpe({ data: new Float64Array(4), shape: [3, 4] }), RangeError);
});

test("C4: a list of arrays into Add names the event reading", async () => {
  await ready();
  const add = (api as any).Add();
  assert.throws(() => add([new Float64Array(30), new Float64Array(30)]),
    (e: unknown) => e instanceof RangeError && /^Add expects 2 inputs; a list is read as a sequence of events, and event 0 has 30 values \(expected 2\)\. To pass 2 separate inputs, call Add\(\)\(a, b\)\.$/.test((e as Error).message));
  assert.throws(() => add([1, 2]),
    (e: unknown) => e instanceof RangeError && /event 0 has 1 value \(expected 2\)/.test((e as Error).message));
});

test("C4: a list of N-element lists and of (K, N) NdArray events reads as events", async () => {
  await ready();
  const add = (api as any).Add();
  assert.deepEqual(Array.from(add([[1, 2], [3, 4]]) as Float64Array), [3, 7]);
  const ev = () => ({ shape: [3, 2], data: Float64Array.from([1, 2, 3, 4, 5, 6]) });
  const viaList = add([ev(), ev()]) as NdArray;
  const viaBatch = add({ shape: [2, 3, 2], data: Float64Array.from([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]) }) as NdArray;
  assert.deepEqual(viaList.shape, viaBatch.shape);
  assert.deepEqual(Array.from(viaList.data), Array.from(viaBatch.data));
  assert.throws(() => add([[1, 2], [3, 4, 5]]), (e: unknown) => e instanceof RangeError && /event 1 has 3 values/.test((e as Error).message));
});
