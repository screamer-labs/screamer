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

// ---- G7: a wrong-kind element raises TypeError naming its type and position ----
{
  class M {}
  const a = api as any;
  const it = (xs: unknown[]) => xs[Symbol.iterator]();
  async function* agen(xs: unknown[]) { for (const x of xs) yield x; }
  const drain = async (g: AsyncIterable<unknown>) => { for await (const _ of g) void _; };
  const pipe = () => { const i = a.Input("a"); return a.Pipeline([i], [a.RollingMean(2)(i)]); };
  const cases: [string, () => unknown, string][] = [
    ["list element", () => a.RollingMean(2)([1, new M()]), "RollingMean: event 1 is a M, expected a number"],
    ["list string", () => a.RollingMean(2)([1, "x"]), "RollingMean: event 1 is a string"],
    ["1-in M-out list", () => a.RollingMinMax(2)([1, new M()]), "RollingMinMax: event 1 is a M"],
    ["two lists", () => a.Add()([1, new M()], [1, 2]), "Add: event 1 of input 0 is a M"],
    ["list of events", () => a.Add()([[1, new M()]]), "Add: item [1] of event 0 is a M"],
    ["later event", () => a.Add()([[1, 2], [new M(), 1]]), "Add: item [0] of event 1 is a M"],
    ["iterator item", () => [...a.RollingMean(2)(it([1, new M()]))], "RollingMean: event 1 is a M"],
    ["iterator first item", () => [...a.RollingMean(2)(it([new M()]))], "RollingMean: event 0 is a M"],
    ["two iterators", () => [...a.Add()(it([1, new M()]), it([1, 2]))], "Add: event 1 of input 0 is a M"],
    ["iterator of events", () => [...a.Add()(it([[1, 2], [1, new M()]]))], "Add: item [1] of event 1 is a M"],
    ["iterator after arrays", () => [...a.RollingMean(2)(it([[1, 2], new M()]))], "RollingMean: event 1 is a M"],
    ["row list", () => a.CrossRank()([[1, 2], [1, new M()]]), "CrossRank: item [1][1] is a M"],
    ["row iterator", () => [...a.CrossRank()(it([[1, 2], new M()]))], "CrossRank: event 1 is a M"],
    ["report list", () => a.PortfolioReport()([[1, 2, 3, 4], [1, 2, 3, new M()]]), "PortfolioReport: item [1][3] is a M"],
    ["report iterator", () => [...a.PortfolioReport()(it([[1, 2, 3, 4], [1, 2, 3, new M()]]))], "PortfolioReport: item [3] of event 1 is a M"],
    ["pipeline feed", () => pipe()([1, new M()]), "Pipeline: event 1 of input 0 is a M"],
    ["pipeline index", () => pipe()({ values: [1, 2], index: [0, new M()] }), "Pipeline: the index of event 1 of input 0 is a M"],
    ["pipeline live push", () => pipe().live().push("a", 0, new M()), "Pipeline: the value pushed to input 0 is a M"],
  ];
  for (const [name, call, text] of cases)
    test(`G7: a wrong-kind element is a TypeError (${name})`, async () => {
      await ready();
      assert.throws(call, (e: unknown) => e instanceof TypeError && (e as Error).message.includes(text));
    });

  test("G7: an async source yielding a wrong-kind item is a TypeError", async () => {
    await ready();
    await assert.rejects(drain(a.RollingMean(2)(agen([1, new M()]))),
      (e: unknown) => e instanceof TypeError && (e as Error).message.includes("RollingMean: event 1 is a M"));
    await assert.rejects(drain(a.CrossRank()(agen([[1, 2], new M()]))),
      (e: unknown) => e instanceof TypeError && (e as Error).message.includes("CrossRank: event 1 is a M"));
  });

  test("G7: an event of the wrong shape is still a RangeError", async () => {
    await ready();
    assert.throws(() => [...a.RollingMean(2)(it([1, [1, 2]]))], RangeError);
  });
}

// ---- G7 review round 1 ----
{
  const a = api as any;
  const it = (xs: unknown[]) => xs[Symbol.iterator]();
  async function* agen(xs: unknown[]) { for (const x of xs) yield x; }
  const drain = async (g: AsyncIterable<unknown>) => { for await (const _ of g) void _; };

  test("G7: self-containing and shared nested arrays return promptly", async () => {
    await ready();
    const l: any[] = [];
    l.push(l, l);
    let x: any = [1];
    for (let i = 0; i < 40; i++) x = [x, x];
    for (const call of [() => a.RollingMean(2)([1, l]), () => a.RollingMean(2)([1, x]),
      () => [...a.RollingMean(2)(it([1, l]))], () => a.Add()([[1, l]]), () => a.Add()([1, l], [1, 2])]) {
      const t = Date.now();
      assert.throws(call, (e: unknown) => e instanceof RangeError || e instanceof TypeError);
      assert.ok(Date.now() - t < 2000);
    }
  });

  const shape: [string, () => unknown, string][] = [
    ["list", () => a.RollingMean(2)([1, [1, 2]]), "RollingMean: event 1 has shape (2,), expected a number"],
    ["iterator", () => [...a.RollingMean(2)(it([1, [1, 2]]))], "RollingMean: event 1 has shape (2,)"],
    ["1-in M-out list", () => a.RollingMinMax(2)([1, [1, 2]]), "RollingMinMax: event 1 has shape (2,)"],
    ["list of events", () => a.Add()([[1, [1, 2]]]), "Add: item [1] of event 0 has shape (2,)"],
    ["iterator of events", () => [...a.Add()(it([[1, 2], [1, [1, 2]]]))], "Add: item [1] of event 1 has shape (2,)"],
    ["two lists", () => a.Add()([1, [1, 2]], [1, 2]), "Add: event 1 of input 0 has shape (2,)"],
    ["two iterators", () => [...a.Add()(it([1, [1, 2]]), it([1, 2]))], "Add: event 1 of input 0 has shape (2,)"],
    ["event length", () => [...a.Add()(it([[1, 2], [1, 2, 3]]))], "Add: event 1 has shape (3,), expected 2 numbers"],
  ];
  for (const [name, call, text] of shape)
    test(`G7: an element of the wrong shape is a RangeError (${name})`, async () => {
      await ready();
      assert.throws(call, (e: unknown) => e instanceof RangeError && (e as Error).message.includes(text));
    });

  test("G7: an async element of the wrong shape is a RangeError", async () => {
    await ready();
    await assert.rejects(drain(a.RollingMean(2)(agen([1, [1, 2]]))),
      (e: unknown) => e instanceof RangeError && (e as Error).message.includes("RollingMean: event 1 has shape (2,)"));
  });

  test("G7: a non-data input names the op and its type", async () => {
    await ready();
    assert.throws(() => a.RollingMean(2)({}), (e: unknown) => e instanceof TypeError &&
      (e as Error).message === "RollingMean: a Object is not a valid input; pass numbers, arrays, lists or iterators");
  });

  test("G7: Pipeline feeds and live pushes keep the primitive coercions", async () => {
    await ready();
    const pipe = () => { const i = a.Input("a"); return a.Pipeline([i], [a.RollingMean(1)(i)]); };
    const out = pipe()([1, "3", true, undefined, null, new Date(5)] as any);
    assert.deepEqual(Array.from(out.values as Float64Array), [1, 3, 1, NaN, 0, 5]);
    const idx = pipe()({ values: [1, 2, 3], index: [0, "1", true] } as any);
    assert.deepEqual(Array.from(idx.index), [0, 1, 1]);
    for (const [v, want] of [["3", 3], [true, 1], [null, 0], [undefined, NaN], [new Date(5), 5]] as [unknown, number][]) {
      const lv = pipe().live();
      lv.push("a", 0, v as any);
      lv.flush();
      assert.deepEqual(Array.from(lv.result().values as Float64Array), [want]);
    }
    const w = (() => { const i = a.Input("a"); return a.Pipeline([i], [i]); })().live();
    assert.throws(() => w.pushWide("a", 0, [1, {}] as any), (e: unknown) => e instanceof TypeError &&
      (e as Error).message.includes("Pipeline: item [1] of the values pushed to input 0 is a Object"));
    assert.throws(() => pipe()([1, [2]] as any), TypeError);
    assert.throws(() => pipe()([1n] as any), TypeError);
  });
}
