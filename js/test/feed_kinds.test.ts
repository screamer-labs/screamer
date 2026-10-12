import { test } from "node:test";
import assert from "node:assert/strict";
import { ready, Input, Pipeline, RollingMean, resample, delay, select, combineLatest } from "../src/index.ts";

// Pipeline feed kinds (Task 18b: F3, G13). A typed array is a values feed in
// every position; a feed of the wrong kind names the input; a string that does
// not read as a number is a TypeError, as in Python.

function pipe() {
  const a = Input("a");
  return new Pipeline([a], [RollingMean(2)(a)]);
}

function vals(out: any): number[] {
  return Array.from(out.values as Float64Array);
}

test("feeds: a typed array is a values feed bare, in {values} and in a record", async () => {
  await ready();
  const p = pipe();
  const expected = vals(p(new Float64Array([1, 2, 4])));
  for (const x of [new Int32Array([1, 2, 4]), new Uint8Array([1, 2, 4]), new Float32Array([1, 2, 4])]) {
    assert.deepEqual(vals(p(x as any)), expected);
    assert.deepEqual(vals(p({ values: x } as any)), expected);
    assert.deepEqual(vals(p({ a: x } as any)), expected);
  }
});

test("feeds: a wrong-kind feed raises TypeError naming the input", async () => {
  await ready();
  const p = pipe();
  assert.throws(() => p({ a: 5 } as any), (e: unknown) =>
    e instanceof TypeError && (e as Error).message.startsWith("Pipeline: the feed for input 'a' is a number, expected"));
  assert.throws(() => p({ a: undefined } as any), (e: unknown) =>
    e instanceof TypeError && (e as Error).message.includes("is an undefined"));
  assert.throws(() => p({ a: { foo: 1 } } as any), (e: unknown) =>
    e instanceof TypeError && (e as Error).message.includes("the feed for input 'a' is an Object"));
  assert.throws(() => p(new Map() as any), (e: unknown) =>
    e instanceof TypeError && (e as Error).message.startsWith("Pipeline: the feeds are a Map"));
  for (const f of [new BigInt64Array([1n, 2n]), { values: new BigUint64Array([1n]) }, { a: new BigInt64Array([1n]) }]) {
    assert.throws(() => p(f as any), (e: unknown) =>
      e instanceof TypeError && /are a Big(Ui|I)nt64Array; values must be numbers/.test((e as Error).message));
  }
  const live = p.live();
  assert.throws(() => live.push("a", 0, 5n as any), (e: unknown) =>
    e instanceof TypeError && (e as Error).message.includes("is a bigint"));
});

test("feeds: any string is a TypeError; null is NaN", async () => {
  await ready();
  const p = pipe();
  for (const bad of ["abc", "", " ", "NaN", "3", "Infinity"]) {
    const isBad = (e: unknown) => e instanceof TypeError && (e as Error).message.includes("is a string, expected a number");
    assert.throws(() => p({ values: [1, bad] } as any), isBad);
    assert.throws(() => p.live().push("a", 0, bad as any), isBad);
    assert.throws(() => p.live().push("a", 0, [bad] as any), isBad);
    assert.throws(() => p.live().pushWide("a", 0, [1, bad] as any), isBad);
    assert.throws(() => RollingMean(2)(bad as any), TypeError);
    assert.throws(() => RollingMean(2)([1, bad] as any), TypeError);
  }
  // null is NaN (A9), a boolean reads as before.
  assert.ok(Number.isNaN(vals(p({ values: [1, null] } as any))[1]));
  assert.deepEqual(vals(p({ values: [1, true] } as any)).slice(1), [1]);
});

test("feeds: a wrong-kind bare feed throws TypeError naming its kind", async () => {
  await ready();
  const p = pipe();
  for (const f of [new DataView(new ArrayBuffer(8)), new Set([1]), (function* () { yield 1; })()])
    assert.throws(() => p(f as any), (e: unknown) => e instanceof TypeError && (e as Error).message.startsWith("Pipeline: the feeds are"));
  assert.throws(() => p({ values() { return 1; } } as any), (e: unknown) => e instanceof TypeError && (e as Error).message.includes("values function"));
});

test("settings: a string setting is a TypeError", async () => {
  await ready();
  const x = Input("x");
  assert.throws(() => resample(x, { threshold: "3" as any }), TypeError);
  assert.throws(() => resample(x, { every: "2" as any }), TypeError);
  assert.throws(() => delay(x, "1" as any), TypeError);
  assert.throws(() => select(x, "0" as any), TypeError);
});

test("R-b: null is NaN in operator iterators, async iterators and row ops", async () => {
  await ready();
  const ref = Array.from(RollingMean(2)(Float64Array.from([1, NaN, 3])) as Float64Array, String);
  async function* g() { yield 1; yield null; yield 3; }
  const got: number[] = [];
  for await (const x of RollingMean(2)(g() as any) as AsyncIterable<number>) got.push(x);
  assert.deepEqual(got.map(String), ref);
  assert.deepEqual([...(RollingMean(2)([1, null, 3][Symbol.iterator]() as any) as Iterable<number>)].map(String), ref);
  const cr = (await import("../src/index.ts") as any).CrossRank();
  assert.deepEqual(Array.from(cr([1, null, 3]) as number[], String), ["-1", "NaN", "1"]);
});

test("round 2: String objects, operator number settings and maxPending refuse strings", async () => {
  await ready();
  const p = pipe();
  assert.throws(() => p({ values: [1, new String("3")] } as any), TypeError);
  assert.throws(() => p.live().push("a", 0, new String("3") as any), TypeError);
  assert.throws(() => p.live().push("a", 0, [new String("3")] as any), TypeError);
  for (const bad of ["3", new String("3")]) {
    assert.throws(() => RollingMean(bad as any), (e: unknown) => e instanceof TypeError && (e as Error).message === "windowSize must be a number, got a string");
  }
  const x = Input("x"), y = Input("y");
  assert.throws(() => combineLatest([x, y], { maxPending: "3" as any }), TypeError);
  assert.equal(typeof RollingMean(3), "function");
});

test("round 3: every number[] operator setting refuses strings, found from the manifest", async () => {
  await ready();
  const { readFileSync } = await import("node:fs");
  const api: any = await import("../src/index.ts");
  const manifest = JSON.parse(readFileSync(new URL("../../devtools/wasm/wasm_manifest.json", import.meta.url), "utf8"));
  const camel = (s: string) => s.replace(/_([a-z])/g, (_m, c) => c.toUpperCase());
  let checked = 0;
  for (const op of manifest) {
    op.ctor.forEach((type: string, i: number) => {
      if (!type.includes("vector")) return;
      const name = camel(op.ctor_args[i].name);
      for (const bad of ["3", new String("3"), ["3"], [1, new String("3")]]) {
        const args: unknown[] = new Array(i + 1).fill(undefined);
        args[i] = bad;
        assert.throws(() => api[op.name](...args), (e: unknown) => e instanceof TypeError &&
          (e as Error).message === `${name} must be a number or an array of numbers, got a string`, `${op.name}(${name})`);
      }
      // Q1 (Task 18c): only a number, an Array or a typed array is a vector
      // setting, and every element is a number, as in Python.
      const otherKinds: unknown[] = [new Set([1, 2]), (function* () { yield 1; yield 2; })(),
                                     [[1], [2]], [1, {}], [1, true], new Map(), {}];
      for (const bad of otherKinds) {
        const args: unknown[] = new Array(i + 1).fill(undefined);
        args[i] = bad;
        assert.throws(() => api[op.name](...args), (e: unknown) => e instanceof TypeError &&
          (e as Error).message.startsWith(`${name} must be a number or an array of numbers, got `), `${op.name}(${name}) ${String(bad)}`);
      }
      for (const good of [[1, 2], Float64Array.from([1, 2]), 1]) {
        const ok: unknown[] = new Array(i + 1).fill(undefined);
        ok[i] = good;
        assert.equal(typeof api[op.name](...ok), "function", `${op.name}(${name}) ${String(good)}`);
      }
      checked++;
    });
  }
  assert.equal(checked, 11);
});
