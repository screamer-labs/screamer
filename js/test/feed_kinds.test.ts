import { test } from "node:test";
import assert from "node:assert/strict";
import { ready, Input, Pipeline, RollingMean } from "../src/index.ts";

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

test("feeds: a non-numeric string is a TypeError; a numeric string reads as before", async () => {
  await ready();
  const p = pipe();
  for (const bad of ["abc", "", " ", "NaN"]) {
    const isBad = (e: unknown) => e instanceof TypeError && (e as Error).message.includes("which is not a number");
    assert.throws(() => p({ values: [1, bad] } as any), isBad);
    assert.throws(() => p.live().push("a", 0, bad as any), isBad);
    assert.throws(() => p.live().push("a", 0, [bad] as any), isBad);
    assert.throws(() => p.live().pushWide("a", 0, [1, bad] as any), isBad);
  }
  assert.deepEqual(vals(p({ values: [1, "3"] } as any)).slice(1), [2]);
  const live = p.live();
  live.push("a", 0, 1);
  live.push("a", 1, "3" as any);
  assert.deepEqual(vals(live.result()).slice(1), [2]);
  // null is NaN (A9), a boolean reads as before.
  assert.ok(Number.isNaN(vals(p({ values: [1, null] } as any))[1]));
  assert.deepEqual(vals(p({ values: [1, true] } as any)).slice(1), [1]);
});
