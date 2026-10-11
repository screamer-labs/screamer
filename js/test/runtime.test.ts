import { test } from "node:test";
import assert from "node:assert/strict";
import { init, ready } from "../src/index.ts";
import * as ops from "../src/generated/ops.ts";
import { wrapOp } from "../src/runtime.ts";
import type { NdArray } from "../src/ndarray.ts";

// NaN-aware equality for a sequence of numbers.
function eqSeq(actual: ArrayLike<number>, expected: ArrayLike<number>) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < expected.length; i++) {
    const a = actual[i], e = expected[i];
    if (Number.isNaN(e)) assert.ok(Number.isNaN(a), `index ${i}: expected NaN, got ${a}`);
    else assert.equal(a, e, `index ${i}`);
  }
}

test("wrapOp: RollingMean(3, strict) across all regimes", async () => {
  const M = await init();
  // A factory, not a raw op: every stream runs on copies of its own (D8).
  const op = wrapOp(M, () => new M.RollingMean(3, "strict"));

  // N numbers -> one streaming event -> number
  const s = op(1);
  assert.equal(typeof s, "number");

  // single Float64Array -> Float64Array batch
  const fa = op(new Float64Array([1, 2, 3, 4, 5]));
  assert.ok(fa instanceof Float64Array);
  eqSeq(fa, [NaN, NaN, 2, 3, 4]);

  // single number[] -> number[] batch, container preserved
  const na = op([1, 2, 3]);
  assert.ok(Array.isArray(na));
  eqSeq(na, [NaN, NaN, 2]);

  // sync iterable -> generator streaming the same values
  const streamed = [...op([1, 2, 3, 4, 5][Symbol.iterator]())];
  eqSeq(streamed, [NaN, NaN, 2, 3, 4]);

  op.dispose();
});

test("wrapOp: RollingMinMax(3) multi-out batch returns NdArray", async () => {
  const M = await init();
  const op = wrapOp(M, new M.RollingMinMax(3));

  const res = op(new Float64Array([1, 2, 3, 4, 5])) as NdArray;
  assert.ok(res && res.data instanceof Float64Array);
  assert.deepEqual(res.shape, [5, 2]);
  assert.equal(res.data.length, 10);

  op.dispose();
});

// A dropped stream must give its op copies back once the stream is garbage
// collected, without dispose(). The factory counts its calls: if the dropped
// stream's copy returns to the pool, the next stream reuses it.
import { setFlagsFromString } from "node:v8";
import { runInNewContext } from "node:vm";
import { wrapReducerOp, wrapRowOp } from "../src/runtime.ts";
setFlagsFromString("--expose-gc");
const forceGc: () => void = runInNewContext("gc");

async function collect() {
  for (let i = 0; i < 5; i++) {
    forceGc();
    await new Promise((r) => setTimeout(r, 0));
  }
}

for (const [label, wrap, mk, ev] of [
  ["wrapOp", wrapOp, (M: any) => new M.RollingMean(3, "strict"), () => 1],
  ["wrapReducerOp", wrapReducerOp, (M: any) => new M.PortfolioReport(), (w: number) => new Array(2 * w).fill(1)],
  ["wrapRowOp", wrapRowOp, (M: any) => new M.CrossRank("signed"), () => [1, 2, 3]],
] as const) {
  for (const kind of ["sync", "async"] as const) {
    test(`${label}: a dropped ${kind} stream returns its copies when collected`, async () => {
      const M = await init();
      let made = 0;
      const op = (wrap as any)(M, () => { made++; return (mk as any)(M); });
      const width = op.numInputs;
      const ev0 = (ev as any)(width);
      const src = () => kind === "sync"
        ? (function* () { while (true) yield ev0; })()
        : (async function* () { while (true) yield ev0; })();
      const start = async () => {
        const s = op(src());
        await s.next();               // opens the stream: takes a copy
      };
      await start();
      const before = made;
      await collect();
      await start();
      assert.equal(made, before, "the second stream reused the dropped stream's copy");
      // An explicitly closed stream still releases on return().
      const s = op(src());
      await s.next();
      await s.return(undefined);
      const after = made;
      await op(src()).next();
      assert.equal(made, after);
      op.dispose();
    });
  }
}

test("copyPool: a new stream copy is not reset, a reused one is", async () => {
  const M = await init();
  let made = 0, copyResets = 0;
  const op = wrapOp(M, () => {
    const o = new M.RollingMean(3, "strict");
    if (made++ > 0) {                 // the first call builds raw; later calls build copies
      const reset = o.reset.bind(o);
      o.reset = () => { copyResets++; reset(); };
    }
    return o;
  });
  eqSeq([...op([1, 2, 3][Symbol.iterator]())], [NaN, NaN, 2]);
  assert.equal(copyResets, 0, "a just-made copy is already clean");
  eqSeq([...op([4, 5, 6][Symbol.iterator]())], [NaN, NaN, 5]);
  assert.equal(copyResets, 1, "a reused copy is reset once");
  op.dispose();
});

// G1: normalizeError wraps only calls into WASM. A value thrown by the caller's
// own iterator reaches the caller unchanged, and a C++ validation error becomes
// a RangeError carrying the C++ message.
await ready();

for (const thrown of [70000, "boom", new Error("user")])
  test(`G1: a user iterator's throw of ${String(thrown)} reaches the caller unchanged`, () => {
    const src = { [Symbol.iterator]() { return this; }, next() { throw thrown; } };
    try { [...(ops.RollingMean(3)(src as any) as Iterable<number>)]; assert.fail("no throw"); }
    catch (e) { assert.equal(e, thrown); }
  });

for (const thrown of [70000, "boom", new Error("user")])
  test(`G1: a user async iterator's throw of ${String(thrown)} reaches the caller unchanged`, async () => {
    const src = { [Symbol.asyncIterator]() { return this; }, async next(): Promise<IteratorResult<number>> { throw thrown; } };
    try { for await (const _ of ops.RollingMean(3)(src as any) as AsyncIterable<number>) { /* drain */ } assert.fail("no throw"); }
    catch (e) { assert.equal(e, thrown); }
  });

// A user value whose getter throws, read while the op reads its argument or an
// event of a stream. A user Error whose text looks like a C++ validation
// message must not be rewritten into a RangeError either.
const userThrows = [70000, "boom", new Error("user"), new Error("value must be positive")];
const throwingNd = (thrown: unknown) => ({ get data(): Float64Array { throw thrown; }, shape: [3] });

for (const thrown of userThrows)
  test(`G1: a getter's throw of ${String(thrown)} in a batch argument reaches the caller unchanged`, () => {
    try { ops.RollingMean(3)(throwingNd(thrown) as any); assert.fail("no throw"); }
    catch (e) { assert.equal(e, thrown); }
  });

for (const thrown of userThrows)
  test(`G1: a getter's throw of ${String(thrown)} in a stream event reaches the caller unchanged`, () => {
    try { [...(ops.RollingMean(3)([throwingNd(thrown)][Symbol.iterator]() as any) as Iterable<unknown>)]; assert.fail("no throw"); }
    catch (e) { assert.equal(e, thrown); }
  });

for (const thrown of userThrows)
  test(`G1: a getter's throw of ${String(thrown)} in a row-op or reducer argument reaches the caller unchanged`, () => {
    for (const op of [ops.CrossDemean(), ops.PortfolioReport()]) {
      try { op(throwingNd(thrown) as any); assert.fail("no throw"); }
      catch (e) { assert.equal(e, thrown); }
    }
  });

// The C++ message of RollingMean(0) is "Size must be at least 1." (it names
// the window size, not the word "window").
test("G1: a C++ validation error is a RangeError with its message", () => {
  assert.throws(() => ops.RollingMean(0), (e: unknown) => e instanceof RangeError && /size must be at least 1/i.test((e as Error).message));
});

// A C++ exception that leaves WASM does not unwind the WASM stack; before G1
// about 4,200 caught errors overran the stack into the heap.
test("G1: many caught C++ errors leave the module usable", () => {
  for (let i = 0; i < 20000; i++) assert.throws(() => ops.RollingMean(0), RangeError);
  eqSeq(ops.RollingMean(2)([1, 2, 3]) as number[], [NaN, 1.5, 2.5]);
});

// Every throwing entry frees what it took: a string argument reaches C++ as a
// val, so a rejected constructor or layout leaks nothing (Embind frees a
// std::string argument only on a normal return).
test("G1: failing string-argument calls do not grow the WASM heap", async () => {
  const M: any = await init();
  const { reshape } = await import("../src/index.ts");
  const nd = { data: new Float64Array(8), shape: [2, 4] };
  const fail = () => {
    assert.throws(() => ops.RollingMean(0, "strict"), RangeError);
    assert.throws(() => reshape(nd, "TF->T"), RangeError);
  };
  // The address of a large block moves up when small blocks leak below it.
  const probe = () => { const p = M.allocF64(1 << 16); M.freeBuf(p); return p; };
  for (let i = 0; i < 1000; i++) fail();
  const heap0 = M.HEAPF64.buffer.byteLength, p0 = probe();
  for (let i = 0; i < 50000; i++) fail();
  assert.equal(M.HEAPF64.buffer.byteLength, heap0);
  assert.equal(probe(), p0);
});

// The type string is the mangled name only when wasm/exception_message.cpp
// replaces libc++abi's demangling __get_exception_message; a silently
// un-linked override would give "std::invalid_argument".
test("G1: the exception type comes from the demangler-free override", async () => {
  const M: any = await init();
  let type = "";
  try { M.layoutPlan("TF->T", [2, 4]); assert.fail("no throw"); }
  catch (e) {
    assert.ok(e instanceof (WebAssembly as any).Exception);
    [type] = M.getExceptionMessage(e);
    const { normalizeError } = await import("../src/errors.ts");
    normalizeError(e);   // frees the exception and resets the stack
  }
  assert.equal(type, "St16invalid_argument");
});

test("G1: a non-string string argument is a TypeError", () => {
  assert.throws(() => ops.RollingMean(3, 5 as any), TypeError);
});
