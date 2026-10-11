import { test } from "node:test";
import assert from "node:assert/strict";
import { init } from "../src/index.ts";
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
