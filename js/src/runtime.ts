import type { Screamer, RawOp } from "./loader.js";
import { normalizeError } from "./errors.js";
import type { NdArray } from "./ndarray.js";
import { Node, isNode } from "./node.js";

// `Symbol.dispose` (explicit resource management) is provided by the
// "ESNext.Disposable" lib entry in tsconfig.json.
const REG = new FinalizationRegistry<() => void>((free) => free());

export type ScreamerOp = {
  (...args: any[]): any;
  reset(): void;
  dispose(): void;
  [Symbol.dispose](): void;
};

export function wrapOp(M: Screamer, raw: RawOp): ScreamerOp {
  const nIn = raw.nIn(), nOut = raw.nOut();
  const inBuf = M.allocF64(nIn), outBuf = M.allocF64(nOut);
  let disposed = false;
  const free = () => { if (!disposed) { disposed = true; M.freeBuf(inBuf); M.freeBuf(outBuf); raw.delete(); } };

  const event = (inputs: ArrayLike<number>): number | number[] => {
    if (disposed) throw new Error("operation used after dispose()");
    M.viewF64(inBuf, nIn).set(inputs as number[]);
    raw.evalInto(inBuf, outBuf);
    const o = M.viewF64(outBuf, nOut);
    return nOut === 1 ? o[0] : Array.from(o);
  };

  const isTyped = (x: any) => x instanceof Float64Array;
  const isNumArr = (x: any) => Array.isArray(x) && x.every((v) => typeof v === "number");
  // Strings are technically Symbol.iterator-bearing (of chars, not numbers) and
  // must NOT be accepted as a streaming source; without this guard `op("x")`
  // would silently return a lazy generator instead of raising. Plain arrays
  // are excluded too (not just numeric ones): a non-numeric/mixed array
  // (e.g. `[1, "x"]`) must raise a synchronous TypeError from the dispatcher
  // below, not be treated as a lazy streaming source that only fails once
  // consumed.
  const isSyncIter = (x: any) =>
    x != null && typeof x !== "string" && typeof x[Symbol.iterator] === "function" && !isTyped(x) && !Array.isArray(x);
  const isAsyncIter = (x: any) => x != null && typeof x[Symbol.asyncIterator] === "function";

  function batch1(arr: ArrayLike<number>, typed: boolean): any {
    raw.reset();
    const rows = arr.length;
    if (nOut === 1) {
      const out = typed ? new Float64Array(rows) : new Array(rows);
      for (let i = 0; i < rows; i++) (out as any)[i] = event([arr[i]]);
      raw.reset();
      return out;
    }
    const data = new Float64Array(rows * nOut);
    for (let i = 0; i < rows; i++) { const o = event([arr[i]]) as number[]; data.set(o, i * nOut); }
    raw.reset();
    return { data, shape: [rows, nOut] } as NdArray;
  }

  function* gen(src: Iterable<number>) {
    raw.reset();
    for (const v of src) yield event([v]);
  }
  async function* agen(src: AsyncIterable<number>) {
    raw.reset();
    for await (const v of src) yield event([v]);
  }

  const call = (...args: any[]) => {
    try {
      if (disposed) throw new Error("operation used after dispose()");
      // Define-then-bind: if any argument is a symbolic Node, defer the call and
      // return a functor Node holding BOTH the raw C++ op (for `opPtr` at graph
      // compile) and this wrapper (so the Pipeline can pin its lifetime; the
      // Embind layer keeps only a non-owning EvalOp*).
      if (args.some(isNode)) {
        return new Node({ functor: raw, wrapper: op }, args.filter(isNode));
      }
      if (nIn === 1) {
        const a = args[0];
        if (typeof a === "number") return event([a]);
        if (isTyped(a)) return batch1(a, true);
        if (isNumArr(a)) return batch1(a, false);
        if (isAsyncIter(a)) return agen(a);
        if (isSyncIter(a)) return gen(a);
        throw new TypeError(`unsupported input for a 1-input op: ${typeof a}`);
      }
      // nIn > 1: N scalars -> one event; N arrays -> columnar batch.
      if (args.length === nIn && args.every((x) => typeof x === "number")) return event(args);
      if (args.length === nIn && args.every((x) => isTyped(x) || isNumArr(x))) {
        const rows = (args[0] as ArrayLike<number>).length;
        if (args.some((x) => x.length !== rows)) {
          throw new TypeError(`all input arrays must have the same length (got ${args.map((a) => a.length).join(", ")})`);
        }
        raw.reset();
        const single = nOut === 1;
        const out = single ? new Float64Array(rows) : new Float64Array(rows * nOut);
        for (let i = 0; i < rows; i++) {
          const o = event(args.map((c) => c[i]));
          if (single) (out as Float64Array)[i] = o as number; else (out as Float64Array).set(o as number[], i * nOut);
        }
        raw.reset();
        return single ? out : ({ data: out, shape: [rows, nOut] } as NdArray);
      }
      throw new TypeError(`expected ${nIn} numeric inputs`);
    } catch (e) { throw normalizeError(e); }
  };

  const op = call as ScreamerOp;
  op.reset = () => { if (disposed) throw new Error("operation used after dispose()"); raw.reset(); };
  op.dispose = free;
  (op as any)[Symbol.dispose] = free;
  REG.register(op, free);
  return op;
}
