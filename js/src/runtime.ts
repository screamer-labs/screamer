import type { Screamer, RawOp } from "./loader.js";
import { normalizeError } from "./errors.js";
import type { NdArray } from "./ndarray.js";

// The `explicit resource management` symbol ships in every runtime we target
// (Node 20+, modern browsers) but is absent from the ES2022 lib typings used
// here. Declare it so `[Symbol.dispose]` typechecks without widening `lib`.
declare global {
  interface SymbolConstructor { readonly dispose: unique symbol; }
}

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
    M.viewF64(inBuf, nIn).set(inputs as number[]);
    raw.evalInto(inBuf, outBuf);
    const o = M.viewF64(outBuf, nOut);
    return nOut === 1 ? o[0] : Array.from(o);
  };

  const isTyped = (x: any) => x instanceof Float64Array;
  const isNumArr = (x: any) => Array.isArray(x) && (x.length === 0 || typeof x[0] === "number");
  const isSyncIter = (x: any) => x != null && typeof x[Symbol.iterator] === "function" && !isTyped(x) && !isNumArr(x);
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
        raw.reset();
        const rows = (args[0] as ArrayLike<number>).length;
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
  op.reset = () => raw.reset();
  op.dispose = free;
  (op as any)[Symbol.dispose] = free;
  REG.register(op, free);
  return op;
}
