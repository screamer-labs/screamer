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

  // Run a whole (rows x nIn) row-major block through evalBatchInto in one C++
  // pass and return the (rows x nOut) output as a fresh Float64Array copied off
  // the WASM heap. `fill` writes the interleaved input into the heap view.
  function batchInto(rows: number, fill: (view: Float64Array) => void): Float64Array {
    if (rows === 0) return new Float64Array(0);
    const inPtr = M.allocF64(rows * nIn), outPtr = M.allocF64(rows * nOut);
    try {
      fill(M.viewF64(inPtr, rows * nIn));
      raw.reset();
      raw.evalBatchInto(inPtr, outPtr, rows);
      raw.reset();
      return new Float64Array(M.viewF64(outPtr, rows * nOut));
    } finally {
      M.freeBuf(inPtr); M.freeBuf(outPtr);
    }
  }

  function batch1(arr: ArrayLike<number>, typed: boolean): any {
    const rows = arr.length;
    const data = batchInto(rows, (view) => view.set(arr as unknown as ArrayLike<number> as number[]));
    if (nOut !== 1) return { data, shape: [rows, nOut] } as NdArray;
    if (typed) return data;
    const out = new Array(rows);
    for (let i = 0; i < rows; i++) out[i] = data[i];
    return out;
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
        // Interleave the N input columns into row-major (rows x nIn) order, then
        // one C++ pass. The interleave is JS-side memory writes, no per-event
        // boundary crossing.
        const cols = args as ArrayLike<number>[];
        const data = batchInto(rows, (view) => {
          for (let i = 0; i < rows; i++) for (let k = 0; k < nIn; k++) view[i * nIn + k] = cols[k][i];
        });
        return nOut === 1 ? data : ({ data, shape: [rows, nOut] } as NdArray);
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

const isNd = (x: any): x is NdArray =>
  x != null && x.data instanceof Float64Array && Array.isArray(x.shape);

// A nested batch is an array of rows, each row itself an array of groups.
const isNestedBatch = (x: any) =>
  Array.isArray(x) && x.length > 0 && Array.isArray(x[0]) && (Array.isArray(x[0][0]) || isTyped(x[0][0]));

// Wraps a dynamic-width reducer: an op that folds a variable number of groups
// into one event, reading a `(groups, nIn)` block and writing nOut outputs.
// The group count is a dimension of the data (the assets in a portfolio), not
// a property of the op, so it is read from the input on every call rather than
// fixed at construction the way nIn and nOut are.
//
// Everything that does not involve that extra dimension -- the pre-aggregated
// scalar path, define-then-bind Node composition, reset, dispose -- is the
// ordinary op behaviour, so this delegates to a wrapOp instance instead of
// reimplementing it.
export function wrapReducerOp(M: Screamer, raw: RawOp): ScreamerOp {
  const width = raw.nIn(), nOut = raw.nOut();
  const base = wrapOp(M, raw);

  // Scratch for the per-event path, grown on demand and reused across events.
  // Batch calls size their buffers exactly and release them, so a single large
  // batch does not pin memory for the lifetime of the op.
  let rowBuf = 0, rowCap = 0;
  const outBuf = M.allocF64(nOut);
  let disposed = false;
  const free = () => {
    if (disposed) return;
    disposed = true;
    if (rowBuf) M.freeBuf(rowBuf);
    M.freeBuf(outBuf);
    base.dispose();
  };

  // Normalize one event into a flat (groups, width) block: an NdArray, a flat
  // numeric array whose length is a multiple of width, or an array of rows.
  function asRow(x: any): { flat: ArrayLike<number>; groups: number } {
    if (isNd(x)) {
      const s = x.shape;
      if (s.length === 2) {
        if (s[1] !== width) {
          throw new TypeError(`expected an event of shape (groups, ${width}), got [${s}]`);
        }
        return { flat: x.data, groups: s[0] };
      }
      if (s.length === 1) return asRow(x.data);
      throw new TypeError(`expected a rank-1 or rank-2 event, got shape [${s}]`);
    }
    if (isTyped(x) || isNumArr(x)) {
      const n = (x as ArrayLike<number>).length;
      if (n === 0 || n % width !== 0) {
        throw new TypeError(`a flat event must hold a positive multiple of ${width} values, got ${n}`);
      }
      return { flat: x, groups: n / width };
    }
    if (Array.isArray(x) && x.length > 0) {
      const groups = x.length;
      const flat = new Float64Array(groups * width);
      for (let g = 0; g < groups; g++) {
        const row = x[g];
        if (!isTyped(row) && !isNumArr(row)) {
          throw new TypeError(`group ${g} is not a numeric array`);
        }
        if (row.length !== width) {
          throw new TypeError(`each group must hold ${width} values, got ${row.length} at group ${g}`);
        }
        flat.set(row as ArrayLike<number>, g * width);
      }
      return { flat, groups };
    }
    throw new TypeError(`unsupported event for a reducer op: ${typeof x}`);
  }

  // Single-output ops yield a bare number per event and a flat array per batch,
  // the same convention wrapOp follows, so a future 1-output reducer does not
  // silently return a width-1 container where every other op returns a scalar.
  const event = (flat: ArrayLike<number>, groups: number): number | number[] => {
    const need = groups * width;
    if (need > rowCap) {
      if (rowBuf) M.freeBuf(rowBuf);
      rowBuf = M.allocF64(need);
      rowCap = need;
    }
    M.viewF64(rowBuf, need).set(flat as number[]);
    raw.reduceInto!(rowBuf, outBuf, groups);
    const o = M.viewF64(outBuf, nOut);
    return nOut === 1 ? o[0] : Array.from(o);
  };

  const batch = (flat: ArrayLike<number>, rows: number, groups: number): NdArray | Float64Array => {
    if (rows === 0) {
      return nOut === 1 ? new Float64Array(0) : { data: new Float64Array(0), shape: [0, nOut] };
    }
    const inN = rows * groups * width;
    const inP = M.allocF64(inN), outP = M.allocF64(rows * nOut);
    try {
      M.viewF64(inP, inN).set(flat as number[]);
      raw.reduceBatchInto!(inP, outP, rows, groups);
      const data = new Float64Array(M.viewF64(outP, rows * nOut));
      return nOut === 1 ? data : { data, shape: [rows, nOut] };
    } finally {
      M.freeBuf(inP);
      M.freeBuf(outP);
    }
  };

  const nestedBatch = (x: any[]): NdArray | Float64Array => {
    const rows = x.length;
    const first = asRow(x[0]);
    const groups = first.groups;
    const flat = new Float64Array(rows * groups * width);
    flat.set(first.flat as number[], 0);
    for (let r = 1; r < rows; r++) {
      const { flat: f, groups: g } = asRow(x[r]);
      if (g !== groups) {
        throw new TypeError(`every event must hold ${groups} groups, got ${g} at event ${r}`);
      }
      flat.set(f as number[], r * groups * width);
    }
    return batch(flat, rows, groups);
  };

  function* gen(src: Iterable<any>) {
    raw.reset();
    for (const row of src) { const { flat, groups } = asRow(row); yield event(flat, groups); }
  }
  async function* agen(src: AsyncIterable<any>) {
    raw.reset();
    for await (const row of src) { const { flat, groups } = asRow(row); yield event(flat, groups); }
  }

  const call = (...args: any[]) => {
    try {
      if (disposed) throw new Error("operation used after dispose()");
      // Anything that isn't a single reducer-shaped input -- N scalars, Node
      // composition, an arity error -- is the ordinary op contract.
      if (args.length !== 1) return base(...args);
      const x = args[0];
      if (isNode(x) || typeof x === "number") return base(x);
      if (isNd(x)) {
        const s = x.shape;
        if (s.length === 3) {
          if (s[2] !== width) {
            throw new TypeError(`expected a batch of shape (events, groups, ${width}), got [${s}]`);
          }
          return batch(x.data, s[0], s[1]);
        }
        const { flat, groups } = asRow(x);
        return event(flat, groups);
      }
      if (isNestedBatch(x)) return nestedBatch(x);
      if (isAsyncIter(x)) return agen(x);
      if (isSyncIter(x)) return gen(x);
      const { flat, groups } = asRow(x);
      return event(flat, groups);
    } catch (e) { throw normalizeError(e); }
  };

  const op = call as ScreamerOp;
  op.reset = () => { if (disposed) throw new Error("operation used after dispose()"); raw.reset(); };
  op.dispose = free;
  (op as any)[Symbol.dispose] = free;
  REG.register(op, free);
  return op;
}
