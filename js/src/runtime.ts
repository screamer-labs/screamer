import type { Screamer, RawOp } from "./loader.js";
import { normalizeError } from "./errors.js";
import { checkNd, type NdArray } from "./ndarray.js";
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

const isNd = (x: any): x is NdArray =>
  x != null && x.data instanceof Float64Array && Array.isArray(x.shape);

type OpSource = RawOp | (() => RawOp);
const resolve = (src: OpSource): RawOp => (typeof src === "function" ? (src as () => RawOp)() : src);

export function wrapOp(M: Screamer, src: OpSource): ScreamerOp {
  const make = typeof src === "function" ? (src as () => RawOp) : null;
  const raw = resolve(src);
  const nIn = raw.nIn(), nOut = raw.nOut();
  const inBuf = M.allocF64(nIn), outBuf = M.allocF64(nOut);
  let copies: RawOp[] = [];          // ops for assets 1..K-1 of an asset-event stream
  let disposed = false;
  const free = () => {
    if (disposed) return;
    disposed = true;
    M.freeBuf(inBuf); M.freeBuf(outBuf);
    copies.forEach((c) => c.delete());
    raw.delete();
  };

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

  // An N-D or broadcast call: the C++ shape module splits a packed argument,
  // consumes a size-1 field axis, broadcasts the assets, and runs each column.
  function shaped(args: any[]): NdArray {
    const parts: NdArray[] = args.map((a) =>
      isNd(a) ? a : { data: Float64Array.from(a as ArrayLike<number>), shape: [(a as ArrayLike<number>).length] });
    parts.forEach(checkNd);
    const ptrs: number[] = [];
    try {
      const spec: number[] = [parts.length];
      for (const p of parts) {
        const ptr = M.allocF64(Math.max(1, p.data.length));
        ptrs.push(ptr);
        M.viewF64(ptr, p.data.length).set(p.data);
        spec.push(ptr, p.shape.length, ...p.shape);
      }
      const specPtr = M.allocF64(spec.length);
      ptrs.push(specPtr);
      M.viewF64(specPtr, spec.length).set(spec);
      const shape: number[] = M.shapedResult(specPtr, nIn);
      const n = shape.reduce((a, b) => a * b, 1) * nOut;
      const outPtr = M.allocF64(Math.max(1, n));
      ptrs.push(outPtr);
      M.evalShaped(raw, specPtr, outPtr);
      return { data: new Float64Array(M.viewF64(outPtr, n)), shape: nOut === 1 ? shape : [...shape, nOut] };
    } finally {
      ptrs.forEach((p) => M.freeBuf(p));
    }
  }

  // One event of an asset stream: K values, one per asset, each run by its own
  // op (raw for asset 0). The first event fixes K, as a (T, K) batch does.
  let assets = -1;                    // -1: not started, 0: scalar events, K: asset events
  const eventRow = (v: any): ArrayLike<number> => {
    if (!isNd(v)) {
      if (!isTyped(v) && !isNumArr(v)) throw new TypeError(`unsupported event for a 1-input op: ${typeof v}`);
      return v as ArrayLike<number>;
    }
    const s = v.shape;
    if (s.length === 1 || (s.length === 2 && s[1] === 1)) return v.data;
    throw new TypeError(`an event must hold one value per asset; got shape [${s}]`);
  };
  const assetEvent = (row: ArrayLike<number>): Float64Array | NdArray => {
    const k = row.length;
    const out = new Float64Array(k * nOut);
    for (let j = 0; j < k; j++) {
      M.viewF64(inBuf, 1)[0] = row[j];
      (j === 0 ? raw : copies[j - 1]).evalInto(inBuf, outBuf);
      out.set(M.viewF64(outBuf, nOut), j * nOut);
    }
    return nOut === 1 ? out : { data: out, shape: [k, nOut] };
  };
  const step = (v: any) => {
    if (disposed) throw new Error("operation used after dispose()");
    if (typeof v === "number") {
      if (assets > 0) throw new TypeError(`every event must have the same shape: the first held ${assets} values, this one is a number`);
      assets = 0;
      return event([v]);
    }
    const row = eventRow(v);
    if (assets === -1) {
      if (row.length > 1 && !make) throw new TypeError("events with several assets need an op built by its factory");
      assets = row.length;
      raw.reset();
      while (copies.length < assets - 1) copies.push(make!());
      copies.forEach((c) => c.reset());
    } else if (row.length !== assets) {
      throw new TypeError(`every event must have the same shape: the first held ${assets === 0 ? "a number" : `${assets} values`}, this one holds ${row.length}`);
    }
    return assetEvent(row);
  };

  function* gen(src: Iterable<any>) {
    if (disposed) throw new Error("operation used after dispose()");
    raw.reset();
    assets = -1;
    for (const v of src) yield step(v);
  }
  async function* agen(src: AsyncIterable<any>) {
    if (disposed) throw new Error("operation used after dispose()");
    raw.reset();
    assets = -1;
    for await (const v of src) yield step(v);
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
        if (isNd(a)) return shaped([a]);
        if (isTyped(a)) return batch1(a, true);
        if (isNumArr(a)) return batch1(a, false);
        if (isAsyncIter(a)) return agen(a);
        if (isSyncIter(a)) return gen(a);
        throw new TypeError(`unsupported input for a 1-input op: ${typeof a}`);
      }
      // nIn > 1: N scalars -> one event; N arrays -> columnar batch.
      if (args.length === nIn && args.every((x) => typeof x === "number")) return event(args);
      if (args.length === 1 && isNd(args[0])) return shaped(args);      // one packed (T, ..., nIn) array
      if (args.length === nIn && args.every((x) => isTyped(x) || isNumArr(x) || isNd(x))) {
        if (args.some(isNd)) return shaped(args);
        const cols = args as ArrayLike<number>[];
        const rows = cols[0].length;
        if (cols.some((x) => x.length !== rows)) {
          throw new TypeError(`all input arrays must have the same length (got ${cols.map((a) => a.length).join(", ")})`);
        }
        // Interleave the N input columns into row-major (rows x nIn) order, then
        // one C++ pass. The interleave is JS-side memory writes, no per-event
        // boundary crossing.
        const data = batchInto(rows, (view) => {
          for (let i = 0; i < rows; i++) for (let k = 0; k < nIn; k++) view[i * nIn + k] = cols[k][i];
        });
        return nOut === 1 ? data : ({ data, shape: [rows, nOut] } as NdArray);
      }
      throw new TypeError(`expected ${nIn} numeric inputs`);
    } catch (e) { throw normalizeError(e); }
  };

  const op = call as ScreamerOp;
  op.reset = () => { if (disposed) throw new Error("operation used after dispose()"); raw.reset(); copies.forEach((c) => c.reset()); };
  op.dispose = free;
  (op as any)[Symbol.dispose] = free;
  REG.register(op, free);
  return op;
}

// A nested batch is an array of rows, each row itself an array of assets.
const isNestedBatch = (x: any) =>
  Array.isArray(x) && x.length > 0 && Array.isArray(x[0]) && (Array.isArray(x[0][0]) || isTyped(x[0][0]));

// Wraps a dynamic-width reducer: an op that folds a variable number of assets
// into one event, reading a `(assets, nIn)` block and writing nOut outputs.
// The asset count is a dimension of the data (the assets in a portfolio), not
// a property of the op, so it is read from the input on every call rather than
// fixed at construction the way nIn and nOut are.
//
// Everything that does not involve that extra dimension -- the pre-aggregated
// scalar path, define-then-bind Node composition, reset, dispose -- is the
// ordinary op behaviour, so this delegates to a wrapOp instance instead of
// reimplementing it.
export function wrapReducerOp(M: Screamer, src: OpSource): ScreamerOp {
  const raw = resolve(src);
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

  // Normalize one event into a flat (assets, width) block: an NdArray, a flat
  // numeric array whose length is a multiple of width, or an array of rows.
  function asRow(x: any): { flat: ArrayLike<number>; assets: number } {
    if (isNd(x)) {
      const s = x.shape;
      if (s.length === 2) {
        if (s[1] !== width) {
          throw new TypeError(`expected an event of shape (assets, ${width}), got [${s}]`);
        }
        return { flat: x.data, assets: s[0] };
      }
      if (s.length === 1) return asRow(x.data);
      throw new TypeError(`expected a rank-1 or rank-2 event, got shape [${s}]`);
    }
    if (isTyped(x) || isNumArr(x)) {
      const n = (x as ArrayLike<number>).length;
      if (n === 0 || n % width !== 0) {
        throw new TypeError(`a flat event must hold a positive multiple of ${width} values, got ${n}`);
      }
      return { flat: x, assets: n / width };
    }
    if (Array.isArray(x) && x.length > 0) {
      const assets = x.length;
      const flat = new Float64Array(assets * width);
      for (let g = 0; g < assets; g++) {
        const row = x[g];
        if (!isTyped(row) && !isNumArr(row)) {
          throw new TypeError(`asset ${g} is not a numeric array`);
        }
        if (row.length !== width) {
          throw new TypeError(`each asset must hold ${width} values, got ${row.length} at asset ${g}`);
        }
        flat.set(row as ArrayLike<number>, g * width);
      }
      return { flat, assets };
    }
    throw new TypeError(`unsupported event for a reducer op: ${typeof x}`);
  }

  // Single-output ops yield a bare number per event and a flat array per batch,
  // the same convention wrapOp follows, so a future 1-output reducer does not
  // silently return a width-1 container where every other op returns a scalar.
  const event = (flat: ArrayLike<number>, assets: number): number | number[] => {
    const need = assets * width;
    if (need > rowCap) {
      if (rowBuf) M.freeBuf(rowBuf);
      rowBuf = M.allocF64(need);
      rowCap = need;
    }
    M.viewF64(rowBuf, need).set(flat as number[]);
    raw.reduceInto!(rowBuf, outBuf, assets);
    const o = M.viewF64(outBuf, nOut);
    return nOut === 1 ? o[0] : Array.from(o);
  };

  const batch = (flat: ArrayLike<number>, rows: number, assets: number): NdArray | Float64Array => {
    if (rows === 0) {
      return nOut === 1 ? new Float64Array(0) : { data: new Float64Array(0), shape: [0, nOut] };
    }
    const inN = rows * assets * width;
    const inP = M.allocF64(inN), outP = M.allocF64(rows * nOut);
    try {
      M.viewF64(inP, inN).set(flat as number[]);
      raw.reduceBatchInto!(inP, outP, rows, assets);
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
    const assets = first.assets;
    const flat = new Float64Array(rows * assets * width);
    flat.set(first.flat as number[], 0);
    for (let r = 1; r < rows; r++) {
      const { flat: f, assets: g } = asRow(x[r]);
      if (g !== assets) {
        throw new TypeError(`every event must hold ${assets} assets, got ${g} at event ${r}`);
      }
      flat.set(f as number[], r * assets * width);
    }
    return batch(flat, rows, assets);
  };

  function* gen(src: Iterable<any>) {
    raw.reset();
    for (const row of src) { const { flat, assets } = asRow(row); yield event(flat, assets); }
  }
  async function* agen(src: AsyncIterable<any>) {
    raw.reset();
    for await (const row of src) { const { flat, assets } = asRow(row); yield event(flat, assets); }
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
            throw new TypeError(`expected a batch of shape (events, assets, ${width}), got [${s}]`);
          }
          return batch(x.data, s[0], s[1]);
        }
        const { flat, assets } = asRow(x);
        return event(flat, assets);
      }
      if (isNestedBatch(x)) return nestedBatch(x);
      if (isAsyncIter(x)) return agen(x);
      if (isSyncIter(x)) return gen(x);
      const { flat, assets } = asRow(x);
      return event(flat, assets);
    } catch (e) { throw normalizeError(e); }
  };

  const op = call as ScreamerOp;
  op.reset = () => { if (disposed) throw new Error("operation used after dispose()"); raw.reset(); };
  op.dispose = free;
  (op as any)[Symbol.dispose] = free;
  REG.register(op, free);
  return op;
}

// Wraps a row op: each event is one row of K assets with fieldsIn() values
// per asset (one value for a cross-sectional op; target and prices for a
// portfolio engine), and returns fieldsOut() values per asset. K is a
// dimension of the data, fixed by the first row after reset(), so it travels
// with every call. With one input per field:
//
//   (T, K) NdArrays or arrays of rows  -> (T, K) or (T, K, fieldsOut) NdArray
//   one row each (Float64Array/number[]) -> one row, or a (K, fieldsOut) NdArray
//   an iterable of rows (one field)     -> a generator of rows
//
// Mirrors the Python binding of screamer.RowOp.
export function wrapRowOp(M: Screamer, src: OpSource): ScreamerOp {
  const raw = resolve(src);
  const fi = raw.fieldsIn!(), fo = raw.fieldsOut!();
  let evIn = 0, evOut = 0, cap = 0;
  let disposed = false;
  const free = () => {
    if (disposed) return;
    disposed = true;
    if (evIn) { M.freeBuf(evIn); M.freeBuf(evOut); }
    raw.delete();
  };

  // One event from fi rows of k values each.
  const event = (rows: ArrayLike<number>[]): Float64Array => {
    const k = rows[0].length;
    if (rows.some((r) => r.length !== k)) throw new TypeError("all inputs must hold the same number of assets");
    if (k > cap) {
      if (evIn) { M.freeBuf(evIn); M.freeBuf(evOut); }
      evIn = M.allocF64(fi * k); evOut = M.allocF64(fo * k); cap = k;
    }
    const view = M.viewF64(evIn, fi * k);
    rows.forEach((r, f) => view.set(r as number[], f * k));
    raw.mapRowInto!(evIn, evOut, k);
    return new Float64Array(M.viewF64(evOut, fo * k));
  };

  // T events from fi (T, K) blocks, each flat and row-major.
  const batch = (blocks: ArrayLike<number>[], T: number, k: number): NdArray => {
    if (k === 0) throw new TypeError("a row op needs at least one asset");
    const shape = fo === 1 ? [T, k] : [T, k, fo];
    if (T === 0) return { data: new Float64Array(0), shape };
    const nIn = T * fi * k, nOut = T * fo * k;
    const inP = M.allocF64(nIn), outP = M.allocF64(nOut);
    try {
      const view = M.viewF64(inP, nIn);
      if (fi === 1) view.set(blocks[0] as number[]);
      else for (let t = 0; t < T; t++)
        for (let f = 0; f < fi; f++)
          for (let j = 0; j < k; j++) view[(t * fi + f) * k + j] = blocks[f][t * k + j];
      raw.mapRowsInto!(inP, outP, T, k);
      return { data: new Float64Array(M.viewF64(outP, nOut)), shape };
    } finally {
      M.freeBuf(inP); M.freeBuf(outP);
    }
  };

  // A (T, K) input as a flat block: an NdArray or an array of rows.
  const asBlock = (x: any): { flat: ArrayLike<number>; T: number; k: number } | null => {
    if (isNd(x)) {
      // Rule 3: a [T, K, 1] array reads as [T, K]; the data is already in that order.
      const unit = x.shape.length === 3 && x.shape[2] === 1;
      if (x.shape.length !== 2 && !unit) throw new TypeError(`expected a (T, K) array, got shape [${x.shape}]`);
      return { flat: x.data, T: x.shape[0], k: x.shape[1] };
    }
    if (Array.isArray(x) && x.length > 0 && x.every((r) => isTyped(r) || isNumArr(r))) {
      const T = x.length, k = x[0].length;
      const flat = new Float64Array(T * k);
      x.forEach((r: ArrayLike<number>, i: number) => {
        if (r.length !== k) throw new TypeError(`every row must hold ${k} values, got ${r.length} at row ${i}`);
        flat.set(r as number[], i * k);
      });
      return { flat, T, k };
    }
    return null;
  };

  const eventOut = (rows: ArrayLike<number>[]): any => {
    const out = event(rows);
    if (fo !== 1) return { data: out, shape: [rows[0].length, fo] } as NdArray;
    return isTyped(rows[0]) ? out : Array.from(out);
  };

  // An event is a row of K values, or an NdArray of shape [K] or [K, 1] (rule 3).
  const asRow = (row: any): ArrayLike<number> => {
    if (!isNd(row)) return row;
    if (row.shape.length === 1 || (row.shape.length === 2 && row.shape[1] === 1)) return row.data;
    throw new TypeError(`each event of a row stream is one row of K values, got shape [${row.shape}]`);
  };
  function* gen(src: Iterable<any>) {
    raw.reset();
    for (const row of src) yield eventOut([asRow(row)]);
  }
  async function* agen(src: AsyncIterable<any>) {
    raw.reset();
    for await (const row of src) yield eventOut([asRow(row)]);
  }

  const call = (...args: any[]) => {
    try {
      if (disposed) throw new Error("operation used after dispose()");
      if (args.some(isNode)) throw new TypeError("row ops run on arrays and row streams, not inside a Pipeline");
      if (args.length !== fi) throw new TypeError(`this op takes ${fi} inputs, got ${args.length}`);
      const x = args[0];
      if (fi === 1 && isAsyncIter(x)) return agen(x);
      if (fi === 1 && isSyncIter(x)) return gen(x);
      if (args.every((a) => isTyped(a) || isNumArr(a) || (isNd(a) && a.shape.length === 1)))
        return eventOut(args.map((a) => (isNd(a) ? a.data : a)));
      const blocks = args.map(asBlock);
      if (blocks.some((b) => b === null))
        throw new TypeError("expected (T, K) arrays, arrays of rows, or one row per input");
      const b0 = blocks[0]!;
      if (blocks.some((b) => b!.T !== b0.T || b!.k !== b0.k))
        throw new TypeError("all inputs must have the same shape");
      return batch(blocks.map((b) => b!.flat), b0.T, b0.k);
    } catch (e) { throw normalizeError(e); }
  };

  const op = call as ScreamerOp;
  op.reset = () => { if (disposed) throw new Error("operation used after dispose()"); raw.reset(); };
  op.dispose = free;
  (op as any)[Symbol.dispose] = free;
  // Marks a row op (rows of K assets), so callers such as positionTest can
  // tell a portfolio engine from a per-column one.
  (op as any).rowOp = { fieldsIn: fi, fieldsOut: fo };
  REG.register(op, free);
  return op;
}
