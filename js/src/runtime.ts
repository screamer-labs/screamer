import type { Screamer, RawOp } from "./loader.js";
import { normalizeError } from "./errors.js";
import { checkNd, type NdArray } from "./ndarray.js";
import { Node, isNode } from "./node.js";

// `Symbol.dispose` (explicit resource management) is provided by the
// "ESNext.Disposable" lib entry in tsconfig.json.
const REG = new FinalizationRegistry<() => void>((free) => free());

export type ScreamerOp = {
  (...args: any[]): any;
  /** Number of input fields per event (Python's `num_inputs`). */
  readonly numInputs: number;
  /** Number of output fields per event (Python's `num_outputs`). */
  readonly numOutputs: number;
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

// Whether a call is a stream: "sync" when every argument is an iterator,
// "async" when every argument is an async iterable, null when none is. A
// stream takes n sources, or one source whose events hold all n fields.
function streamKind(args: any[], n: number): "sync" | "async" | null {
  let nAsync = 0, nSync = 0;
  for (const a of args) {
    if (isAsyncIter(a)) nAsync++;
    else if (isSyncIter(a)) nSync++;
  }
  if (nAsync + nSync === 0) return null;
  if (nAsync && nSync)
    throw new TypeError("async and synchronous inputs cannot be mixed: pass every input as an async iterable, or none");
  if (nAsync + nSync !== args.length)
    throw new TypeError("iterator and array inputs cannot be mixed: pass every input as an iterator of events, or none");
  if (args.length !== n && args.length !== 1)
    throw new TypeError(`this op takes ${n} inputs, got ${args.length} ${nAsync ? "async iterables" : "iterators"}`);
  return nAsync ? "async" : "sync";
}

// One stream's per-event work: `step` runs one event (one item per source),
// `close` releases what the stream holds. Each stream opens its own.
type Stepper = { step(items: any[]): any; close?(): void };

// One event per step: the next item of every source, in order (the yielded
// array is reused). Ends with the first exhausted source. When the stream ends
// for any reason (exhausted, return(), throw(), or an error in an event), every
// other source is closed with return(), as a for...of loop closes its source.
function* zipSources(srcs: Iterable<any>[]): Generator<any[], void, undefined> {
  const its = srcs.map((s) => s[Symbol.iterator]());
  const items = new Array(its.length);
  let ended = -1, failed = false;
  try {
    for (;;) {
      for (let i = 0; i < its.length; i++) {
        const r = its[i].next();
        if (r.done) { ended = i; return; }
        items[i] = r.value;
      }
      yield items;
    }
  } catch (e) {
    failed = true;
    throw e;
  } finally {
    let err: unknown, bad = false;
    for (let i = 0; i < its.length; i++) {
      if (i === ended) continue;
      try { its[i].return?.(); } catch (e) { if (!bad) { err = e; bad = true; } }
    }
    if (bad && !failed) throw err;
  }
}
async function* zipAsyncSources(srcs: AsyncIterable<any>[]): AsyncGenerator<any[], void, undefined> {
  const its = srcs.map((s) => s[Symbol.asyncIterator]());
  const items = new Array(its.length);
  let ended = -1, failed = false;
  try {
    for (;;) {
      for (let i = 0; i < its.length; i++) {
        const r = await its[i].next();
        if (r.done) { ended = i; return; }
        items[i] = r.value;
      }
      yield items;
    }
  } catch (e) {
    failed = true;
    throw e;
  } finally {
    // Close every source even when one return() rejects.
    const res = await Promise.allSettled(its.map((it, i) => (i === ended ? undefined : it.return?.())));
    const bad = res.find((r): r is PromiseRejectedResult => r.status === "rejected");
    if (bad && !failed) throw bad.reason;
  }
}

// The generator a stream call returns. Nothing is read, and the stepper is not
// opened, before the first next(). An error in an event ends the stream and
// closes the sources. `owner` is the op wrapper: the generator holds it, so the
// finalizer cannot free a temporary op, such as `RollingMean(3)(src)`, while
// its stream runs.
function* runStream(srcs: Iterable<any>[], open: (n: number) => Stepper, owner: unknown) {
  void owner;
  const s = open(srcs.length);
  try {
    for (const items of zipSources(srcs)) yield s.step(items);
  } finally {
    s.close?.();
  }
}
async function* runAsyncStream(srcs: AsyncIterable<any>[], open: (n: number) => Stepper, owner: unknown) {
  void owner;
  const s = open(srcs.length);
  try {
    for await (const items of zipAsyncSources(srcs)) yield s.step(items);
  } finally {
    s.close?.();
  }
}

export function wrapOp(M: Screamer, src: OpSource): ScreamerOp {
  const make = typeof src === "function" ? (src as () => RawOp) : null;
  const raw = resolve(src);
  const nIn = raw.nIn(), nOut = raw.nOut();
  const inBuf = M.allocF64(nIn), outBuf = M.allocF64(nOut);
  let disposed = false;
  const free = () => {
    if (disposed) return;
    disposed = true;
    M.freeBuf(inBuf); M.freeBuf(outBuf);
    [...live].forEach((h) => h.release());
    pool.forEach((c) => c.delete());
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

  // ---- Streams -------------------------------------------------------------
  //
  // A stream reads one item per source per event: nIn sources of one field
  // each, or, with nIn > 1, one source whose events hold every field (an array
  // of nIn numbers, an array of nIn fields, or a (..., nIn) NdArray). The first
  // event fixes the event shape. Numbers give a number (or nOut numbers) per
  // event. Arrays hold one value per asset: the inputs broadcast like a batch
  // call (the C++ shape module plans the columns), and every asset column runs
  // its own op, raw for column 0 and copies from the factory for the rest.
  //
  // Each stream has its own state and its own copies, so two streams of one op
  // never mix event shapes. Column 0 is raw itself, as in Python. A finished
  // stream returns its copies to `pool` for the next stream; `live` holds the
  // release of every open stream, so dispose() frees what they hold.
  const pool: RawOp[] = [];
  const live = new Set<{ release(): void; reset(): void }>();
  const EMPTY: number[] = [];

  const openStream = (n: number): Stepper => {
    if (disposed) throw new Error("operation used after dispose()");
    raw.reset();
    const UNKNOWN = 0, SCALAR = 1, ASSETS = 2;
    let mode = UNKNOWN;
    const unpack = n === 1 && nIn > 1;            // one source holds all nIn fields
    let plan: { cols: number; offset: number[]; shapes: number[][]; out: number[] | null } | null = null;
    let copies: RawOp[] = [];                     // columns 1..cols-1
    let colIn = 0, colOut = 0;                    // heap blocks of (cols x nIn) and (cols x nOut)
    // Per-event views of the current event's parts, reused across events.
    const partData: ArrayLike<number>[] = [], partShape: number[][] = [];
    const numBuf: Float64Array[] = [], lenShape: number[][] = [];

    const handle = {
      release() {
        if (!live.delete(handle)) return;
        if (colIn) { M.freeBuf(colIn); M.freeBuf(colOut); colIn = colOut = 0; }
        pool.push(...copies);
        copies = [];
      },
      reset() { copies.forEach((c) => c.reset()); },
    };
    live.add(handle);

    const scalarValues = (items: any[]): ArrayLike<number> | null => {
      if (unpack) {
        const v = items[0];
        if ((isTyped(v) || isNumArr(v)) && v.length === nIn) return v;
        if (isNd(v) && v.shape.length === 1 && v.shape[0] === nIn && v.data.length === nIn) return v.data;
        return null;
      }
      for (let i = 0; i < items.length; i++) if (typeof items[i] !== "number") return null;
      return items;
    };

    // Reads the event's parts (one per field, or one packed part) into
    // partData / partShape and returns their count.
    const readParts = (items: any[]): number => {
      const src = unpack && Array.isArray(items[0]) ? items[0] : items;
      for (let k = 0; k < src.length; k++) {
        const v = src[k];
        if (typeof v === "number") {
          const b = (numBuf[k] ??= new Float64Array(1));
          b[0] = v;
          partData[k] = b; partShape[k] = EMPTY;
        } else if (isNd(v)) {
          checkNd(v);
          partData[k] = v.data; partShape[k] = v.shape;
        } else if (isTyped(v) || isNumArr(v)) {
          const s = (lenShape[k] ??= [0]);
          s[0] = v.length;
          partData[k] = v; partShape[k] = s;
        } else {
          throw new TypeError(`unsupported event: ${Array.isArray(v) ? "an array of non-numbers" : typeof v}`);
        }
      }
      return src.length;
    };

    const fmtShapes = (shapes: number[][]) => shapes.map((s) => `[${s}]`).join(" and ");

    // The first asset event fixes the shape and takes one op per asset column
    // from the pool or the factory. The plan is committed only after every copy
    // exists; a failed factory call ends the stream like any other error.
    const startAssets = (k: number) => {
      const shapes = partShape.slice(0, k).map((s) => s.slice());
      const p = M.eventPlan(shapes.map((s) => [1, ...s]), nIn) as { shape: number[]; columns: number; offset: number[] };
      const cols = p.columns;
      if (cols > 1 && !make) throw new TypeError("events with several assets need an op built by its factory");
      while (copies.length < cols - 1) copies.push(pool.pop() ?? make!());
      raw.reset();
      copies.forEach((c) => c.reset());
      colIn = M.allocF64(Math.max(1, cols * nIn)); colOut = M.allocF64(Math.max(1, cols * nOut));
      const out = p.shape.length === 1 && nOut === 1 ? null : (nOut === 1 ? p.shape : [...p.shape, nOut]);
      plan = { cols, offset: p.offset, shapes, out };
      mode = ASSETS;
    };

    const sameShapes = (k: number): boolean => {
      const want = plan!.shapes;
      if (k !== want.length) return false;
      for (let j = 0; j < k; j++) {
        const a = partShape[j], b = want[j];
        if (a.length !== b.length) return false;
        for (let d = 0; d < a.length; d++) if (a[d] !== b[d]) return false;
      }
      return true;
    };

    // One asset event: gather every column's fields into one heap block, run
    // each column's op on its slice, and copy the outputs off the heap once.
    const assetEvent = (items: any[]): Float64Array | NdArray => {
      const k = readParts(items);
      if (!plan) startAssets(k);
      else if (!sameShapes(k))
        throw new TypeError(`every event must have the same shape: the first was ${fmtShapes(plan.shapes)}, this one is ${fmtShapes(partShape.slice(0, k))}`);
      const { cols, offset, out } = plan!;
      const packed = k === 1 && nIn > 1;
      const view = M.viewF64(colIn, cols * nIn);
      for (let i = 0; i < nIn; i++) {
        const d = partData[packed ? 0 : i], o = i * cols;
        for (let c = 0; c < cols; c++) view[c * nIn + i] = d[offset[o + c]];
      }
      for (let c = 0; c < cols; c++)
        (c === 0 ? raw : copies[c - 1]).evalInto(colIn + 8 * c * nIn, colOut + 8 * c * nOut);
      const data = new Float64Array(M.viewF64(colOut, cols * nOut));
      return out ? { data, shape: out.slice() } : data;
    };

    const step = (items: any[]) => {
      try {
        if (disposed) throw new Error("operation used after dispose()");
        if (mode !== ASSETS) {
          const v = scalarValues(items);
          if (v) { mode = SCALAR; return event(v); }
          if (mode === SCALAR) throw new TypeError("every event must have the same shape: the first events were numbers, this one is not");
        }
        return assetEvent(items);
      } catch (e) { throw normalizeError(e); }
    };
    return { step, close: handle.release };
  };

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
      const kind = streamKind(args, nIn);
      if (kind) return kind === "async" ? runAsyncStream(args, openStream, op) : runStream(args, openStream, op);
      if (nIn === 1) {
        const a = args[0];
        if (typeof a === "number") return event([a]);
        if (isNd(a)) return shaped([a]);
        if (isTyped(a)) return batch1(a, true);
        if (isNumArr(a)) return batch1(a, false);
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
  op.reset = () => { if (disposed) throw new Error("operation used after dispose()"); raw.reset(); live.forEach((h) => h.reset()); };
  op.dispose = free;
  (op as any)[Symbol.dispose] = free;
  (op as any).numInputs = nIn;
  (op as any).numOutputs = nOut;
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

  // A stream of (assets, width) events, through the same iterators as wrapOp.
  const step = (items: any[]) => {
    try {
      if (disposed) throw new Error("operation used after dispose()");
      const { flat, assets } = asRow(items[0]);
      return event(flat, assets);
    } catch (e) { throw normalizeError(e); }
  };
  const openStream = (): Stepper => {
    if (disposed) throw new Error("operation used after dispose()");
    raw.reset();
    return { step };
  };

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
      if (isAsyncIter(x)) return runAsyncStream([x], openStream, op);
      if (isSyncIter(x)) return runStream([x], openStream, op);
      const { flat, assets } = asRow(x);
      return event(flat, assets);
    } catch (e) { throw normalizeError(e); }
  };

  const op = call as ScreamerOp;
  op.reset = () => { if (disposed) throw new Error("operation used after dispose()"); raw.reset(); };
  op.dispose = free;
  (op as any)[Symbol.dispose] = free;
  (op as any).numInputs = width;
  (op as any).numOutputs = nOut;
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
//   fi iterables of rows, or one iterable of fi-row arrays -> an iterator of rows
//   (async iterables give an async iterator)
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

  // A stream event is fi rows of K values, one per source, or with one source
  // and fi > 1 an array of fi rows. A row is a Float64Array, a number[], or an
  // NdArray of shape [K] or [K, 1] (rule 3).
  const asRow = (row: any): ArrayLike<number> => {
    if (isTyped(row) || isNumArr(row)) return row;
    if (isNd(row) && (row.shape.length === 1 || (row.shape.length === 2 && row.shape[1] === 1))) return row.data;
    throw new TypeError(`each event of a row stream is one row of K values, got ${isNd(row) ? `shape [${row.shape}]` : typeof row}`);
  };
  const openStream = (n: number): Stepper => {
    if (disposed) throw new Error("operation used after dispose()");
    raw.reset();
    const unpack = n === 1 && fi > 1;
    const rows: ArrayLike<number>[] = new Array(fi);
    const step = (items: any[]) => {
      try {
        if (disposed) throw new Error("operation used after dispose()");
        const ev = unpack ? items[0] : items;
        if (unpack && (!Array.isArray(ev) || ev.length !== fi))
          throw new TypeError(`each event of one row stream is an array of ${fi} rows of K values`);
        for (let f = 0; f < fi; f++) rows[f] = asRow(ev[f]);
        return eventOut(rows);
      } catch (e) { throw normalizeError(e); }
    };
    return { step };
  };

  const call = (...args: any[]) => {
    try {
      if (disposed) throw new Error("operation used after dispose()");
      if (args.some(isNode)) throw new TypeError("row ops run on arrays and row streams, not inside a Pipeline");
      const kind = streamKind(args, fi);
      if (kind) return kind === "async" ? runAsyncStream(args, openStream, op) : runStream(args, openStream, op);
      if (args.length !== fi) throw new TypeError(`this op takes ${fi} inputs, got ${args.length}`);
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
  (op as any).numInputs = raw.nIn();
  (op as any).numOutputs = raw.nOut();
  REG.register(op, free);
  return op;
}
