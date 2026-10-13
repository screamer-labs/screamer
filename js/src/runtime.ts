import type { Screamer, RawOp } from "./loader.js";
import { normalizeError } from "./errors.js";
import { asNd, type NdArray } from "./ndarray.js";
import { Node, isNode } from "./node.js";
import {
  aKind, flattenRows, shapeText, throwBadShape, throwIfWrongKind, throwIfWrongKindEvent,
  throwNotANumber, whereText,
} from "./kinds.js";

// The op's class name for error messages, e.g. "RollingMean".
const opName = (raw: RawOp): string => (raw as any).constructor?.name ?? "this op";

// `Symbol.dispose` (explicit resource management) is provided by the
// "ESNext.Disposable" lib entry in tsconfig.json.
//
// What a wrapper owns in WASM, freed by `release()` when the wrapper is
// disposed or collected. REG holds it strongly until then, so it must never
// reach its wrapper: a held value that does keeps the wrapper alive for good,
// and every op ever built stays on both heaps. A closure made inside a wrapper
// factory shares the factory's scope, which holds the wrapper, so the release
// code lives in the classes below and not in such a closure.
export interface Owned { release(): void }
const REG = new FinalizationRegistry<Owned>((r) => r.release());

// Frees `res` when `owner` is collected, and returns the function that frees
// it now. `release()` must run once, so a finalizer after dispose() does
// nothing.
// There is no unregister token: V8 keeps the token table at its peak size, and
// a release that already ran costs nothing more than the small `res` kept
// until the owner is collected.
export function releaseWith(owner: object, res: Owned): () => void {
  REG.register(owner, res);
  return () => res.release();
}

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

// `args` with every null, at the top or inside plain Arrays (events, fields),
// read as NaN; null when there is none. Error paths only.
function nullsToNaN(args: any[]): any[] | null {
  const fix = (x: any, depth: number): any => {
    if (x === null) return NaN;
    if (depth > 0 && Array.isArray(x)) {
      let out: any[] | null = null;
      for (let i = 0; i < x.length; i++) {
        const v = x[i];
        if (v !== null && !Array.isArray(v)) continue;
        const y = fix(v, depth - 1);
        if (y !== v) (out ??= x.slice())[i] = y;
      }
      return out ?? x;
    }
    return x;
  };
  const out = fix(args, 4);
  return out === args ? null : out;
}
// A number, or null as NaN.
const numOrNull = (v: any) => typeof v === "number" || v === null;
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

// Every NdArray test goes through asNd, so a malformed array throws here.
const isNd = (x: any): x is NdArray => asNd(x) !== null;

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
    throw new RangeError(`this op takes ${n} inputs, got ${args.length} ${nAsync ? "async iterables" : "iterators"}`);
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
  let ended = -1, threw = -1, failed = false;
  try {
    for (;;) {
      for (let i = 0; i < its.length; i++) {
        let r;
        try { r = its[i].next(); } catch (e) { threw = i; throw e; }
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
      if (i === ended || i === threw) continue;
      try { its[i].return?.(); } catch (e) { if (!bad) { err = e; bad = true; } }
    }
    if (bad && !failed) throw err;
  }
}
async function* zipAsyncSources(srcs: AsyncIterable<any>[]): AsyncGenerator<any[], void, undefined> {
  const its = srcs.map((s) => s[Symbol.asyncIterator]());
  const items = new Array(its.length);
  let ended = -1, threw = -1, failed = false;
  try {
    for (;;) {
      for (let i = 0; i < its.length; i++) {
        let r;
        try { r = await its[i].next(); } catch (e) { threw = i; throw e; }
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
    const res = await Promise.allSettled(its.map((it, i) => (i === ended || i === threw ? undefined : it.return?.())));
    const bad = res.find((r): r is PromiseRejectedResult => r.status === "rejected");
    if (bad && !failed) throw bad.reason;
  }
}

// The generator a stream call returns. Nothing is read, and the stepper is not
// opened, before the first next(). An error in an event ends the stream and
// closes the sources. `owner` is the op wrapper: the generator holds it, so the
// finalizer cannot free a temporary op, such as `RollingMean(3)(src)`, while
// its stream runs.
//
// A stream that is started and then dropped never reaches its finally block,
// so STREAMS closes it when the generator is collected. The registry holds
// `held`, which holds the stepper's close function and never the generator.
// A finished stream clears `held.close`, so its finalizer does nothing (no
// unregister token, for the reason given at `releaseWith`).
type Held = { close?: () => void };
const STREAMS = new FinalizationRegistry<Held>((h) => h.close?.());

function* runSync(srcs: Iterable<any>[], open: (n: number) => Stepper, owner: unknown, held: Held) {
  void owner;
  const s = open(srcs.length);
  held.close = s.close;
  try {
    for (const items of zipSources(srcs)) yield s.step(items);
  } finally {
    held.close = undefined;
    s.close?.();
  }
}
async function* runAsync(srcs: AsyncIterable<any>[], open: (n: number) => Stepper, owner: unknown, held: Held) {
  void owner;
  const s = open(srcs.length);
  held.close = s.close;
  try {
    for await (const items of zipAsyncSources(srcs)) yield s.step(items);
  } finally {
    held.close = undefined;
    s.close?.();
  }
}
function runStream(srcs: Iterable<any>[], open: (n: number) => Stepper, owner: unknown) {
  const held: Held = {};
  const gen = runSync(srcs, open, owner, held);
  STREAMS.register(gen, held);
  return gen;
}
function runAsyncStream(srcs: AsyncIterable<any>[], open: (n: number) => Stepper, owner: unknown) {
  const held: Held = {};
  const gen = runAsync(srcs, open, owner, held);
  STREAMS.register(gen, held);
  return gen;
}

// The same for a stream generator made elsewhere (combineLatest's iterator
// forms): `held.close` frees what the stream holds, and the generator clears
// it when it finishes. Returns `gen`.
export type StreamHold = Held;
export function trackStream<G extends object>(gen: G, held: Held): G {
  STREAMS.register(gen, held);
  return gen;
}

// The copies an op's streams run on (decision D8: every stream is an
// independent run with its own state). A stream takes a reset copy with
// acquire() and gives it back with release(), so the next stream reuses it
// instead of building a new op; dispose() deletes every copy, in use or not.
// A raw op with no factory cannot be copied, so it cannot stream. The pool is
// part of what a finalizer frees, and a loop that never yields keeps every
// dropped op's share until it does, so the lists are made on first use.
class CopyPool {
  private idle: RawOp[] | null = null;
  private used: Set<RawOp> | null = null;
  constructor(private make: (() => RawOp) | null) {}
  check() {
    if (!this.make) throw new TypeError("a stream needs an op built by its factory, which can copy it for every stream and asset");
  }
  acquire(): RawOp {
    // A reused copy holds its last stream's state; a new one starts clean.
    const reused = this.idle?.pop();
    const c = reused ?? this.make!();
    if (reused) c.reset();
    (this.used ??= new Set()).add(c);
    return c;
  }
  release(c: RawOp) { if (this.used?.delete(c)) (this.idle ??= []).push(c); }
  dispose() {
    this.idle?.forEach((c) => c.delete());
    this.used?.forEach((c) => c.delete());
    this.idle = this.used = null;
  }
}

// One open stream of a wrapOp op: release() gives back what it holds.
type OpenStream = { release(): void };

// What wrapOp owns: the op, its event buffers, the stream copies, and the
// open streams (`live`, made by the first stream).
class OpRes implements Owned {
  private done = false;
  live: Set<OpenStream> | null = null;
  constructor(private M: Screamer, private raw: RawOp, private inBuf: number, private outBuf: number,
              private pool: CopyPool) {}
  release() {
    if (this.done) return;
    this.done = true;
    this.M.freeBuf(this.inBuf); this.M.freeBuf(this.outBuf);
    if (this.live) [...this.live].forEach((h) => h.release());
    this.pool.dispose();
    this.raw.delete();
  }
}

// `copy` builds the stream copies when `src` is a raw op whose factory lives
// elsewhere (wrapReducerOp passes its own). `finalize` false leaves freeing
// to a wrapper that owns this one and always disposes it (wrapReducerOp).
export function wrapOp(M: Screamer, src: OpSource, copy?: () => RawOp, finalize = true): ScreamerOp {
  const make = copy ?? (typeof src === "function" ? (src as () => RawOp) : null);
  const raw = resolve(src);
  const nIn = raw.nIn(), nOut = raw.nOut();
  const inBuf = M.allocF64(nIn), outBuf = M.allocF64(nOut);
  // The stream copies, described under Streams below.
  const pool = new CopyPool(make);
  const res = new OpRes(M, raw, inBuf, outBuf, pool);
  let disposed = false;
  const free = () => {
    if (disposed) return;
    disposed = true;
    res.release();
  };

  // One event on `o`: raw for a live call, a stream's own copy for a stream.
  const evalOn = (o: RawOp, inputs: ArrayLike<number>): number | number[] => {
    M.viewF64(inBuf, nIn).set(inputs as number[]);
    try { o.evalInto(inBuf, outBuf); } catch (e) { throw normalizeError(e); }
    const v = M.viewF64(outBuf, nOut);
    return nOut === 1 ? v[0] : Array.from(v);
  };
  const event = (inputs: ArrayLike<number>): number | number[] => {
    if (disposed) throw new Error("operation used after dispose()");
    return evalOn(raw, inputs);
  };

  // A batch call: the C++ shape module plans the broadcast once (it splits a
  // packed argument, consumes a size-1 field axis, broadcasts the assets) and
  // runs one eval_column per asset column. Each part is one argument, a
  // C-contiguous block of numbers with its shape. Returns the flat
  // (T, *A, nOut) result and the (T, *A) shape.
  type Part = { data: ArrayLike<number>; shape: number[] };
  function runShaped(parts: Part[]): { data: Float64Array; shape: number[] } {
    const ptrs: number[] = [];
    let outPtr = 0;
    try {
      const spec: number[] = [parts.length];
      let maxNd = 0;
      for (const p of parts) {
        const ptr = M.allocF64(Math.max(1, p.data.length));
        ptrs.push(ptr);
        M.viewF64(ptr, p.data.length).set(p.data as number[]);
        spec.push(ptr, p.shape.length, ...p.shape);
        maxNd = Math.max(maxNd, p.shape.length);
      }
      // The spec, then room for the result shape that C++ writes back.
      const specPtr = M.allocF64(spec.length + 1 + maxNd);
      ptrs.push(specPtr);
      M.viewF64(specPtr, spec.length).set(spec);
      const shapePtr = specPtr + 8 * spec.length;
      try { outPtr = M.evalShaped(raw, specPtr, shapePtr); } catch (e) { throw normalizeError(e); }
      const sv = M.viewF64(shapePtr, 1 + maxNd);
      const shape = Array.from(sv.subarray(1, 1 + sv[0]));
      const n = shape.reduce((p, d) => p * d, 1) * nOut;
      return { data: new Float64Array(M.viewF64(outPtr, n)), shape };
    } finally {
      ptrs.forEach((p) => M.freeBuf(p));
      if (outPtr) M.freeBuf(outPtr);
    }
  }

  function batch1(arr: ArrayLike<number>, typed: boolean): any {
    const rows = arr.length;
    const { data } = runShaped([{ data: arr, shape: [rows] }]);
    if (nOut !== 1) return { data, shape: [rows, nOut] } as NdArray;
    if (typed) return data;
    const out = new Array(rows);
    for (let i = 0; i < rows; i++) out[i] = data[i];
    return out;
  }

  // An N-D or broadcast call.
  function shaped(args: any[]): NdArray {
    const parts: Part[] = args.map((a) =>
      isNd(a) ? a : { data: a as ArrayLike<number>, shape: [(a as ArrayLike<number>).length] });
    const { data, shape } = runShaped(parts);
    return { data, shape: nOut === 1 ? shape : [...shape, nOut] };
  }

  // Rule 3: an Array of T events gives an Array of T results, each in the
  // container a stream gives for that event: a number (or nOut numbers) for
  // an event of numbers, a Float64Array for an event of assets, else an
  // NdArray. `res` is the batch result over the stacked events.
  function perEvent(res: Float64Array | NdArray, T: number): any[] {
    if (res instanceof Float64Array) return Array.from(res);
    const rest = res.shape.slice(1), w = rest.reduce((p, d) => p * d, 1);
    const out = new Array(T);
    for (let t = 0; t < T; t++) {
      const d = res.data.slice(t * w, (t + 1) * w);
      out[t] = rest.length === 0 ? d[0]
        : rest.length === 1 ? (nOut === 1 ? d : Array.from(d))
        : ({ data: d, shape: rest.slice() } as NdArray);
    }
    return out;
  }

  // Asset events stacked into one (T, ...event) NdArray. The first event is a
  // Float64Array or an NdArray and fixes the event shape; a later one may also
  // be a plain Array (nested for an N-D shape) of that shape, as in Python.
  // null in an Array is NaN. A wrong kind is a TypeError, another shape a
  // RangeError.
  function stackEvents(evs: any[]): NdArray {
    const shapeOf = (e: any): number[] | null => (isNd(e) ? e.shape : isTyped(e) ? [e.length] : null);
    const s0 = shapeOf(evs[0])!;
    const last = s0.length - 1;
    const per = s0.reduce((p, d) => p * d, 1), data = new Float64Array(per * evs.length);
    // Writes the nested Array x (dimension d of s0) at data[off...]; returns
    // the next offset, or -1 when x is not of the shape or holds a non-number.
    const fill = (x: any, d: number, off: number): number => {
      if (!(Array.isArray(x) || (d === last && isTyped(x))) || x.length !== s0[d]) return -1;
      if (d === last) {
        for (let i = 0; i < x.length; i++) {
          const v = x[i];
          if (typeof v === "number") data[off + i] = v;
          else if (v === null) data[off + i] = NaN;
          else return -1;
        }
        return off + x.length;
      }
      for (let i = 0; i < x.length && off >= 0; i++) off = fill(x[i], d + 1, off);
      return off;
    };
    for (let t = 0; t < evs.length; t++) {
      const e = evs[t], s = shapeOf(e);
      if (s) {
        if (s.length !== s0.length || s.some((d, j) => d !== s0[j]))
          throw new RangeError(`every event must have the same shape: the first was [${s0}], event ${t} is [${s}]`);
        data.set(isNd(e) ? e.data : e, t * per);
        continue;
      }
      if (fill(e, 0, t * per) >= 0) continue;
      const where = whereText("event", t);
      throwIfWrongKind(opName(raw), e, where);
      const st = typeof e === "number" || e === null ? "is a number" : shapeText(e) ? `has shape ${shapeText(e)}` : "has an irregular shape";
      throw new RangeError(`every event must have the same shape: the first was [${s0}], event ${t} ${st}`);
    }
    return { data, shape: [evs.length, ...s0] };
  }

  // ---- Streams -------------------------------------------------------------
  //
  // A stream reads one item per source per event: nIn sources of one field
  // each, or, with nIn > 1, one source whose events hold every field (an array
  // of nIn numbers, an array of nIn fields, or a (..., nIn) NdArray). The first
  // event fixes the event shape. Numbers give a number (or nOut numbers) per
  // event. Arrays hold one value per asset: the inputs broadcast like a batch
  // call (the C++ shape module plans the columns), and every asset column runs
  // its own copy from the pool, column 0 included.
  //
  // Every stream is an independent run (decision D8): it runs on copies of its
  // own, one per asset column, so two streams of one op never share state and
  // live calls on raw never touch a stream. Opening a stream (the call) resets
  // raw, like a batch call; the stream never uses raw afterwards. A finished
  // stream returns its copies to `pool` for the next stream; `live` holds the
  // release of every open stream, so dispose() frees what they hold.
  const EMPTY: number[] = [];

  const openStream = (n: number): Stepper => {
    if (disposed) throw new Error("operation used after dispose()");
    const UNKNOWN = 0, SCALAR = 1, ASSETS = 2;
    let mode = UNKNOWN;
    const unpack = n === 1 && nIn > 1;            // one source holds all nIn fields
    let plan: { cols: number; offset: number[]; shapes: number[][]; out: number[] | null } | null = null;
    let ops: RawOp[] = [];                        // the stream's op per column
    let colIn = 0, colOut = 0;                    // heap blocks of (cols x nIn) and (cols x nOut)
    // Per-event views of the current event's parts, reused across events.
    const partData: ArrayLike<number>[] = [], partShape: number[][] = [];
    const numBuf: Float64Array[] = [], lenShape: number[][] = [];
    let nEv = 0;                                  // events seen, for error messages

    const handle = {
      release() {
        if (!res.live?.delete(handle)) return;
        if (colIn) { M.freeBuf(colIn); M.freeBuf(colOut); colIn = colOut = 0; }
        ops.forEach((c) => pool.release(c));
        ops = [];
      },
    };
    (res.live ??= new Set()).add(handle);

    const scalarValues = (items: any[]): ArrayLike<number> | null => {
      if (unpack) {
        const v = items[0];
        if ((isTyped(v) || isNumArr(v)) && v.length === nIn) return v;
        if (isNd(v) && v.shape.length === 1 && v.shape[0] === nIn && v.data.length === nIn) return v.data;
        if (Array.isArray(v) && v.length === nIn && v.every(numOrNull)) return v.map((x) => x ?? NaN);
        return null;
      }
      for (let i = 0; i < items.length; i++)
        if (typeof items[i] !== "number")   // null is NaN (R-b), off the number path
          return items.every(numOrNull) ? items.map((x) => x ?? NaN) : null;
      return items;
    };

    // Reads the event's parts (one per field, or one packed part) into
    // partData / partShape and returns their count.
    const readParts = (items: any[]): number => {
      const src = unpack && Array.isArray(items[0]) ? items[0] : items;
      for (let k = 0; k < src.length; k++) {
        const v = src[k];
        if (typeof v === "number" || v === null) {
          const b = (numBuf[k] ??= new Float64Array(1));
          b[0] = v ?? NaN;                    // null is NaN (R-b)
          partData[k] = b; partShape[k] = EMPTY;
        } else if (isNd(v)) {
          partData[k] = v.data; partShape[k] = v.shape;
        } else if (isTyped(v) || isNumArr(v) || (Array.isArray(v) && v.every(numOrNull))) {
          const s = (lenShape[k] ??= [0]);
          s[0] = v.length;
          partData[k] = isTyped(v) || isNumArr(v) ? v : v.map((x: any) => x ?? NaN);
          partShape[k] = s;
        } else {
          throwIfWrongKindEvent(opName(raw), items, nEv - 1);
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
      let p: { shape: number[]; columns: number; offset: number[] };
      try { p = M.eventPlan(shapes.map((s) => [1, ...s]), nIn); } catch (e) { throw normalizeError(e); }
      const cols = p.columns;
      while (ops.length < cols) ops.push(pool.acquire());
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
        throw new RangeError(`every event must have the same shape: the first was ${fmtShapes(plan.shapes)}, this one is ${fmtShapes(partShape.slice(0, k))}`);
      const { cols, offset, out } = plan!;
      const packed = k === 1 && nIn > 1;
      const view = M.viewF64(colIn, cols * nIn);
      for (let i = 0; i < nIn; i++) {
        const d = partData[packed ? 0 : i], o = i * cols;
        for (let c = 0; c < cols; c++) view[c * nIn + i] = d[offset[o + c]];
      }
      try {
        for (let c = 0; c < cols; c++)
          ops[c].evalInto(colIn + 8 * c * nIn, colOut + 8 * c * nOut);
      } catch (e) { throw normalizeError(e); }
      const data = new Float64Array(M.viewF64(colOut, cols * nOut));
      return out ? { data, shape: out.slice() } : data;
    };

    // A later event of a number stream is not numbers: TypeError for a wrong
    // kind, else RangeError giving the shape of the first field that is not a
    // number (the Python binding raises the same).
    const scalarMismatch = (items: any[]): never => {
      const who = opName(raw), ev = nEv - 1;
      throwIfWrongKindEvent(who, items, ev);
      if (unpack) {
        const v = items[0], where = whereText("event", ev);
        if (!(Array.isArray(v) || isTyped(v)) || v.length !== nIn) throwBadShape(who, where, v, `${nIn} numbers`);
        for (let j = 0; j < nIn; j++)
          if (typeof v[j] !== "number") throwBadShape(who, `item [${j}] of ${where}`, v[j]);
      }
      for (let i = 0; i < items.length; i++)
        if (typeof items[i] !== "number")
          throwBadShape(who, whereText("event", ev, items.length > 1 ? i : -1), items[i]);
      throw new RangeError("every event must have the same shape: the first events were numbers, this one is not");
    };

    const step = (items: any[]) => {
      if (disposed) throw new Error("operation used after dispose()");
      nEv++;
      if (mode !== ASSETS) {
        const v = scalarValues(items);
        if (v) {
          if (mode === UNKNOWN) { ops.push(pool.acquire()); mode = SCALAR; }
          return evalOn(ops[0], v);
        }
        if (mode === SCALAR) scalarMismatch(items);
      }
      return assetEvent(items);
    };
    return { step, close: handle.release };
  };

  // A null value is a missing value, NaN (R-b). The number paths below never
  // see one; a call holding null fails its checks, and only then is it read
  // again with null as NaN, so numbers pay nothing.
  const call = (...args: any[]): any => {
    try {
      return dispatch(...args);
    } catch (e) {
      const fixed = nullsToNaN(args);
      if (fixed) return dispatch(...fixed);
      throw e;
    }
  };

  const dispatch = (...args: any[]): any => {
    if (disposed) throw new Error("operation used after dispose()");
    // Define-then-bind: if any argument is a symbolic Node, defer the call and
    // return a functor Node holding BOTH the raw C++ op (for `opPtr` at graph
    // compile) and this wrapper (so the Pipeline can pin its lifetime; the
    // Embind layer keeps only a non-owning EvalOp*).
    if (args.some(isNode)) {
      return new Node({ functor: raw, wrapper: op }, args.filter(isNode));
    }
    const kind = streamKind(args, nIn);
    if (kind) {
      pool.check();
      raw.reset();
      return kind === "async" ? runAsyncStream(args, openStream, op) : runStream(args, openStream, op);
    }
    if (nIn === 1) {
      const a = args[0];
      if (typeof a === "number") return event([a]);
      if (isNd(a)) return shaped([a]);
      if (isTyped(a)) return batch1(a, true);
      if (isNumArr(a)) return batch1(a, false);
      // An Array of asset events runs like a stream of them and gives an Array
      // of per-event results. The first event (a Float64Array or an NdArray)
      // makes it one; later events may be plain Arrays of its shape (A8).
      if (Array.isArray(a) && a.length > 0 && (isTyped(a[0]) || isNd(a[0])))
        return perEvent(shaped([stackEvents(a)]), a.length);
      if (Array.isArray(a)) {
        for (let i = 0; i < a.length; i++)
          if (typeof a[i] !== "number") throwNotANumber(opName(raw), a[i], whereText("event", i));
      }
      throw new TypeError(`${opName(raw)}: ${aKind(a)} is not a valid input; pass numbers, arrays, lists or iterators`);
    }
    // nIn > 1: N scalars -> one event; N arrays -> columnar batch.
    if (args.length === nIn && args.every((x) => typeof x === "number")) return event(args);
    if (args.length === 1 && isNd(args[0])) return shaped(args);      // one packed (T, ..., nIn) array
    if (args.length === nIn && args.every((x) => isTyped(x) || isNumArr(x) || isNd(x))) {
      if (args.some(isNd)) return shaped(args);
      const cols = args as ArrayLike<number>[];
      const rows = cols[0].length;
      if (cols.some((x) => x.length !== rows)) {
        throw new RangeError(`all input arrays must have the same length (got ${cols.map((a) => a.length).join(", ")})`);
      }
      // Each column goes to the heap as its own argument, one C++ pass.
      const { data } = runShaped(cols.map((c) => ({ data: c, shape: [rows] })));
      const out = nOut === 1 ? data : ({ data, shape: [rows, nOut] } as NdArray);
      // Rule 3: all number[] inputs give an Array, one result per event, as
      // Python lists give a list. A typed array among them gives an array (A8).
      return args.every((x) => Array.isArray(x)) ? perEvent(out, rows) : out;
    }
    if (args.length === 1 && Array.isArray(args[0])) {
      // A list is a sequence of events, and each event must hold every input.
      const evs = args[0] as any[];
      const last = (e: any) => (isNd(e) ? e.shape[e.shape.length - 1] : -1);
      for (let i = 0; i < evs.length; i++) {
        const e = evs[i];
        if (isNd(e) && last(e) === nIn) continue;
        const m = typeof e === "number" ? 1 : isNd(e) ? e.data.length : isTyped(e) || Array.isArray(e) ? e.length : -1;
        if (m < 0 || m === nIn) continue;
        const name = opName(raw);
        throw new RangeError(`${name} expects ${nIn} inputs; a list is read as a sequence of events, and event ${i} has ${m} value${m === 1 ? "" : "s"} (expected ${nIn}). To pass ${nIn} separate inputs, call ${name}()(${nIn > 2 ? "a, b, ..." : "a, b"}).`);
      }
      // Events of nIn values each read like the columns of a batch, and the
      // results come back one per event (rule 3).
      if (evs.length > 0 && evs.every((e) => (isTyped(e) || Array.isArray(e)) && e.length === nIn && Array.from(e as ArrayLike<any>).every((x) => typeof x === "number"))) {
        return perEvent(call(...Array.from({ length: nIn }, (_, k) => Float64Array.from(evs, (e) => e[k]))), evs.length);
      }
      // Events of shape (..., nIn) stack into one packed (T, ..., nIn) array.
      if (evs.length > 0 && evs.every(isNd)) return perEvent(shaped([stackEvents(evs)]), evs.length);
      // The first event that is not nIn numbers: a wrong kind is a TypeError,
      // a field of the wrong shape a RangeError.
      for (let i = 0; i < evs.length; i++) {
        const e = evs[i], where = whereText("event", i);
        if (!(Array.isArray(e) || isTyped(e))) { if (!isNd(e)) throwNotANumber(opName(raw), e, where); continue; }
        for (let j = 0; j < e.length; j++)
          if (typeof e[j] !== "number") {
            throwIfWrongKind(opName(raw), e, where);
            throwBadShape(opName(raw), `item [${j}] of ${where}`, e[j]);
          }
      }
    }
    if (args.length !== nIn) throw new RangeError(`expected ${nIn} inputs, got ${args.length}`);
    // N lists: the first element that is not a number, event by event as the
    // Python binding reads them.
    if (args.every((x) => Array.isArray(x) || isTyped(x))) {
      const rows = Math.min(...args.map((x) => x.length));
      for (let i = 0; i < rows; i++)
        for (let k = 0; k < nIn; k++)
          if (typeof args[k][i] !== "number") throwNotANumber(opName(raw), args[k][i], whereText("event", i, k));
    }
    throw new TypeError(`expected ${nIn} numeric inputs`);
  };

  const op = call as ScreamerOp;
  op.reset = () => { if (disposed) throw new Error("operation used after dispose()"); raw.reset(); };
  op.dispose = free;
  (op as any)[Symbol.dispose] = free;
  (op as any).numInputs = nIn;
  (op as any).numOutputs = nOut;
  if (finalize) releaseWith(op, res);
  return op;
}

// A nested batch is an array of rows, each row itself an array of assets.
// An array of (assets, width) NdArray events is one too.
const isNestedBatch = (x: any) =>
  Array.isArray(x) && x.length > 0 &&
  ((Array.isArray(x[0]) && (Array.isArray(x[0][0]) || isTyped(x[0][0]))) || (isNd(x[0]) && x[0].shape.length === 2));

// What wrapReducerOp owns: its event scratch, the stream copies, and the base
// wrapper, which owns the op.
class ReducerRes implements Owned {
  private done = false;
  rowBuf = 0; rowCap = 0; priceBuf = 0; priceCap = 0;
  constructor(private M: Screamer, private outBuf: number, private pool: CopyPool, private base: ScreamerOp) {}
  release() {
    if (this.done) return;
    this.done = true;
    if (this.rowBuf) this.M.freeBuf(this.rowBuf);
    if (this.priceBuf) this.M.freeBuf(this.priceBuf);
    this.M.freeBuf(this.outBuf);
    this.pool.dispose();
    this.base.dispose();
  }
}

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
  const make = typeof src === "function" ? (src as () => RawOp) : null;
  const raw = resolve(src);
  const width = raw.nIn(), nOut = raw.nOut();
  // The base has no finalizer of its own: ReducerRes disposes it.
  const base = wrapOp(M, raw, make ?? undefined, false);
  const pool = new CopyPool(make);             // the streams' own copies (D8)

  // Scratch for the per-event path (in `res`), grown on demand and reused
  // across events. Batch calls size their buffers exactly and release them, so
  // a single large batch does not pin memory for the lifetime of the op.
  const outBuf = M.allocF64(nOut);
  const res = new ReducerRes(M, outBuf, pool, base);
  let disposed = false;
  const free = () => {
    if (disposed) return;
    disposed = true;
    res.release();
  };

  // Normalize one event into a flat (assets, width) block: an NdArray, a flat
  // numeric array whose length is a multiple of width, or an array of rows.
  function asRow(x: any): { flat: ArrayLike<number>; assets: number } {
    if (isNd(x)) {
      const s = x.shape;
      if (s.length === 2) {
        if (s[1] !== width) {
          throw new RangeError(`expected an event of shape (assets, ${width}), got [${s}]`);
        }
        return { flat: x.data, assets: s[0] };
      }
      if (s.length === 1) return asRow(x.data);
      throw new RangeError(`expected a rank-1 or rank-2 event, got shape [${s}]`);
    }
    if (isTyped(x) || isNumArr(x)) {
      const n = (x as ArrayLike<number>).length;
      if (n === 0 || n % width !== 0) {
        throw new RangeError(`a flat event must hold a positive multiple of ${width} values, got ${n}`);
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
          throw new RangeError(`each asset must hold ${width} values, got ${row.length} at asset ${g}`);
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
  // With `prices` (one per asset), turnover is traded notional.
  const event = (flat: ArrayLike<number>, assets: number, o: RawOp = raw,
                 prices?: ArrayLike<number>): number | number[] => {
    const need = assets * width;
    if (need > res.rowCap) {
      if (res.rowBuf) M.freeBuf(res.rowBuf);
      res.rowBuf = M.allocF64(need);
      res.rowCap = need;
    }
    M.viewF64(res.rowBuf, need).set(flat as number[]);
    try {
      if (prices) {
        if (assets > res.priceCap) {
          if (res.priceBuf) M.freeBuf(res.priceBuf);
          res.priceBuf = M.allocF64(assets);
          res.priceCap = assets;
        }
        M.viewF64(res.priceBuf, assets).set(prices as number[]);
        o.reducePricedInto!(res.rowBuf, res.priceBuf, outBuf, assets);
      } else {
        o.reduceInto!(res.rowBuf, outBuf, assets);
      }
    } catch (e) { throw normalizeError(e); }
    const v = M.viewF64(outBuf, nOut);
    return nOut === 1 ? v[0] : Array.from(v);
  };

  const batch = (flat: ArrayLike<number>, rows: number, assets: number,
                 prices?: ArrayLike<number>): NdArray | Float64Array => {
    if (rows === 0) {
      return nOut === 1 ? new Float64Array(0) : { data: new Float64Array(0), shape: [0, nOut] };
    }
    const inN = rows * assets * width;
    const inP = M.allocF64(inN), outP = M.allocF64(rows * nOut);
    const priceP = prices ? M.allocF64(rows * assets) : 0;
    try {
      M.viewF64(inP, inN).set(flat as number[]);
      try {
        if (prices) {
          M.viewF64(priceP, rows * assets).set(prices as number[]);
          raw.reducePricedBatchInto!(inP, priceP, outP, rows, assets);
        } else {
          raw.reduceBatchInto!(inP, outP, rows, assets);
        }
      } catch (e) { throw normalizeError(e); }
      const data = new Float64Array(M.viewF64(outP, rows * nOut));
      return nOut === 1 ? data : { data, shape: [rows, nOut] };
    } finally {
      M.freeBuf(inP);
      M.freeBuf(outP);
      if (priceP) M.freeBuf(priceP);
    }
  };

  // A nested batch as one flat (rows, assets, width) block.
  const nestedFlat = (x: any[]): { flat: Float64Array; rows: number; assets: number } => {
    const rows = x.length;
    const first = asRow(x[0]);
    const assets = first.assets;
    const flat = new Float64Array(rows * assets * width);
    flat.set(first.flat as number[], 0);
    for (let r = 1; r < rows; r++) {
      const { flat: f, assets: g } = asRow(x[r]);
      if (g !== assets) {
        throw new RangeError(`every event must hold ${assets} assets, got ${g} at event ${r}`);
      }
      flat.set(f as number[], r * assets * width);
    }
    return { flat, rows, assets };
  };
  const nestedBatch = (x: any[]): NdArray | Float64Array => {
    const { flat, rows, assets } = nestedFlat(x);
    return batch(flat, rows, assets);
  };

  // Prices for one event, (assets,) values, or a batch of `rows` events,
  // (rows, assets): a Float64Array, a number[], an NdArray (a trailing
  // size-1 axis reads away, rule 1), or for a batch an array of such rows.
  function asPrices(x: any, rows: number | null, assets: number): ArrayLike<number> {
    const want = rows === null ? `[${assets}]` : `[${rows},${assets}]`;
    const bad = (got: string) =>
      new RangeError(`prices must have shape ${want}, one price per asset${rows === null ? "" : " and event"}, got ${got}`);
    if (isNd(x)) {
      let s = x.shape;
      if (s.length === (rows === null ? 2 : 3) && s[s.length - 1] === 1) s = s.slice(0, -1);
      const ok = rows === null ? s.length === 1 && s[0] === assets
        : s.length === 2 && s[0] === rows && s[1] === assets;
      if (!ok) throw bad(`[${x.shape}]`);
      return x.data;
    }
    if (rows === null) {
      if (!(isTyped(x) || isNumArr(x))) throw new TypeError("prices must be a numeric array of one price per asset");
      if (x.length !== assets) throw bad(`[${x.length}]`);
      return x;
    }
    if (!Array.isArray(x) || !x.every((r) => isTyped(r) || isNumArr(r)))
      throw new TypeError("batch prices must be a [T, K] NdArray or an array of numeric rows");
    if (x.length !== rows) throw bad(`${x.length} rows`);
    const flat = new Float64Array(rows * assets);
    x.forEach((r: ArrayLike<number>, i: number) => {
      if (r.length !== assets) throw bad(`${r.length} prices at row ${i}`);
      flat.set(r as number[], i * assets);
    });
    return flat;
  }

  // The number of values in an array, -1 for anything else. A nested array
  // counts its leaves.
  const count = (x: any): number => {
    if (typeof x === "number") return -1;
    if (Array.isArray(x)) {
      let n = 0;
      for (const v of x) { const c = count(v); n += c < 0 ? 1 : c; }
      return n;
    }
    if (isTyped(x)) return x.length;
    if (isNd(x)) return x.data.length;
    return -1;
  };
  // What a 2-item event of a one-source stream (or the first item of an
  // Array) holds: a (row, prices) pair has parts of 4 * K and K values, a row
  // given as two asset records has two parts of 4 values, anything else is
  // malformed.
  const pairKind = (x: any): "row" | "pair" | "bad" => {
    if (!Array.isArray(x) || x.length !== 2) return "row";
    const r = count(x[0]), p = count(x[1]);
    if (r < 0 || p < 0) return "row";                 // a flat row; read as one
    if (p >= 1 && r === width * p) return "pair";
    if (r === width && p === width) return "row";     // two asset records
    return "bad";
  };

  // One priced event on `o`.
  const pricedEvent = (row: any, prices: any, o: RawOp = raw) => {
    const r = asRow(row);
    return event(r.flat, r.assets, o, asPrices(prices, null, r.assets));
  };

  // PortfolioReport()(out, prices) on arrays: a batch or one event. An Array
  // of rows with an Array of price rows gives an Array of result rows; any
  // NdArray among them gives an NdArray (A8).
  const pricedCall = (x: any, p: any) => {
    if (isNd(x) && x.shape.length === 3) {
      const s = x.shape;
      if (s[2] !== width) {
        throw new RangeError(`expected a batch of shape (events, assets, ${width}), got [${s}]`);
      }
      return batch(x.data, s[0], s[1], asPrices(p, s[0], s[1]));
    }
    if (isNestedBatch(x)) {
      const { flat, rows, assets } = nestedFlat(x);
      const b = batch(flat, rows, assets, asPrices(p, rows, assets));
      if (isNd(p)) return b;
      if (b instanceof Float64Array) return Array.from(b);
      return Array.from({ length: rows }, (_, t) => b.data.slice(t * nOut, (t + 1) * nOut));
    }
    return pricedEvent(x, p);
  };

  // A stream of (assets, width) events, through the same iterators as wrapOp,
  // on a copy of its own (decision D8).
  // Two sources give rows and prices; one source gives rows, or (row, prices)
  // pairs. Whether a stream carries prices is fixed in C++ by its first row.
  const openStream = (n: number): Stepper => {
    if (disposed) throw new Error("operation used after dispose()");
    const own = pool.acquire();
    let nEv = 0;                                  // events seen, for error messages
    const step = (items: any[]) => {
      if (disposed) throw new Error("operation used after dispose()");
      const ev = nEv++;
      try {
        if (n === 2) return pricedEvent(items[0], items[1], own);
        const kind = pairKind(items[0]);
        if (kind === "bad") {
          throw new RangeError(`stream events are engine rows of shape (assets, ${width}) or [row, prices] ` +
            `pairs with ${width} * K and K values; got two parts of ${count(items[0][0])} and ${count(items[0][1])} values`);
        }
        if (kind === "pair") return pricedEvent(items[0][0], items[0][1], own);
        const row = asRow(items[0]);
        return event(row.flat, row.assets, own);
      } catch (e) { throwIfWrongKindEvent(opName(raw), items, ev); throw e; }
    };
    return { step, close: () => pool.release(own) };
  };
  const stream = (srcs: any[], async: boolean) => {
    pool.check();
    raw.reset();
    return async ? runAsyncStream(srcs, openStream, op) : runStream(srcs, openStream, op);
  };

  // null is NaN (R-b): read again only after the call failed (see wrapOp).
  const call = (...args: any[]): any => {
    try {
      return dispatch(...args);
    } catch (e) {
      const fixed = nullsToNaN(args);
      if (fixed) return dispatch(...fixed);
      throw e;
    }
  };

  const dispatch = (...args: any[]): any => {
    if (disposed) throw new Error("operation used after dispose()");
    // Anything that isn't a single reducer-shaped input -- N scalars, Node
    // composition, an arity error -- is the ordinary op contract.
    if (args.length === 2 && !args.some(isNode)) {
      const kind = streamKind(args, 2);
      if (kind) return stream(args, kind === "async");
      try { return pricedCall(args[0], args[1]); } catch (e) {
        args.forEach((a, i) => throwIfWrongKind(opName(raw), a, whereText("input", i)));
        throw e;
      }
    }
    if (args.length !== 1) return base(...args);
    const x = args[0];
    if (isNode(x) || typeof x === "number") return base(x);
    if (isNd(x)) {
      const s = x.shape;
      if (s.length === 3) {
        if (s[2] !== width) {
          throw new RangeError(`expected a batch of shape (events, assets, ${width}), got [${s}]`);
        }
        return batch(x.data, s[0], s[1]);
      }
      const { flat, assets } = asRow(x);
      return event(flat, assets);
    }
    if (isAsyncIter(x)) return stream([x], true);
    if (isSyncIter(x)) return stream([x], false);
    if (Array.isArray(x) && x.length > 0 && pairKind(x[0]) === "pair") {
      throw new TypeError("PortfolioReport does not read an Array of [row, prices] pairs; pass the rows " +
        "and the prices as two arguments, PortfolioReport()(out, prices), or stream the pairs " +
        "from an iterable, PortfolioReport()(pairs.values())");
    }
    let row: { flat: ArrayLike<number>; assets: number };
    try {
      if (isNestedBatch(x)) {
        // Rule 3: an Array of rows gives an Array of result rows.
        const b = nestedBatch(x);
        if (b instanceof Float64Array) return Array.from(b);
        return Array.from({ length: x.length }, (_, t) => b.data.slice(t * nOut, (t + 1) * nOut));
      }
      row = asRow(x);
    } catch (e) { throwIfWrongKind(opName(raw), x, ""); throw e; }
    return event(row.flat, row.assets);
  };

  const op = call as ScreamerOp;
  op.reset = () => { if (disposed) throw new Error("operation used after dispose()"); raw.reset(); };
  op.dispose = free;
  (op as any)[Symbol.dispose] = free;
  (op as any).numInputs = width;
  (op as any).numOutputs = nOut;
  releaseWith(op, res);
  return op;
}

// What wrapRowOp owns: the op, its event buffers, and the stream copies.
class RowRes implements Owned {
  private done = false;
  evIn = 0; evOut = 0; cap = 0;
  constructor(private M: Screamer, private raw: RawOp, private pool: CopyPool) {}
  release() {
    if (this.done) return;
    this.done = true;
    if (this.evIn) { this.M.freeBuf(this.evIn); this.M.freeBuf(this.evOut); }
    this.pool.dispose();
    this.raw.delete();
  }
}

// Wraps a row op: each event is one row of K assets with fieldsIn() values
// per asset (one value for a cross-sectional op; target and prices for a
// portfolio engine), and returns fieldsOut() values per asset. K is a
// dimension of the data, fixed by the first row after reset(), so it travels
// with every call. With one input per field:
//
//   (T, K) NdArrays  -> (T, K) or (T, K, fieldsOut) NdArray
//   arrays of rows   -> an array of result rows (Float64Array, or (K, fieldsOut) NdArray)
//   one row each (Float64Array/number[]) -> one row, or a (K, fieldsOut) NdArray
//   fi iterables of rows, or one iterable of fi-row arrays -> an iterator of rows
//   (async iterables give an async iterator)
//
// Mirrors the Python binding of screamer.RowOp.
export function wrapRowOp(M: Screamer, src: OpSource): ScreamerOp {
  const make = typeof src === "function" ? (src as () => RawOp) : null;
  const raw = resolve(src);
  const fi = raw.fieldsIn!(), fo = raw.fieldsOut!();
  const pool = new CopyPool(make);             // the streams' own copies (D8)
  const res = new RowRes(M, raw, pool);        // with the event buffers
  let disposed = false;
  const free = () => {
    if (disposed) return;
    disposed = true;
    res.release();
  };

  // One event from fi rows of k values each, on raw (a live call) or on a
  // stream's own copy.
  const event = (rows: ArrayLike<number>[], o: RawOp): Float64Array => {
    const k = rows[0].length;
    if (rows.some((r) => r.length !== k)) throw new RangeError("all inputs must hold the same number of assets");
    if (k > res.cap) {
      if (res.evIn) { M.freeBuf(res.evIn); M.freeBuf(res.evOut); }
      res.evIn = M.allocF64(fi * k); res.evOut = M.allocF64(fo * k); res.cap = k;
    }
    const view = M.viewF64(res.evIn, fi * k);
    rows.forEach((r, f) => view.set(r as number[], f * k));
    try { o.mapRowInto!(res.evIn, res.evOut, k); } catch (e) { throw normalizeError(e); }
    return new Float64Array(M.viewF64(res.evOut, fo * k));
  };

  // T events from fi (T, K) blocks, each flat and row-major.
  const batch = (blocks: ArrayLike<number>[], T: number, k: number): NdArray => {
    if (k === 0) throw new RangeError("a row op needs at least one asset");
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
      try { raw.mapRowsInto!(inP, outP, T, k); } catch (e) { throw normalizeError(e); }
      return { data: new Float64Array(M.viewF64(outP, nOut)), shape };
    } finally {
      M.freeBuf(inP); M.freeBuf(outP);
    }
  };

  // A (T, K) input as a flat block: an NdArray or an array of rows.
  const asBlock = (x: any): { flat: ArrayLike<number>; T: number; k: number } | null => {
    if (isNd(x)) {
      // Rule 1: a [T, K, 1] array reads as [T, K]; the data is already in that order.
      const unit = x.shape.length === 3 && x.shape[2] === 1;
      if (x.shape.length !== 2 && !unit) throw new RangeError(`expected a (T, K) array, got shape [${x.shape}]`);
      return { flat: x.data, T: x.shape[0], k: x.shape[1] };
    }
    if (Array.isArray(x) && x.length > 0 && x.every((r) => isTyped(r) || isNumArr(r))) {
      const { data, T, K } = flattenRows(x, opName(raw), "the input");
      return { flat: data, T, k: K };
    }
    return null;
  };

  const eventOut = (rows: ArrayLike<number>[], o: RawOp = raw): any => {
    const out = event(rows, o);
    if (fo !== 1) return { data: out, shape: [rows[0].length, fo] } as NdArray;
    return isTyped(rows[0]) ? out : Array.from(out);
  };

  // A stream event is fi rows of K values, one per source, or with one source
  // and fi > 1 an array of fi rows. A row is a Float64Array, a number[], or an
  // NdArray of shape [K] or [K, 1] (rule 1).
  const asRow = (row: any): ArrayLike<number> => {
    if (isTyped(row) || isNumArr(row)) return row;
    if (isNd(row) && (row.shape.length === 1 || (row.shape.length === 2 && row.shape[1] === 1))) return row.data;
    if (isNd(row)) throw new RangeError(`each event of a row stream is one row of K values, got shape [${row.shape}]`);
    if (Array.isArray(row) && row.every(numOrNull)) return row.map((x) => x ?? NaN);   // null is NaN
    throw new TypeError(`each event of a row stream is one row of K values, got ${typeof row}`);
  };
  // A stream runs on a copy of its own (decision D8); the call resets raw.
  const openStream = (n: number): Stepper => {
    if (disposed) throw new Error("operation used after dispose()");
    const own = pool.acquire();
    const unpack = n === 1 && fi > 1;
    const rows: ArrayLike<number>[] = new Array(fi);
    let nEv = 0;                                  // events seen, for error messages
    const step = (items: any[]) => {
      if (disposed) throw new Error("operation used after dispose()");
      const k = nEv++;
      try {
        const ev = unpack ? items[0] : items;
        if (unpack && (!Array.isArray(ev) || ev.length !== fi))
          throw new RangeError(`each event of one row stream is an array of ${fi} rows of K values`);
        for (let f = 0; f < fi; f++) rows[f] = asRow(ev[f]);
      } catch (e) { throwIfWrongKindEvent(opName(raw), items, k); throw e; }
      return eventOut(rows, own);
    };
    return { step, close: () => pool.release(own) };
  };

  // null is NaN (R-b): read again only after the call failed (see wrapOp).
  const call = (...args: any[]): any => {
    try {
      return dispatch(...args);
    } catch (e) {
      const fixed = nullsToNaN(args);
      if (fixed) return dispatch(...fixed);
      throw e;
    }
  };

  const dispatch = (...args: any[]): any => {
    if (disposed) throw new Error("operation used after dispose()");
    if (args.some(isNode)) throw new TypeError("row ops run on arrays and row streams, not inside a Pipeline");
    const kind = streamKind(args, fi);
    if (kind) {
      pool.check();
      raw.reset();
      return kind === "async" ? runAsyncStream(args, openStream, op) : runStream(args, openStream, op);
    }
    if (args.length !== fi) throw new RangeError(`this op takes ${fi} inputs, got ${args.length}`);
    if (args.every((a) => isTyped(a) || isNumArr(a) || (isNd(a) && a.shape.length === 1)))
      return eventOut(args.map((a) => (isNd(a) ? a.data : a)));
    const blocks = args.map(asBlock);
    if (blocks.some((b) => b === null)) {
      for (let i = 0; i < fi; i++) throwIfWrongKind(opName(raw), args[i], fi > 1 ? whereText("input", i) : "");
      throw new TypeError("expected (T, K) arrays, arrays of rows, or one row per input");
    }
    const b0 = blocks[0]!;
    if (blocks.some((b) => b!.T !== b0.T || b!.k !== b0.k))
      throw new RangeError("all inputs must have the same shape");
    const out = batch(blocks.map((b) => b!.flat), b0.T, b0.k);
    // Rule 3: an array of rows gives an array of result rows. An NdArray
    // keeps returning an NdArray, and so does a mix of the two. positionPortfolio
    // and walkForward in validation.ts rely on the mix giving an NdArray (an
    // NdArray target with an array-of-rows market).
    if (!args.every((a) => Array.isArray(a))) return out;
    const w = b0.k * fo;
    return Array.from({ length: b0.T }, (_, t) => {
      const row = out.data.slice(t * w, (t + 1) * w);
      return fo === 1 ? row : ({ data: row, shape: [b0.k, fo] } as NdArray);
    });
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
  releaseWith(op, res);
  return op;
}
