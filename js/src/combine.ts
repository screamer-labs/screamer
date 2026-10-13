// The direct forms of combineLatest: Python's combine_latest on arrays,
// {values, index} streams, iterators and async iterators. The layout join, the
// as-of join and the assembly of rows are C++ (include/screamer/streams/
// join_plan.h and include/screamer/dag/combine_stream.h, bound in
// wasm/join_bindings.cpp), shared with Python; this module reads the inputs
// into WASM buffers, pulls the iterators, and copies the rows out.
import { current } from "./index.js";
import { asNd, type NdArray } from "./ndarray.js";
import { normalizeError, wasmCall } from "./errors.js";
import { aKind, feedValue, flattenRows, kindName, throwWrongKind, toF64Checked } from "./kinds.js";
import { trackStream, type StreamHold } from "./runtime.js";

const WHO = "combineLatest";

/** A layout per input, or one for every input (null: the default reading). */
export type CombineLayout = string | Array<string | null>;

export interface DirectOptions { whenAll: boolean; layout: CombineLayout | null }

/** Values with time first: a typed array, an Array, T rows or an NdArray. */
export type CombineValues = Float64Array | ArrayLike<number | null> | NdArray | ArrayLike<ArrayLike<number | null>>;

/**
 * One input of the array form: values (paired by position), or an indexed
 * stream `{ values, index }`, the counterpart of Python's `(values, index)`
 * tuple. A plain Array is always values.
 */
export type CombineInput = CombineValues | { values: CombineValues; index?: ArrayLike<number> | null };

/** An event of an iterator: a value, or `{ value, index }` for an indexed source. */
export type CombineEvent = number | null | ArrayLike<number | null> | NdArray | { value: unknown; index: number };

// ---- reading values ---------------------------------------------------------

interface Block { data: Float64Array; shape: number[] }

const isView = (x: unknown) => ArrayBuffer.isView(x) && !(x instanceof DataView);
const isRows = (x: unknown) =>
  Array.isArray(x) && x.length > 0 && x[0] !== null && typeof x[0] === "object" && !(x[0] instanceof Date);

// Values with time first: an NdArray, T rows (each item read as feedValue reads
// it: null is NaN, a string a TypeError), or a flat array or typed array.
function block(x: unknown, where: string): Block {
  const nd = asNd(x);
  if (nd) return { data: nd.data, shape: nd.shape };
  if (isRows(x)) {
    const { data, T, K } = flattenRows(x as ArrayLike<unknown>, WHO, where);
    return { data, shape: [T, K] };
  }
  if (x instanceof BigInt64Array || x instanceof BigUint64Array)
    throw new TypeError(`${WHO}: ${where} is ${aKind(x)}; values must be numbers`);
  if (Array.isArray(x) || isView(x)) {
    const data = toF64Checked(x as ArrayLike<unknown>, WHO, (i) => `item ${i} of ${where}`, true);
    return { data, shape: [data.length] };
  }
  throwWrongKind(WHO, where, x, "an array, a typed array or an NdArray");
}

// An index that is not a number is a TypeError; the engine owns the message.
function badIndex(index: unknown): never {
  const M: any = current();
  wasmCall(() => M.throwIndexKind(kindName(index)));
  throw new TypeError("index must be a number");   // not reached
}

// A batch index: one number per row.
function indexArray(x: unknown, where: string): Float64Array {
  const nd = asNd(x);
  if (nd) {
    if (nd.shape.length !== 1) throw new RangeError(`${WHO}: ${where} has shape [${nd.shape}], expected one number per row`);
    return nd.data;
  }
  if (x instanceof Float64Array) return x;
  if (!Array.isArray(x) && !isView(x)) throwWrongKind(WHO, where, x, "an array of numbers");
  const a = x as ArrayLike<unknown>, out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    out[i] = typeof v === "number" ? v : badIndex(v);
  }
  return out;
}

// A plain object: the marker of an indexed stream ({values, index}) or event
// ({value, index}), never inferred from the items of an array.
const isPlainObject = (x: unknown): x is Record<string, unknown> => {
  if (x === null || typeof x !== "object" || Array.isArray(x) || isView(x) || asNd(x) !== null) return false;
  const proto = Object.getPrototypeOf(x);
  return proto === Object.prototype || proto === null;
};

function checkKeys(x: Record<string, unknown>, keys: string[], where: string): void {
  for (const key of Object.keys(x))
    if (!keys.includes(key))
      throw new TypeError(`${WHO}: ${where} has an unknown key '${key}'; expected { ${keys.join(", ")} }`);
}

// ---- the array form ---------------------------------------------------------

/** The join of arrays: `{ values, index }`, index null without an index. */
export function combineArrays(inputs: unknown[], o: DirectOptions): { values: NdArray; index: Float64Array | null } {
  const M: any = current();
  const blocks: Block[] = [], indexes: Array<Float64Array | null> = [];
  inputs.forEach((x, k) => {
    if (isPlainObject(x)) {
      checkKeys(x, ["values", "index"], `input ${k}`);
      if (!("values" in x)) throw new TypeError(`${WHO}: input ${k} is an object without values; expected { values, index }`);
      blocks.push(block(x.values, `the values of input ${k}`));
      indexes.push(x.index == null ? null : indexArray(x.index, `the index of input ${k}`));
    } else {
      blocks.push(block(x, `input ${k}`));
      indexes.push(null);
    }
  });
  const indexed = indexes.filter((i) => i !== null).length;
  if (indexed && indexed !== inputs.length)
    throw new RangeError(`${WHO}: cannot align positional and indexed streams; give every stream an index, or none`);
  const ptrs: number[] = [];
  const alloc = (n: number) => { const p = M.allocF64(Math.max(1, n)); ptrs.push(p); return p; };
  let out: any = null;
  try {
    // Every buffer first: an allocation may grow the heap, which replaces its views.
    const spec: number[] = [inputs.length];
    const data = blocks.map((b) => alloc(b.data.length));
    const index = indexes.map((i) => (i ? alloc(i.length) : 0));
    const specPtr = alloc(1 + blocks.reduce((n, b) => n + 4 + b.shape.length, 0));
    blocks.forEach((b, k) => {
      M.viewF64(data[k], b.data.length).set(b.data);
      const i = indexes[k];
      if (i) M.viewF64(index[k], i.length).set(i);
      spec.push(data[k], index[k], i ? i.length : 0, b.shape.length, ...b.shape);
    });
    M.viewF64(specPtr, spec.length).set(spec);
    out = wasmCall(() => M.combineLatestBatch(specPtr, o.whenAll, o.layout));
    const shape = out.shape as number[];
    const size = shape.reduce((a: number, b: number) => a * b, 1);
    const values = new Float64Array(M.viewF64(out.values, out.rows * size));
    const idx = new Float64Array(M.viewF64(out.index, out.rows));
    return { values: { data: values, shape: [out.rows, ...shape] }, index: indexed ? idx : null };
  } finally {
    for (const p of ptrs) M.freeBuf(p);
    if (out) { M.freeBuf(out.index); M.freeBuf(out.values); }
  }
}

// ---- the iterator forms -----------------------------------------------------

/** One row of an iterator join and its index (null without an index). */
export interface CombineRow { value: number | number[] | Float64Array | NdArray; index: number | null }

const EMPTY = Symbol("empty");

// One event as flat values and its shape: a number (null is NaN), an
// NdArray, a typed array, or an array of numbers or of rows.
function event(x: unknown, where: () => string): Block {
  if (x === null || typeof x !== "object") return { data: new Float64Array([feedValue(x, WHO, where)]), shape: [] };
  const nd = asNd(x);
  if (nd) return { data: nd.data, shape: nd.shape };
  if (isRows(x)) {
    const { data, T, K } = flattenRows(x as ArrayLike<unknown>, WHO, where());
    return { data, shape: [T, K] };
  }
  if (Array.isArray(x) || isView(x)) {
    const data = toF64Checked(x as ArrayLike<unknown>, WHO, (i) => `item ${i} of ${where()}`, true);
    return { data, shape: [data.length] };
  }
  throwWrongKind(WHO, where(), x, "a number, an array, an NdArray or a { value, index } event");
}

// The state of one iterator join, opened on the first event of each source:
// the C++ JoinStream, which owns one event buffer per source.
class Joiner {
  readonly indexed: boolean;
  private M: any = current();
  private js: any;
  private bufs: number[] = [];
  private widths: number[] = [];
  private wide = false;
  private rowSize = 0;
  private rowShape: number[] = [];
  private sizes = 0;
  private zipRow = -1;
  private count: number[];

  constructor(heads: unknown[], o: DirectOptions, held: StreamHold) {
    const kinds = new Set(heads.filter((h) => h !== EMPTY).map(isPlainObject));
    if (kinds.size > 1)
      throw new RangeError(`${WHO}: cannot mix positional (bare-value) and indexed ({ value, index }) sources; ` +
        "give every source an index, or none");
    this.indexed = kinds.has(true);
    this.count = heads.map(() => 0);
    const shapes = heads.map((h, i) =>
      h === EMPTY ? null : [1, ...event(this.indexed ? this.split(i, h).value : h, () => `event 0 of input ${i}`).shape]);
    this.js = wasmCall(() => new this.M.JoinStream(shapes, o.layout, this.indexed, o.whenAll));
    // Frees the stream's WASM state when the stream is collected unfinished;
    // the closure holds this Joiner, never the generator.
    held.close = () => this.close();
    const js = this.js;
    try {
      this.wide = js.wide();
      this.rowSize = js.rowSize();
      this.rowShape = js.rowShape();
      this.sizes = js.sizes();
      for (let i = 0; i < heads.length; i++) {
        this.widths.push(js.width(i));
        this.bufs.push(js.eventBuffer(i));
      }
    } catch (e) {
      this.close();
      throw e;
    }
  }

  close(): void {
    if (!this.js) return;
    this.js.delete();
    this.js = null;
  }

  // An event of an indexed source: { value, index } with a number index.
  private split(i: number, x: unknown): { value: unknown; index: number } {
    if (!isPlainObject(x) || !("value" in x))
      throwWrongKind(WHO, `event ${this.count[i]} of input ${i}`, x, "a { value, index } event");
    checkKeys(x, ["value", "index"], `event ${this.count[i]} of input ${i}`);
    const t = typeof x.index === "number" ? x.index : badIndex(x.index);
    return { value: x.value, index: t };
  }

  // Event x of input i into its buffer; returns its number of values. A
  // number is written as it is. A wrong count is passed on for C++ to
  // reject, without a write.
  private load(i: number, x: unknown): number {
    const k = this.count[i]++;
    const heap: Float64Array = this.M.HEAPF64;   // replaced when memory grows: read it here
    if (typeof x === "number") { heap[this.bufs[i] >> 3] = x; return 1; }
    const e = event(x, () => `event ${k} of input ${i}`);
    if (e.data.length === this.widths[i]) heap.set(e.data, this.bufs[i] >> 3);
    return e.data.length;
  }

  // A joined row of rowSize values at heap offset `p` (in doubles), as JavaScript.
  private rowAt(heap: Float64Array, p: number): CombineRow["value"] {
    if (!this.wide) {
      const w = this.rowSize, a = new Array<number>(w);
      for (let j = 0; j < w; j++) a[j] = heap[p + j];
      return a;
    }
    const data = heap.slice(p, p + this.rowSize);
    return this.rowShape.length > 1 ? { data, shape: this.rowShape } : data;
  }

  // Positional: one step, an item per source (EMPTY when it has ended).
  // Returns the joined row, or null when every source has ended; the C++
  // zip raises when only some have.
  zip(items: unknown[]): CombineRow | null {
    const s = this.sizes >> 3;
    for (let i = 0; i < items.length; i++) {
      const n = items[i] === EMPTY ? -1 : this.load(i, items[i]);
      this.M.HEAPF64[s + i] = n;   // read HEAPF64 after load: an event read may grow memory
    }
    try {
      if (!this.js.zip()) return null;
      // A positional join writes its row to one fixed place: read it once.
      if (this.zipRow < 0) this.zipRow = this.js.rowValues() >> 3;
      return { value: this.rowAt(this.M.HEAPF64, this.zipRow), index: null };
    } catch (e) { throw normalizeError(e); }
  }

  // Indexed: the next event of source i. With `run`, feeds the join and
  // returns its status (see take).
  offer(i: number, item: unknown, run: boolean): number {
    const { value, index } = this.split(i, item);
    const n = this.load(i, value);
    try { return this.js.offer(i, index, n, run); } catch (e) { throw normalizeError(e); }
  }

  // Indexed: source i has ended.
  end(i: number, run: boolean): number {
    try { return this.js.end(i, run); } catch (e) { throw normalizeError(e); }
  }

  start(): number {
    try { return this.js.start(); } catch (e) { throw normalizeError(e); }
  }

  // The rows a feed settled, and the source whose next event it wants (-1:
  // the stream is done), from its status rows * (n + 1) + (wanted + 1).
  take(status: number): { rows: CombineRow[]; want: number } {
    const n1 = this.count.length + 1;
    const m = Math.floor(status / n1);
    const rows: CombineRow[] = [];
    if (m) {
      const heap: Float64Array = this.M.HEAPF64;
      const ix = this.js.rowIndex() >> 3, vals = this.js.rowValues() >> 3, w = this.rowSize;
      for (let r = 0; r < m; r++) {
        const p = vals + r * w;
        // One input of one field: a number, as Python gives a float.
        const value = !this.wide && w === 1 ? heap[p] : this.rowAt(heap, p);
        rows.push({ value, index: heap[ix + r] });
      }
    }
    return { rows, want: (status % n1) - 1 };
  }
}

// The sync join. Nothing is read before the first next(); when the stream
// ends for any reason, every source not yet ended is closed with return().
function* runSync(srcs: Iterable<unknown>[], o: DirectOptions, held: StreamHold): Generator<CombineRow, void, undefined> {
  const its = srcs.map((s) => s[Symbol.iterator]());
  const ended = its.map(() => false);
  let j: Joiner | null = null;
  const pull = (i: number): unknown => {
    const r = its[i].next();
    if (r.done) { ended[i] = true; return EMPTY; }
    return r.value;
  };
  let failed = false;
  try {
    const heads = its.map((_, i) => pull(i));
    if (heads.every((h) => h === EMPTY)) return;
    j = new Joiner(heads, o, held);
    if (!j.indexed) {
      for (let items = heads; ; items = its.map((_, i) => (ended[i] ? EMPTY : pull(i)))) {
        const row = j.zip(items);
        if (!row) return;
        yield row;
      }
    }
    heads.forEach((h, i) => (h === EMPTY ? j!.end(i, false) : j!.offer(i, h, false)));
    for (let status = j.start(); ; ) {
      const { rows, want } = j.take(status);
      yield* rows;
      if (want < 0) return;
      const x = pull(want);
      status = x === EMPTY ? j.end(want, true) : j.offer(want, x, true);
    }
  } catch (e) {
    failed = true;
    throw e;
  } finally {
    held.close = undefined;
    j?.close();
    let err: unknown, bad = false;
    for (let i = 0; i < its.length; i++) {
      if (ended[i]) continue;
      try { its[i].return?.(); } catch (e) { if (!bad) { err = e; bad = true; } }
    }
    if (bad && !failed) throw err;
  }
}

// The async join, the same steps with every pull awaited.
async function* runAsync(srcs: AsyncIterable<unknown>[], o: DirectOptions, held: StreamHold): AsyncGenerator<CombineRow, void, undefined> {
  const its = srcs.map((s) => s[Symbol.asyncIterator]());
  const ended = its.map(() => false);
  let j: Joiner | null = null;
  const pull = async (i: number): Promise<unknown> => {
    const r = await its[i].next();
    if (r.done) { ended[i] = true; return EMPTY; }
    return r.value;
  };
  const pullAll = async () => {
    const out = [];
    for (let i = 0; i < its.length; i++) out.push(ended[i] ? EMPTY : await pull(i));
    return out;
  };
  let failed = false;
  try {
    const heads = await pullAll();
    if (heads.every((h) => h === EMPTY)) return;
    j = new Joiner(heads, o, held);
    if (!j.indexed) {
      for (let items = heads; ; items = await pullAll()) {
        const row = j.zip(items);
        if (!row) return;
        yield row;
      }
    }
    heads.forEach((h, i) => (h === EMPTY ? j!.end(i, false) : j!.offer(i, h, false)));
    for (let status = j.start(); ; ) {
      const { rows, want } = j.take(status);
      yield* rows;
      if (want < 0) return;
      const x = await pull(want);
      status = x === EMPTY ? j.end(want, true) : j.offer(want, x, true);
    }
  } catch (e) {
    failed = true;
    throw e;
  } finally {
    held.close = undefined;
    j?.close();
    // Close every source even when one return() rejects.
    const res = await Promise.allSettled(its.map((it, i) => (ended[i] ? undefined : it.return?.())));
    const bad = res.find((r): r is PromiseRejectedResult => r.status === "rejected");
    if (bad && !failed) throw bad.reason;
  }
}

/** The join of sync iterators, as a lazy iterator of rows. */
export function combineIterators(srcs: Iterable<unknown>[], o: DirectOptions): Generator<CombineRow, void, undefined> {
  const held: StreamHold = {};
  return trackStream(runSync(srcs, o, held), held);
}

/** The join of async iterables, as an async iterator of rows. */
export function combineAsync(srcs: AsyncIterable<unknown>[], o: DirectOptions): AsyncGenerator<CombineRow, void, undefined> {
  const held: StreamHold = {};
  return trackStream(runAsync(srcs, o, held), held);
}
