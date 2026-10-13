// merge: Python's merge on arrays, {values, index} streams, iterators and
// async iterators. The merge order (equal indexes keep the input order), the
// width rule and the index rules are C++ (include/screamer/streams/
// merge_rows.h, bound in wasm/merge_bindings.cpp), shared with Python; this
// module reads the inputs into WASM buffers, pulls the iterators, and copies
// the rows out.
import { current } from "./index.js";
import type { NdArray } from "./ndarray.js";
import { normalizeError, wasmCall } from "./errors.js";
import { feedValue, throwWrongKind } from "./kinds.js";
import { trackStream, type StreamHold } from "./runtime.js";
import { badIndex, block, checkKeys, event, indexArray, isPlainObject, named, type Block } from "./combine.js";

const WHO = "merge";

/** The merged rows of arrays: values, the input of each row, and the index (null without one). */
export interface MergeResult {
  values: Float64Array | NdArray;
  sources: Uint32Array;
  index: Float64Array | null;
}

/** One event of an iterator merge: its value, its index (null without one) and its input. */
export interface MergeEvent {
  value: number | Float64Array | NdArray;
  index: number | null;
  source: number;
}

// ---- the array form ---------------------------------------------------------

export function mergeArrays(inputs: unknown[]): MergeResult {
  const M: any = current();
  const blocks: Block[] = [], indexes: Array<Float64Array | null> = [];
  inputs.forEach((x, k) => {
    if (isPlainObject(x)) {
      checkKeys(x, ["values", "index"], `input ${k}`, WHO);
      if (!("values" in x)) throw new TypeError(`${WHO}: input ${k} is an object without values; expected { values, index }`);
      blocks.push(block(x.values, `the values of input ${k}`, WHO));
      indexes.push(x.index == null ? null : indexArray(x.index, `the index of input ${k}`, WHO));
    } else {
      blocks.push(block(x, `input ${k}`, WHO));
      indexes.push(null);
    }
  });
  const indexed = indexes.filter((i) => i !== null).length;
  if (indexed && indexed !== inputs.length)
    throw new RangeError(`${WHO}: cannot mix positional and indexed streams; give every stream an index, or none`);
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
    try { out = M.mergeBatch(specPtr); } catch (e) { throw named(normalizeError(e), WHO); }
    const rows = out.rows as number, shape = out.shape as number[];
    const size = shape.reduce((a: number, b: number) => a * b, 1);
    const data2 = new Float64Array(M.viewF64(out.values, rows * size));
    const idx = new Float64Array(M.viewF64(out.index, rows));
    const words = M.viewF64(out.sources, Math.ceil(rows / 2)) as Float64Array;
    const sources = new Uint32Array(words.buffer, words.byteOffset, rows).slice();
    return {
      values: shape.length ? { data: data2, shape: [rows, ...shape] } : data2,
      sources,
      index: indexed ? idx : null,
    };
  } finally {
    for (const p of ptrs) M.freeBuf(p);
    if (out) { M.freeBuf(out.index); M.freeBuf(out.values); M.freeBuf(out.sources); }
  }
}

// ---- the iterator forms -----------------------------------------------------

const EVENT_KEYS = ["value", "index"];

// The C++ merge order of one iterator merge (wasm/merge_bindings.cpp
// MergeStream) and each source's pending event. Every step is one call to a
// plain export, which returns the next step: k + 1 to pull source k, -(k + 1)
// to emit the pending event of source k, 0 when every source has ended.
class Merger {
  private M: any = current();
  private who: string;
  private ms: any;
  private addr = 0;
  private shape = 0;   // the heap offset (in doubles) of the event-shape scratch
  private indexed: boolean | null = null;
  private count: number[];
  private value: Array<MergeEvent["value"]>;
  private index: number[];

  // `who` names the caller in messages (a lazy Pipeline merges its feeds).
  // `sameWidth` false: no width rule across the inputs (a lazy Pipeline).
  constructor(n: number, held: StreamHold, who = WHO, sameWidth = true) {
    this.who = who;
    this.ms = wasmCall(() => new this.M.MergeStream(n, sameWidth));
    // Frees the merge's WASM state when the stream is collected unfinished.
    held.close = () => this.close();
    try {
      this.addr = this.ms.address();
      this.shape = this.ms.shapeBuffer() >> 3;
    } catch (e) {
      this.close();
      throw e;
    }
    this.count = new Array(n).fill(0);
    this.value = new Array(n);
    this.index = new Array(n).fill(0);
  }

  close(): void {
    if (!this.ms) return;
    this.ms.delete();
    this.ms = null;
    this.addr = 0;
  }

  private live(): number {
    if (!this.addr) throw new Error(`${this.who}: the stream was closed`);
    return this.addr;
  }

  start(): number {
    try { return this.ms.start(); } catch (e) { throw named(normalizeError(e), this.who); }
  }

  // The event of source k (the step was -(k + 1)).
  event(k: number): MergeEvent {
    return { value: this.value[k], index: this.indexed ? this.index[k] : null, source: k };
  }

  end(i: number): number {
    try { return this.M._screamer_merge_end(this.live(), i); } catch (e) { throw named(normalizeError(e), this.who); }
  }

  // Event x of source i: a value (positional), or { value, index }.
  offer(i: number, x: unknown): number {
    const k = this.count[i]++;
    let v: unknown = x, t = k, marked = false;
    if (typeof x === "object" && x !== null && isPlainObject(x)) {
      marked = true;
      const e = x as Record<string, unknown>;
      if (!("value" in e)) throwWrongKind(this.who, `event ${k} of input ${i}`, x, "a { value, index } event");
      checkKeys(e, EVENT_KEYS, () => `event ${k} of input ${i}`, this.who);
      t = typeof e.index === "number" ? e.index : badIndex(e.index, this.who);
      v = e.value;
    }
    if (this.indexed === null) this.indexed = marked;
    else if (marked !== this.indexed)
      throw new RangeError(`${this.who}: cannot mix positional (bare-value) and indexed ({ value, index }) sources; ` +
        "give every source an index, or none");
    let ndim = 0;
    if (typeof v === "number") {
      this.value[i] = v;
    } else if (v === null || typeof v !== "object") {
      this.value[i] = feedValue(v, this.who, () => `event ${k} of input ${i}`);
    } else {
      const b = event(v, () => `event ${k} of input ${i}`, this.who);
      const shape = b.shape;
      this.value[i] = shape.length === 1 ? b.data.slice() : { data: b.data.slice(), shape: shape.slice() };
      ndim = shape.length;
      const heap: Float64Array = this.M.HEAPF64;   // read after the event: a read may grow memory
      for (let d = 0; d < ndim && d < 32; d++) heap[this.shape + d] = shape[d];
    }
    this.index[i] = t;
    try { return this.M._screamer_merge_offer(this.live(), i, t, ndim); } catch (e) { throw named(normalizeError(e), this.who); }
  }
}

function* runSync(srcs: Iterable<unknown>[], held: StreamHold, who = WHO, sameWidth = true): Generator<MergeEvent, void, undefined> {
  const its = srcs.map((s) => s[Symbol.iterator]());
  const ended = its.map(() => false);
  let m: Merger | null = null;
  let failed = false;
  try {
    m = new Merger(its.length, held, who, sameWidth);
    for (let step = m.start(); step !== 0;) {
      if (step > 0) {
        const i = step - 1, r = its[i].next();
        if (r.done) { ended[i] = true; step = m.end(i); } else step = m.offer(i, r.value);
      } else {
        const k = -step - 1;
        yield m.event(k);
        const r = its[k].next();   // source k owes its next event
        if (r.done) { ended[k] = true; step = m.end(k); } else step = m.offer(k, r.value);
      }
    }
  } catch (e) {
    failed = true;
    throw e;
  } finally {
    held.close = undefined;
    m?.close();
    let err: unknown, bad = false;
    for (let i = 0; i < its.length; i++) {
      if (ended[i]) continue;
      try { its[i].return?.(); } catch (e) { if (!bad) { err = e; bad = true; } }
    }
    if (bad && !failed) throw err;
  }
}

async function* runAsync(srcs: AsyncIterable<unknown>[], held: StreamHold): AsyncGenerator<MergeEvent, void, undefined> {
  const its = srcs.map((s) => s[Symbol.asyncIterator]());
  const ended = its.map(() => false);
  let m: Merger | null = null;
  let failed = false;
  try {
    m = new Merger(its.length, held);
    for (let step = m.start(); step !== 0;) {
      let i: number;
      if (step > 0) {
        i = step - 1;
      } else {
        i = -step - 1;
        yield m.event(i);           // then source i owes its next event
      }
      const r = await its[i].next();
      if (r.done) { ended[i] = true; step = m.end(i); } else step = m.offer(i, r.value);
    }
  } catch (e) {
    failed = true;
    throw e;
  } finally {
    held.close = undefined;
    m?.close();
    const res = await Promise.allSettled(its.map((it, i) => (ended[i] ? undefined : it.return?.())));
    const bad = res.find((r): r is PromiseRejectedResult => r.status === "rejected");
    if (bad && !failed) throw bad.reason;
  }
}

/** The merge of sync iterators, as a lazy iterator of events. */
export function mergeIterators(srcs: Iterable<unknown>[]): Generator<MergeEvent, void, undefined> {
  const held: StreamHold = {};
  return trackStream(runSync(srcs, held), held);
}

/** The index-order merge of a lazy Pipeline: `who` names it in messages, and
 * the inputs may have widths of their own (the engine checks each input). */
export function mergeEvents(srcs: Iterable<unknown>[], who: string): Generator<MergeEvent, void, undefined> {
  const held: StreamHold = {};
  return trackStream(runSync(srcs, held, who, false), held);
}

/** The merge of async iterables, as an async iterator of events. */
export function mergeAsync(srcs: AsyncIterable<unknown>[]): AsyncGenerator<MergeEvent, void, undefined> {
  const held: StreamHold = {};
  return trackStream(runAsync(srcs, held), held);
}
