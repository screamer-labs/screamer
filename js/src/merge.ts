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
import { badIndex, block, checkKeys, event, indexArray, isPlainObject, type Block } from "./combine.js";

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
    out = wasmCall(() => M.mergeBatch(specPtr));
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

// The C++ merge order of one iterator merge and each source's pending event.
class Merger {
  private M: any = current();
  private ms: any;
  private indexed: boolean | null = null;
  private count: number[];
  private value: Array<MergeEvent["value"]>;
  private index: number[];

  constructor(n: number, held: StreamHold) {
    this.ms = wasmCall(() => new this.M.MergeStream(n));
    this.count = new Array(n).fill(0);
    this.value = new Array(n);
    this.index = new Array(n).fill(0);
    // Frees the merge's WASM state when the stream is collected unfinished.
    held.close = () => this.close();
  }

  close(): void {
    if (!this.ms) return;
    this.ms.delete();
    this.ms = null;
  }

  private live(): any {
    if (!this.ms) throw new Error(`${WHO}: the stream was closed`);
    return this.ms;
  }

  want(): number { return this.live().want(); }

  pop(): MergeEvent | null {
    const i = this.live().pop();
    if (i < 0) return null;
    return { value: this.value[i], index: this.indexed ? this.index[i] : null, source: i };
  }

  end(i: number): void { this.live().end(i); }

  // Event x of source i: a value (positional), or { value, index }.
  offer(i: number, x: unknown): void {
    const k = this.count[i]++;
    const where = () => `event ${k} of input ${i}`;
    const marked = isPlainObject(x);
    if (this.indexed === null) this.indexed = marked;
    else if (marked !== this.indexed)
      throw new RangeError(`${WHO}: cannot mix positional (bare-value) and indexed ({ value, index }) sources; ` +
        "give every source an index, or none");
    let v: unknown = x, t = k;
    if (marked) {
      const e = x as Record<string, unknown>;
      if (!("value" in e)) throwWrongKind(WHO, where(), x, "a { value, index } event");
      checkKeys(e, EVENT_KEYS, where, WHO);
      t = typeof e.index === "number" ? e.index : badIndex(e.index);
      v = e.value;
    }
    let shape: number[];
    if (typeof v === "number" || v === null || typeof v !== "object") {
      this.value[i] = feedValue(v, WHO, where);
      shape = [];
    } else {
      const b = event(v, where, WHO);
      shape = b.shape;
      this.value[i] = shape.length === 1 ? b.data.slice() : { data: b.data.slice(), shape: shape.slice() };
    }
    this.index[i] = t;
    try { this.live().offerEvent(i, t, shape); } catch (e) { throw normalizeError(e); }
  }
}

function* runSync(srcs: Iterable<unknown>[], held: StreamHold): Generator<MergeEvent, void, undefined> {
  const its = srcs.map((s) => s[Symbol.iterator]());
  const ended = its.map(() => false);
  let m: Merger | null = null;
  let failed = false;
  try {
    m = new Merger(its.length, held);
    for (;;) {
      for (let i: number; (i = m.want()) >= 0;) {
        const r = its[i].next();
        if (r.done) { ended[i] = true; m.end(i); } else m.offer(i, r.value);
      }
      const e = m.pop();
      if (!e) return;
      yield e;
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
    for (;;) {
      for (let i: number; (i = m.want()) >= 0;) {
        const r = await its[i].next();
        if (r.done) { ended[i] = true; m.end(i); } else m.offer(i, r.value);
      }
      const e = m.pop();
      if (!e) return;
      yield e;
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

/** The merge of async iterables, as an async iterator of events. */
export function mergeAsync(srcs: AsyncIterable<unknown>[]): AsyncGenerator<MergeEvent, void, undefined> {
  const held: StreamHold = {};
  return trackStream(runAsync(srcs, held), held);
}
