import type { Screamer } from "./loader.js";
import { asNd, type NdArray } from "./ndarray.js";
import { Node, isNode } from "./node.js";
import { current } from "./index.js";
import { normalizeError, wasmCall } from "./errors.js";
import { aKind, feedValue, kindName, toF64Checked, whereText } from "./kinds.js";
import { releaseWith, type Owned } from "./runtime.js";

// One bound feed for an Input: a values stream and its integer index axis.
type Feed = Float64Array | number[] | NdArray | { values: Float64Array | number[] | NdArray; index?: Float64Array | number[] | null };
type Feeds = Feed | Record<string, Feed>;

export interface Output {
  values: Float64Array | NdArray;
  index: Float64Array;
}

/**
 * Options of a Pipeline.
 *
 * `alignOutputs` (default `true`): with more than one output, put every output
 * on one index, the sorted union of the output indexes with one row per
 * distinct index. Each output holds its latest value at every row, and rows
 * start once every output has a value. The outputs then share one `index`.
 * With `false`, each output keeps its own index and length.
 *
 * `maxPending` (default 1,000,000): the most events a join may hold while it
 * waits for another stream: each `combineLatest` node without its own
 * `maxPending`, each `filter` node, and the join of a cumulative `resample`
 * with its driver. More throws `RangeError`. An integer from 1 to 2**53.
 */
export interface PipelineOptions {
  alignOutputs?: boolean;
  maxPending?: number;
}

// maxPending as the engine's count (an integer from 1 to 2**53); the engine
// owns the rule and its messages (max_pending.h).
// `who` names the op and setting for the message ("Pipeline: maxPending").
export function maxPendingCount(M: Screamer, v: unknown, who = "Pipeline: maxPending"): number {
  if (typeof v !== "number") wasmCall(() => (M as any).throwMaxPendingKind(kindName(v), who));
  return wasmCall(() => (M as any).checkedMaxPending(v, who));
}

// A resample's maxFill under the same rule (resample_params.h); unset, the
// engine's default.
function maxFillCount(M: Screamer, v: unknown): number {
  if (v === undefined) return (M as any).defaultMaxFill();
  if (typeof v !== "number") wasmCall(() => (M as any).throwMaxFillKind(kindName(v), "resample: maxFill"));
  return wasmCall(() => (M as any).checkedMaxFill(v, "resample: maxFill"));
}

// Event-by-event streaming driver returned by `pipeline.live()`. Push feeds one
// event at a time, flush at end of input, then drain the result.
export interface LiveDriver {
  push(name: string, index: number, value: number | ArrayLike<number> | NdArray): void;
  pushWide(name: string, index: number, values: ArrayLike<number> | NdArray): void;
  advance(now: number): void;
  flush(): void;
  result(): Output | Output[];
}

// The options of a Pipeline, checked: a plain object of known keys.
function pipelineOptions(M: Screamer, opts: unknown): { alignOutputs: boolean; maxPending: number } {
  const maxPendingDefault = (M as any).defaultMaxPending() as number;
  if (opts === undefined || opts === null) return { alignOutputs: true, maxPending: maxPendingDefault };
  const proto = typeof opts === "object" ? Object.getPrototypeOf(opts) : undefined;
  if (proto !== Object.prototype && proto !== null)
    throw new TypeError(`Pipeline: the options are ${aKind(opts)}, expected an object such as { alignOutputs: false }`);
  for (const key of Object.keys(opts as object))
    if (key !== "alignOutputs" && key !== "maxPending") throw new TypeError(`Pipeline: unknown option '${key}'`);
  const a = (opts as PipelineOptions).alignOutputs;
  if (a !== undefined && typeof a !== "boolean")
    throw new TypeError(`Pipeline: alignOutputs must be a boolean, got ${aKind(a)}`);
  const m = (opts as PipelineOptions).maxPending;
  return { alignOutputs: a ?? true, maxPending: m === undefined ? maxPendingDefault : maxPendingCount(M, m) };
}

// A push value that is one wide event: an Array, a typed array, or an NdArray.
function isSequence(v: object): boolean {
  if (Array.isArray(v)) return true;
  if (ArrayBuffer.isView(v)) return !(v instanceof DataView);
  const a = v as any;
  return a.data instanceof Float64Array && Array.isArray(a.shape);
}

// An index that is not a number is a TypeError; the engine owns the message
// (checked_index.h). A number goes to the engine's finite/whole/int64 rule.
function badIndex(M: Screamer, index: unknown): never {
  wasmCall(() => (M as any).throwIndexKind(kindName(index)));
  throw new TypeError("index must be a number");   // not reached
}

// A batch index as a Float64Array of numbers; the engine checks each value in
// its int64 cast loop.
function indexArray(M: Screamer, x: Float64Array | ArrayLike<unknown>): Float64Array {
  if (x instanceof Float64Array) return x;
  const n = x.length, out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const v = x[i];
    out[i] = typeof v === "number" ? v : badIndex(M, v);
  }
  return out;
}

// Collect the Input nodes reachable from `outputs` (ports _reachable_inputs).
function reachableInputs(outputs: Node[]): Node[] {
  const seen = new Set<Node>();
  const stack = [...outputs];
  const inputs: Node[] = [];
  while (stack.length) {
    const node = stack.pop()!;
    if (seen.has(node)) continue;
    seen.add(node);
    if ((node.op as any)?.input !== undefined) inputs.push(node);
    else stack.push(...node.inputs);
  }
  return inputs;
}

// A functor instance may back at most one node (ports _check_stateful_safety):
// graph nodes are stateful, so sharing one op across nodes would share state.
function checkStatefulSafety(outputs: Node[]): void {
  const seen = new Set<Node>();
  const used = new Set<unknown>();
  const stack = [...outputs];
  while (stack.length) {
    const node = stack.pop()!;
    if (seen.has(node)) continue;
    seen.add(node);
    const raw = (node.op as any)?.functor;
    if (raw !== undefined) {
      if (used.has(raw)) {
        throw new Error(
          "the same functor instance backs two nodes; construct a fresh " +
            "functor per node (state cannot be shared)",
        );
      }
      used.add(raw);
    }
    stack.push(...node.inputs);
  }
}

// Write a JS number[] into a fresh heap uint32[]; return its byte pointer. A
// Uint32Array is layered over viewF64's WASM-memory buffer (HEAPU32 is not an
// exported module property), fetched fresh so a prior alloc's growth is seen.
function heapU32(M: Screamer, arr: number[]): number {
  const doubles = Math.ceil((arr.length * 4) / 8) || 1;
  const ptr = (M as any).allocF64(doubles);
  const dv = (M as any).viewF64(ptr, doubles) as Float64Array;
  new Uint32Array(dv.buffer, dv.byteOffset, arr.length).set(arr);
  return ptr;
}

/**
 * A reusable N-in / M-out function you define once and call on stored data.
 * Build a symbolic graph from `Input(...)` nodes and op factories, then pass the
 * outputs here. Calling the pipeline binds feeds and runs the compiled graph.
 */
class PipelineImpl implements Owned {
  readonly inputs: Node[];
  readonly outputs: Node[];
  private M: Screamer;
  private names: string[];
  private cg: any = null;
  // The graph's address, for the plain WASM exports of the per-event pushes
  // (wasm/dag_bindings.cpp); 0 once released.
  private cgAddr = 0;
  // Retained op wrappers: the Embind graph holds only non-owning EvalOp*s, so
  // the wrappers (which own the C++ ops) must stay alive for the graph lifetime.
  private ops: any[] = [];
  private _disposed = false;
  readonly alignOutputs: boolean;
  readonly maxPending: number;

  constructor(inputs: Node[], outputs: Node[], opts?: unknown) {
    this.inputs = [...inputs];
    this.outputs = [...outputs];
    this.M = current();
    const o = pipelineOptions(this.M, opts);
    this.alignOutputs = o.alignOutputs;
    this.maxPending = o.maxPending;

    for (const n of this.inputs) {
      if (!isNode(n) || (n.op as any)?.input === undefined) {
        throw new Error("every entry in inputs must be an Input(...) node");
      }
    }
    for (const n of this.outputs) {
      if (!isNode(n)) throw new Error("every entry in outputs must be a Node");
    }

    const reachable = reachableInputs(this.outputs);
    const reachableSet = new Set(reachable);
    const declaredSet = new Set(this.inputs);
    const undeclared = reachable.filter((n) => !declaredSet.has(n));
    if (undeclared.length) {
      const names = undeclared.map((n) => (n.op as any).input);
      throw new Error(`outputs reference undeclared inputs: ${JSON.stringify(names)}`);
    }
    const unused = this.inputs.filter((n) => !reachableSet.has(n));
    if (unused.length) {
      const names = unused.map((n) => (n.op as any).input);
      throw new Error(`declared inputs are unused by any output: ${JSON.stringify(names)}`);
    }
    checkStatefulSafety(this.outputs);

    this.names = this.inputs.map((n) => (n.op as any).input as string);
    this.compile();
  }

  private compile(): void {
    const M: any = this.M;
    const gb = new M.GraphBuilder();
    const ids = new Map<Node, number>();
    wasmCall(() => gb.setMaxPending(this.maxPending));

    const build = (node: Node): number => {
      const cached = ids.get(node);
      if (cached !== undefined) return cached;
      const op = node.op as any;
      let nid: number;
      if (op?.input !== undefined) {
        nid = wasmCall(() => gb.addInput());
      } else if (op?.functor !== undefined) {
        const inp = node.inputs.map(build);
        const vec = new M.VectorSizeT();
        for (const id of inp) vec.push_back(id);
        try { nid = gb.addFunctor(M.opPtr(op.functor), vec); } catch (e) { throw normalizeError(e); } finally { vec.delete(); }
        this.ops.push(op.wrapper); // retain for the graph's lifetime
      } else if (op?.combinator !== undefined) {
        nid = this.buildCombinator(gb, op, node.inputs.map(build));
      } else {
        throw new Error("combinator/operator nodes are not supported yet");
      }
      ids.set(node, nid);
      return nid;
    };

    try {
      // Build Input nodes first so their C++ ids follow signature order.
      for (const n of this.inputs) build(n);
      const outIds = this.outputs.map(build);
      const outVec = new M.VectorSizeT();
      for (const id of outIds) outVec.push_back(id);
      try { gb.setOutputs(outVec); } catch (e) { throw normalizeError(e); } finally { outVec.delete(); }
      this.cg = wasmCall(() => gb.compile());
      this.cgAddr = this.cg.address();
    } finally {
      gb.delete();
    }
  }

  // Dispatch an operator (combinator) node onto the matching GraphBuilder.add*
  // call. `inp` are the already-built C++ ids of the node's inputs. Mirrors the
  // operator branch of dag.py Pipeline._compile_cpp.
  private buildCombinator(gb: any, op: any, inp: number[]): number {
    const M: any = this.M;
    const vec = new M.VectorSizeT();
    for (const id of inp) vec.push_back(id);
    try {
      const kind = op.combinator as string;
      const p = op.params ?? {};
      switch (kind) {
        case "combineLatest": {
          // A node's own maxPending, else the Pipeline's.
          const cap = p.maxPending === undefined ? this.maxPending : maxPendingCount(M, p.maxPending, "combineLatest: maxPending");
          return wasmCall(() => gb.addCombineLatest(vec, !!p.whenAll, cap));
        }
        case "dropna":
          return wasmCall(() => gb.addDropna(vec, !!p.howAll));
        case "filter":
          return wasmCall(() => gb.addFilter(vec));
        case "delay":
          return wasmCall(() => gb.addDelay(vec, p.duration));
        case "select": {
          const cols = new M.VectorSizeT();
          for (const c of p.columns) cols.push_back(c);
          try {
            return wasmCall(() => gb.addSelect(vec, cols));
          } finally {
            cols.delete();
          }
        }
        case "resample":
          return this.buildResample(gb, vec, p);
        case "forecastPairs":
          // The C++ graph definition (dag/forecast_pairs.h), inputs [X, y].
          return wasmCall(() => gb.addForecastPairs(
            inp[0], inp[1], p.count !== undefined, p.count ?? 0,
            p.duration !== undefined, p.duration ?? 0, p.dropna));
        default:
          throw new Error(`unknown combinator '${kind}'`);
      }
    } finally {
      vec.delete();
    }
  }

  // Build a resample node: marshal the fixed-plan entries into a flat int32 heap
  // buffer ([agg0, col0, agg1, col1, ...]) and resolve a functor reducer's
  // EvalOp*, retaining its wrapper for the graph's lifetime.
  private buildResample(gb: any, vec: any, p: any): number {
    const M: any = this.M;
    const maxFill = maxFillCount(M, p.maxFill);   // checked before any allocation
    let planPtr = 0;
    let planLen = 0;
    let reducerPtr = 0;
    if (p.plan && p.plan.length) {
      const flat: number[] = [];
      for (const [agg, col] of p.plan) flat.push(agg, col);
      planLen = p.plan.length; // entry (pair) count; C++ reads 2*planLen ints
      planPtr = heapU32(M, flat);
    }
    if (p.reducer) {
      const rop = p.reducer.op;
      reducerPtr = wasmCall(() => M.opPtr(rop.functor));
      this.ops.push(rop.wrapper); // retain reducer op for the graph's lifetime
    }
    try {
      return wasmCall(() => gb.addResample(
        vec, p.modeCode, p.aggCode, p.labelCode, p.fillCode,
        p.width, p.origin, p.count, p.threshold, p.maxAge,
        planPtr, planLen, reducerPtr, this.maxPending, maxFill,
      ));
    } finally {
      if (planPtr) M.freeBuf(planPtr);
    }
  }

  private normalizeFeeds(feeds: Feeds): Array<{ values: Float64Array; index: Float64Array; rows: number; width: number }> {
    // An array or a typed array (not a DataView) holds values element by element.
    const isFlat = (v: unknown): boolean =>
      Array.isArray(v) || (ArrayBuffer.isView(v) && !(v instanceof DataView));
    // A flat array or an NdArray is a bare values feed.
    const isValues = (v: unknown): boolean => isFlat(v) || asNd(v) !== null;
    // A {values, index} object. A typed array's or Map's values() method is not one.
    const isPair = (v: unknown): boolean =>
      v !== null && typeof v === "object" && !(v instanceof Map) && "values" in (v as any) &&
      typeof (v as any).values !== "function";
    const expected = "expected an array of numbers, a typed array, an NdArray or {values, index}";
    // A non-number value or index raises TypeError naming its event and input.
    const one = (feed: Feed, k: number): { values: Float64Array; index: Float64Array; rows: number; width: number } => {
      let values: Float64Array;
      let index: Float64Array | undefined;
      const at = (i: number) => whereText("event", i, k);
      // Rows of the feed: the values' length, or an NdArray's first dimension;
      // and an NdArray's width, its second dimension, which an empty NdArray
      // keeps (a flat array has width 1).
      let rows: number;
      let ndWidth: number | undefined;
      const read = (v: unknown): Float64Array => {
        const nd = asNd(v);   // a malformed NdArray raises RangeError
        if (nd) {
          if (nd.shape.length < 1 || nd.shape.length > 2)
            throw new RangeError(`Pipeline: the values of input ${k} have shape [${nd.shape}], expected (T,) or (T, w)`);
          rows = nd.shape[0];
          ndWidth = nd.shape.length === 2 ? nd.shape[1] : 1;
          return nd.data;
        }
        if (v instanceof BigInt64Array || v instanceof BigUint64Array)
          throw new TypeError(`Pipeline: the values of input ${k} are ${aKind(v)}; values must be numbers (Float64Array.from(x, Number) converts them)`);
        if (isFlat(v)) {
          const out = toF64Checked(v as ArrayLike<unknown>, "Pipeline", at, true);
          rows = out.length;
          return out;
        }
        throw new TypeError(`Pipeline: the values of input ${k} are ${aKind(v)}, expected an array, a typed array or an NdArray`);
      };
      if (isValues(feed)) {
        values = read(feed);
      } else if (isPair(feed)) {
        values = read((feed as any).values);
        if ((feed as any).index != null) index = indexArray(this.M, (feed as any).index);
      } else {
        throw new TypeError(`Pipeline: the feed for input '${this.names[k]}' is ${aKind(feed)}, ${expected}`);
      }
      if (!index) {
        index = new Float64Array(rows!);
        for (let i = 0; i < rows!; i++) index[i] = i;
      }
      // A flat array is one value per row; rows of several values come as an
      // NdArray. The engine checks that the index has one value per row.
      return { values, index, rows: rows!, width: ndWidth ?? 1 };
    };

    if (isValues(feeds) || isPair(feeds)) {
      if (this.inputs.length !== 1) {
        throw new Error(`this pipeline has ${this.inputs.length} inputs; pass an object keyed by input name`);
      }
      return [one(feeds as Feed, 0)];
    }
    // Feeds by name are a plain object; anything else (a Map, a Set, a DataView,
    // a generator, a class instance) is a feed of the wrong kind.
    const proto = feeds !== null && typeof feeds === "object" ? Object.getPrototypeOf(feeds) : undefined;
    if (proto !== Object.prototype && proto !== null)
      throw new TypeError(`Pipeline: the feeds are ${aKind(feeds)}, ${expected}, or an object keyed by input name`);
    const obj = feeds as Record<string, Feed>;
    return this.names.map((name, k) => {
      if (!(name in obj)) {
        if ("values" in obj)   // {values() {...}}: not values, not feeds by name
          throw new TypeError(`Pipeline: the feeds have a values ${typeof (obj as any).values}, expected an array of numbers, a typed array or an NdArray`);
        throw new Error(`missing feed for input '${name}'`);
      }
      return one(obj[name], k);
    });
  }

  call(feeds: Feeds): Output | Output[] {
    if (this._disposed) throw new Error("pipeline used after dispose()");
    const M: any = this.M;
    const norm = this.normalizeFeeds(feeds);
    const nIn = norm.length;

    const buffers: number[] = [];
    const idxBufs: number[] = [];
    const valBufs: number[] = [];
    const lens: number[] = [];
    const rowCounts: number[] = [];
    const widths: number[] = [];
    for (const { values, index, rows, width } of norm) {
      const len = index.length;
      const idxBuf = M.allocF64(len || 1);
      M.viewF64(idxBuf, len).set(index);
      const valBuf = M.allocF64(values.length || 1);
      M.viewF64(valBuf, values.length).set(values);
      buffers.push(idxBuf, valBuf);
      idxBufs.push(idxBuf);
      valBufs.push(valBuf);
      lens.push(len);
      rowCounts.push(rows);
      widths.push(width);
    }
    const idxPtrs = heapU32(M, idxBufs);
    const valPtrs = heapU32(M, valBufs);
    const lensPtr = heapU32(M, lens);
    const rowsPtr = heapU32(M, rowCounts);
    const widthsPtr = heapU32(M, widths);
    buffers.push(idxPtrs, valPtrs, lensPtr, rowsPtr, widthsPtr);

    try {
      // Run the whole graph ONCE, then read each cached output. Running per
      // output would recompute the graph M times (O(M x) redundant compute).
      // Everything in this block is a call into WASM or a copy off its heap.
      this.cg.runBatchStore(idxPtrs, valPtrs, lensPtr, rowsPtr, widthsPtr, nIn, this.alignOutputs);
      return this.readStored();
    } catch (e) {
      throw normalizeError(e);
    } finally {
      for (const b of buffers) M.freeBuf(b);
    }
  }

  // Read the outputs the graph stored (runBatchStore or drainStore), then free
  // them. One output is one Output; aligned outputs share one index array.
  private readStored(): Output | Output[] {
    const M: any = this.M;
    try {
      const n = this.outputs.length;
      if (n === 1) return this.marshalOut(this.cg.output(0));
      const results: Output[] = [];
      if (this.cg.storedAligned()) {
        // The rows stay in the graph until clearStored(): copy them.
        const ix = this.cg.alignedIndex();
        const index = new Float64Array(M.viewF64(ix.indexPtr, ix.rows));
        for (let o = 0; o < n; o++) results.push({ values: this.copyValues(this.cg.alignedOutput(o)), index });
      } else {
        for (let o = 0; o < n; o++) results.push(this.marshalOut(this.cg.output(o)));
      }
      return results;
    } finally {
      this.cg.clearStored();
    }
  }

  // ---- live streaming --------------------------------------------------
  // Resolve an input name to its C++ input index. compile() adds Input nodes
  // first, in signature order, so the input index equals the name's position.
  private inputIndex(name: string): number {
    const i = this.names.indexOf(name);
    if (i < 0) throw new Error(`unknown input '${name}'`);
    return i;
  }

  // A copy of one stored output (its rows stay in the graph until clearStored()).
  private marshalOut(out: any): Output {
    const M: any = this.M;
    const idxCopy = new Float64Array(M.viewF64(out.indexPtr, out.rows));
    return { values: this.copyValues(out), index: idxCopy };
  }

  // A copy of the values of one output: a Float64Array for one column, else an
  // NdArray of shape [rows, width].
  private copyValues(out: any): Float64Array | NdArray {
    const M: any = this.M;
    const rows = out.rows as number;
    const width = out.width as number;
    const valCopy = new Float64Array(M.viewF64(out.valuePtr, rows * width));
    return width === 1 ? valCopy : ({ data: valCopy, shape: [rows, width] } as NdArray);
  }

  // Event-by-event driver over the compiled graph. Push events (scalar or wide)
  // per input, optionally advance the watermark, flush at end of input, then
  // result() drains the accumulated output. Streaming this way is bit-identical
  // to the batch call() on the same data (the batch==stream invariant).
  // `owner` is the callable Pipeline: the driver holds it, so the finalizer
  // cannot free the graph while the driver is in use.
  live(owner: Pipeline): LiveDriver {
    this.checkLive();
    const M: any = this.M;
    const self = this;
    // Start each live session from a clean slate: run_batch resets on entry but
    // leaves outputs_ populated on exit, and a prior live session leaves node
    // state, so reset here so the stream starts independent of any prior call.
    this.cg.reset();
    const pushWide = (i: number, index: number, values: ArrayLike<number> | NdArray): void => {
      const nd = asNd(values);   // a malformed NdArray raises RangeError
      let src: ArrayLike<number>;
      if (nd) {
        if (nd.shape.length > 1) throw new RangeError(`a wide event is a 1-D sequence of numbers, got shape [${nd.shape}]`);
        src = nd.data;
      } else src = values as ArrayLike<number>;
      const t = typeof index === "number" ? index : badIndex(M, index);
      if (src == null || typeof src.length !== "number")
        throw new TypeError(`a wide event is a sequence of numbers, got ${typeof values}`);
      const width = src.length;
      const buf = M.allocF64(width || 1);   // the engine rejects an empty event
      try {
        const view = M.viewF64(buf, width);
        if (src instanceof Float64Array) view.set(src);
        else for (let j = 0; j < width; j++) {
          const x: unknown = src[j];
          if (x !== null && typeof x === "object" && (Array.isArray(x) || ArrayBuffer.isView(x) || asNd(x)))
            throw new RangeError("a wide event is a 1-D sequence of numbers, item [" + j + "] is a sequence");
          view[j] = typeof x === "number" ? x
            : feedValue(x, "Pipeline", () => `item [${j}] of the values pushed to input ${i}`);
        }
        const g = self.graphAddress();   // after the values: their conversion may dispose
        try { M._screamer_push_event_wide(g, i, t, buf, width); } catch (e) { throw normalizeError(e); }
      } finally {
        M.freeBuf(buf);
      }
    };
    const driver: LiveDriver & { [OWNER]: Pipeline } = {
      [OWNER]: owner,
      push(name: string, index: number, value: number | ArrayLike<number> | NdArray): void {
        self.checkLive();
        const i = self.inputIndex(name);
        // A number is the hot path. A sequence is one wide event.
        if (value !== null && typeof value === "object" && isSequence(value)) {
          pushWide(i, index, value);
          return;
        }
        // null is a missing value (NaN); another primitive passes as before
        // (ToNumber at the WASM call); an object that is not a number is a
        // wrong kind.
        const v = typeof value === "number" ? value : feedValue(value, "Pipeline", () => `the value pushed to input ${i}`);
        const t = typeof index === "number" ? index : badIndex(M, index);
        const g = self.graphAddress();   // after the value: its conversion may dispose
        try { M._screamer_push_event(g, i, t, v); } catch (e) { throw normalizeError(e); }
      },
      pushWide(name: string, index: number, values: ArrayLike<number> | NdArray): void {
        self.checkLive();
        pushWide(self.inputIndex(name), index, values);
      },
      advance(now: number): void {
        self.checkLive();
        const t = typeof now === "number" ? now : badIndex(M, now);
        try { self.cg.advance(t); } catch (e) { throw normalizeError(e); }
      },
      flush(): void {
        self.checkLive();
        try { self.cg.flush(); } catch (e) { throw normalizeError(e); }
      },
      result(): Output | Output[] {
        self.checkLive();
        // Drains every output at once, aligned as the batch call aligns them.
        try {
          self.cg.drainStore(self.alignOutputs);
          return self.readStored();
        } catch (e) {
          throw normalizeError(e);
        }
      },
    };
    return driver;
  }

  // The graph's address for a per-event export. Read after the event's values
  // are converted: a value's valueOf may dispose the Pipeline, and address 0
  // would reach C++ as an object. A released graph raises the TypeError a call
  // on the released Embind object raised.
  private graphAddress(): number {
    if (!this.cgAddr) throw new TypeError("pipeline used after dispose()");
    return this.cgAddr;
  }

  // Raises once the Pipeline is disposed; every live driver call checks it.
  private checkLive(): void {
    if (this._disposed) throw new Error("pipeline used after dispose()");
  }

  // Called once, by dispose() on the Pipeline or when it is collected. It
  // never reaches the callable Pipeline, so it can be the registry's held value.
  release(): void {
    if (this._disposed) return;
    this._disposed = true;
    if (this.cg) {
      this.cg.delete();
      this.cg = null;
      this.cgAddr = 0;
    }
    // Release our retention; the wrappers' FinalizationRegistry (or an explicit
    // user dispose) frees the underlying C++ ops.
    this.ops = [];
  }
}

// The key under which a live driver holds its Pipeline.
const OWNER = Symbol("pipeline");

// A Pipeline is invoked like a function: `p(feeds)`. It also carries `.dispose()`
// and its node metadata. `new Pipeline(inputs, outputs)` returns this callable.
export interface Pipeline {
  (feeds: Feeds): Output | Output[];
  live(): LiveDriver;
  dispose(): void;
  readonly inputs: Node[];
  readonly outputs: Node[];
  readonly alignOutputs: boolean;
  readonly maxPending: number;
}

// Construct a callable Pipeline. Works with or without `new` (the constructor
// returns the callable wrapper, so `new Pipeline(...)` yields the same thing).
export const Pipeline = function (inputs: Node[], outputs: Node[], opts?: PipelineOptions): Pipeline {
  const impl = new PipelineImpl(inputs, outputs, opts);
  // One argument: the feed, or the feeds keyed by input name. A second
  // positional argument (an index, a second input) is a TypeError.
  const fn = (function (this: unknown, feeds: Feeds) {
    // Trailing undefined arguments are not given (p(x, undefined) is p(x)).
    let n = arguments.length;
    while (n > 1 && arguments[n - 1] === undefined) n--;
    if (n > 1)
      throw new TypeError(`Pipeline: takes one argument, the feeds, got ${n}; ` +
        "pass an index as {values, index} and several inputs as an object keyed by input name");
    return impl.call(feeds);
  }) as Pipeline;
  // The compiled graph is freed by dispose() or when `fn` is collected.
  (fn as any).dispose = releaseWith(fn, impl);
  (fn as any).live = () => impl.live(fn);
  Object.defineProperty(fn, "inputs", { value: impl.inputs, enumerable: true });
  Object.defineProperty(fn, "outputs", { value: impl.outputs, enumerable: true });
  Object.defineProperty(fn, "alignOutputs", { value: impl.alignOutputs, enumerable: true });
  Object.defineProperty(fn, "maxPending", { value: impl.maxPending, enumerable: true });
  return fn;
} as unknown as { new (inputs: Node[], outputs: Node[], opts?: PipelineOptions): Pipeline };
