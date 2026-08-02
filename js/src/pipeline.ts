import type { Screamer } from "./loader.js";
import type { NdArray } from "./ndarray.js";
import { Node, isNode } from "./node.js";
import { current } from "./index.js";
import { normalizeError } from "./errors.js";

// One bound feed for an Input: a values stream and its integer index axis.
type Feed = Float64Array | number[] | { values: Float64Array | number[]; index?: Float64Array | number[] };
type Feeds = Feed | Record<string, Feed>;

export interface Output {
  values: Float64Array | NdArray;
  index: Float64Array;
}

// Event-by-event streaming driver returned by `pipeline.live()`. Push feeds one
// event at a time, flush at end of input, then drain the result.
export interface LiveDriver {
  push(name: string, index: number, value: number): void;
  pushWide(name: string, index: number, values: ArrayLike<number>): void;
  advance(now: number): void;
  flush(): void;
  result(): Output;
}

function toF64(x: Float64Array | number[]): Float64Array {
  return x instanceof Float64Array ? x : Float64Array.from(x);
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
class PipelineImpl {
  readonly inputs: Node[];
  readonly outputs: Node[];
  private M: Screamer;
  private names: string[];
  private cg: any = null;
  // Retained op wrappers: the Embind graph holds only non-owning EvalOp*s, so
  // the wrappers (which own the C++ ops) must stay alive for the graph lifetime.
  private ops: any[] = [];
  private _disposed = false;

  constructor(inputs: Node[], outputs: Node[], _opts?: unknown) {
    this.inputs = [...inputs];
    this.outputs = [...outputs];
    this.M = current();

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

    const build = (node: Node): number => {
      const cached = ids.get(node);
      if (cached !== undefined) return cached;
      const op = node.op as any;
      let nid: number;
      if (op?.input !== undefined) {
        nid = gb.addInput();
      } else if (op?.functor !== undefined) {
        const inp = node.inputs.map(build);
        const vec = new M.VectorSizeT();
        for (const id of inp) vec.push_back(id);
        nid = gb.addFunctor(M.opPtr(op.functor), vec);
        vec.delete();
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
      gb.setOutputs(outVec);
      outVec.delete();
      this.cg = gb.compile();
    } catch (e) {
      throw normalizeError(e);
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
        case "combineLatest":
          return gb.addCombineLatest(vec, !!p.whenAll, p.maxPending);
        case "dropna":
          return gb.addDropna(vec, !!p.howAll);
        case "filter":
          return gb.addFilter(vec);
        case "delay":
          return gb.addDelay(vec, p.duration);
        case "select": {
          const cols = new M.VectorSizeT();
          for (const c of p.columns) cols.push_back(c);
          try {
            return gb.addSelect(vec, cols);
          } finally {
            cols.delete();
          }
        }
        case "resample":
          return this.buildResample(gb, vec, p);
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
      reducerPtr = M.opPtr(rop.functor);
      this.ops.push(rop.wrapper); // retain reducer op for the graph's lifetime
    }
    try {
      return gb.addResample(
        vec, p.modeCode, p.aggCode, p.labelCode, p.fillCode,
        p.width, p.origin, p.count, p.threshold, p.maxAge,
        planPtr, planLen, reducerPtr,
      );
    } finally {
      if (planPtr) M.freeBuf(planPtr);
    }
  }

  private normalizeFeeds(feeds: Feeds): Array<{ values: Float64Array; index: Float64Array }> {
    const one = (feed: Feed): { values: Float64Array; index: Float64Array } => {
      let values: Float64Array;
      let index: Float64Array | undefined;
      if (feed instanceof Float64Array || Array.isArray(feed)) {
        values = toF64(feed);
      } else {
        values = toF64(feed.values);
        if (feed.index !== undefined) index = toF64(feed.index);
      }
      if (!index) {
        index = new Float64Array(values.length);
        for (let i = 0; i < values.length; i++) index[i] = i;
      }
      return { values, index };
    };

    const bare = feeds instanceof Float64Array || Array.isArray(feeds) ||
      (feeds != null && typeof feeds === "object" && "values" in (feeds as any));
    if (bare) {
      if (this.inputs.length !== 1) {
        throw new Error(`this pipeline has ${this.inputs.length} inputs; pass an object keyed by input name`);
      }
      return [one(feeds as Feed)];
    }
    const obj = feeds as Record<string, Feed>;
    return this.names.map((name) => {
      if (!(name in obj)) throw new Error(`missing feed for input '${name}'`);
      return one(obj[name]);
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
    const widths: number[] = [];
    for (const { values, index } of norm) {
      const len = index.length;
      const width = len ? values.length / len : 1;
      const idxBuf = M.allocF64(len || 1);
      M.viewF64(idxBuf, len).set(index);
      const valBuf = M.allocF64(values.length || 1);
      M.viewF64(valBuf, values.length).set(values);
      buffers.push(idxBuf, valBuf);
      idxBufs.push(idxBuf);
      valBufs.push(valBuf);
      lens.push(len);
      widths.push(width);
    }
    const idxPtrs = heapU32(M, idxBufs);
    const valPtrs = heapU32(M, valBufs);
    const lensPtr = heapU32(M, lens);
    const widthsPtr = heapU32(M, widths);
    buffers.push(idxPtrs, valPtrs, lensPtr, widthsPtr);

    const results: Output[] = [];
    try {
      for (let o = 0; o < this.outputs.length; o++) {
        const out = this.cg.runBatchFlat(idxPtrs, valPtrs, lensPtr, widthsPtr, nIn, o);
        results.push(this.marshalOut(out));
      }
    } catch (e) {
      throw normalizeError(e);
    } finally {
      for (const b of buffers) M.freeBuf(b);
    }

    return this.outputs.length === 1 ? results[0] : results;
  }

  // ---- live streaming --------------------------------------------------
  // Resolve an input name to its C++ input index. compile() adds Input nodes
  // first, in signature order, so the input index equals the name's position.
  private inputIndex(name: string): number {
    const i = this.names.indexOf(name);
    if (i < 0) throw new Error(`unknown input '${name}'`);
    return i;
  }

  // Marshal one drained output (drainFlat picks it, then clears every output).
  private marshalOut(out: any): Output {
    const M: any = this.M;
    const rows = out.rows as number;
    const width = out.width as number;
    const valView = M.viewF64(out.valuePtr, rows * width);
    const idxView = M.viewF64(out.indexPtr, rows);
    const valCopy = new Float64Array(valView);
    const idxCopy = new Float64Array(idxView);
    M.freeBuf(out.indexPtr);
    M.freeBuf(out.valuePtr);
    const values: Float64Array | NdArray =
      width === 1 ? valCopy : ({ data: valCopy, shape: [rows, width] } as NdArray);
    return { values, index: idxCopy };
  }

  // Event-by-event driver over the compiled graph. Push events (scalar or wide)
  // per input, optionally advance the watermark, flush at end of input, then
  // result() drains the accumulated output. Streaming this way is bit-identical
  // to the batch call() on the same data (the batch==stream invariant).
  live(): LiveDriver {
    if (this._disposed) throw new Error("pipeline used after dispose()");
    const M: any = this.M;
    const self = this;
    // Start each live session from a clean slate: run_batch resets on entry but
    // leaves outputs_ populated on exit, and a prior live session leaves node
    // state, so reset here so the stream starts independent of any prior call.
    this.cg.reset();
    return {
      push(name: string, index: number, value: number): void {
        self.cg.pushEvent(self.inputIndex(name), index, value);
      },
      pushWide(name: string, index: number, values: ArrayLike<number>): void {
        const width = values.length;
        const buf = M.allocF64(width || 1);
        M.viewF64(buf, width).set(values as ArrayLike<number> & number[]);
        try {
          self.cg.pushEventWide(self.inputIndex(name), index, buf, width);
        } finally {
          M.freeBuf(buf);
        }
      },
      advance(now: number): void {
        self.cg.advance(now);
      },
      flush(): void {
        self.cg.flush();
      },
      result(): Output {
        // drainFlat(o) drains ALL outputs then returns output o, so a second
        // drain sees nothing. That serves a single-output pipeline exactly;
        // multi-output live drain is not expressible on the current surface.
        if (self.outputs.length !== 1) {
          throw new Error(
            "live().result() supports single-output pipelines only " +
              "(drain clears every output at once)",
          );
        }
        try {
          return self.marshalOut(self.cg.drainFlat(0));
        } catch (e) {
          throw normalizeError(e);
        }
      },
    };
  }

  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    if (this.cg) {
      this.cg.delete();
      this.cg = null;
    }
    // Release our retention; the wrappers' FinalizationRegistry (or an explicit
    // user dispose) frees the underlying C++ ops.
    this.ops = [];
  }
}

// A Pipeline is invoked like a function: `p(feeds)`. It also carries `.dispose()`
// and its node metadata. `new Pipeline(inputs, outputs)` returns this callable.
export interface Pipeline {
  (feeds: Feeds): Output | Output[];
  live(): LiveDriver;
  dispose(): void;
  readonly inputs: Node[];
  readonly outputs: Node[];
}

// Construct a callable Pipeline. Works with or without `new` (the constructor
// returns the callable wrapper, so `new Pipeline(...)` yields the same thing).
export const Pipeline = function (inputs: Node[], outputs: Node[], opts?: unknown): Pipeline {
  const impl = new PipelineImpl(inputs, outputs, opts);
  const fn = ((feeds: Feeds) => impl.call(feeds)) as Pipeline;
  (fn as any).dispose = () => impl.dispose();
  (fn as any).live = () => impl.live();
  Object.defineProperty(fn, "inputs", { value: impl.inputs, enumerable: true });
  Object.defineProperty(fn, "outputs", { value: impl.outputs, enumerable: true });
  return fn;
} as unknown as { new (inputs: Node[], outputs: Node[], opts?: unknown): Pipeline };
