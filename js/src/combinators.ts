// Combinator factories for the define-then-bind Pipeline API. Each returns a
// symbolic Node whose `op` records `{ combinator, params }`; pipeline.ts compile
// dispatches on `combinator` and calls the matching GraphBuilder.add* method.
//
// The resample factory translates its user opts into the integer codes from
// codes.ts, mirroring the kwarg handling in screamer/streams.py's Resample and
// the graph-node compile path in screamer/dag.py (Pipeline._compile_cpp).

import { Node, isNode } from "./node.js";
import type { NdArray } from "./ndarray.js";
import { aKind } from "./kinds.js";
import {
  combineArrays, combineAsync, combineIterators,
  type CombineEvent, type CombineInput, type CombineLayout, type CombineRow,
} from "./combine.js";
import {
  RESAMPLE_MODE_CODE,
  RESAMPLE_AGG_CODE,
  RESAMPLE_FILL_CODE,
  RESAMPLE_LABEL_CODE,
  BAR_AGG_FIXED_PLANS,
} from "./codes.js";

// ---- combineLatest ---------------------------------------------------------

export interface CombineLatestOpts {
  emit?: "when_all" | "on_any";
  // The node's join cap; unset, the Pipeline's maxPending (default 1,000,000).
  // Node inputs only.
  maxPending?: number;
  // How each input's axes read (a layout string per input, or one for all), as
  // in reshape. Arrays and iterators only: streams inside a Pipeline carry
  // fields only.
  layout?: CombineLayout;
}

// The options as a plain object of known keys.
function combineOptions(opts: unknown): { whenAll: boolean; maxPending?: number; layout: CombineLayout | null } {
  const o = opts ?? {};
  const proto = typeof o === "object" ? Object.getPrototypeOf(o) : undefined;
  if (proto !== Object.prototype && proto !== null)
    throw new TypeError(`combineLatest: the options are ${aKind(opts)}, expected an object such as { emit: "on_any" }`);
  for (const key of Object.keys(o))
    if (key !== "emit" && key !== "maxPending" && key !== "layout")
      throw new TypeError(`combineLatest: unknown option '${key}'`);
  const { emit, maxPending, layout } = o as CombineLatestOpts;
  if (emit !== undefined && emit !== "when_all" && emit !== "on_any") {
    if (typeof emit !== "string") throw new TypeError(`combineLatest: emit is ${aKind(emit)}, expected "when_all" or "on_any"`);
    throw new RangeError('combineLatest: emit must be "when_all" or "on_any"');
  }
  setting("combineLatest: maxPending", maxPending, false);
  if (layout !== undefined && layout !== null && typeof layout !== "string") {
    if (!Array.isArray(layout))
      throw new TypeError(`combineLatest: layout is ${aKind(layout)}, expected a string or an array of strings`);
    for (const l of layout)
      if (l !== null && typeof l !== "string")
        throw new TypeError(`combineLatest: a layout is a string or null, got ${aKind(l)}`);
  }
  return { whenAll: (emit ?? "when_all") === "when_all", maxPending, layout: layout ?? null };
}

const isAsyncIterable = (x: any) => x != null && typeof x[Symbol.asyncIterator] === "function";
const isIterator = (x: any) =>
  x != null && typeof x !== "string" && typeof x[Symbol.iterator] === "function" &&
  !Array.isArray(x) && !(ArrayBuffer.isView(x) && !(x instanceof DataView));

/**
 * As-of latest-value join of N streams: one row per distinct index, holding
 * every input's latest values (same-index events coalesce). `emit: "when_all"`
 * (default) starts once every input has a value; `"on_any"` starts at the
 * first event, with NaN for inputs not seen yet. The fields of the inputs
 * concatenate; with `layout`, inputs with asset axes broadcast them.
 *
 * - `Node` inputs build a graph node for a Pipeline (the only form that takes
 *   `maxPending`, and the one that takes no `layout`).
 * - Arrays (a typed array, an Array, T rows or an NdArray) are paired by
 *   position and must have one length along time. An indexed stream is an
 *   object `{ values, index }`, the counterpart of Python's `(values, index)`
 *   tuple; indexed streams join as of their indexes. A plain Array is always
 *   values. Returns `{ values, index }`: values an NdArray
 *   `[rows, ...fields]` or `[rows, ...assets, fields]`, index a Float64Array,
 *   or null without one.
 * - Iterators, or async iterables, of events: bare events (a number, an
 *   Array, a typed array or an NdArray) are paired by position; an event
 *   `{ value, index }`, the counterpart of Python's `(value, index)` tuple,
 *   joins as of its index. Returns an iterator, or an async iterator, of
 *   `{ value, index }` rows (index null without one).
 *
 * Python's `combine_latest` gives the same values.
 */
export function combineLatest(nodes: Node[], opts?: CombineLatestOpts): Node;
export function combineLatest(inputs: CombineInput[], opts?: CombineLatestOpts): { values: NdArray; index: Float64Array | null };
export function combineLatest(inputs: AsyncIterable<CombineEvent>[], opts?: CombineLatestOpts): AsyncGenerator<CombineRow, void, undefined>;
export function combineLatest(inputs: Iterable<CombineEvent>[], opts?: CombineLatestOpts): Generator<CombineRow, void, undefined>;
export function combineLatest(inputs: unknown[], opts?: CombineLatestOpts): unknown {
  if (!Array.isArray(inputs))
    throw new TypeError(`combineLatest: pass an array of inputs, got ${aKind(inputs)}`);
  if (inputs.length === 0) throw new RangeError("combineLatest: needs at least one stream");
  const o = combineOptions(opts);
  const nodes = inputs.filter(isNode).length;
  if (nodes) {
    if (nodes !== inputs.length)
      throw new TypeError("combineLatest: pass every input as a Node, or none");
    if (o.layout !== null)
      throw new RangeError("combineLatest: layout applies to arrays and iterators; " +
        "streams inside a Pipeline carry fields only");
    // Unset: the Pipeline's maxPending. The engine checks the value at compile.
    return new Node(
      { combinator: "combineLatest", params: { whenAll: o.whenAll, maxPending: o.maxPending } },
      inputs as Node[],
    );
  }
  if (o.maxPending !== undefined)
    throw new TypeError("combineLatest: maxPending applies to Node inputs, the join inside a Pipeline");
  const direct = { whenAll: o.whenAll, layout: o.layout };
  const nAsync = inputs.filter(isAsyncIterable).length;
  const nSync = inputs.filter((x) => !isAsyncIterable(x) && isIterator(x)).length;
  if (nAsync && nSync)
    throw new TypeError("combineLatest: async and synchronous inputs cannot be mixed: pass every input as an async iterable, or none");
  if (nAsync === inputs.length) return combineAsync(inputs as AsyncIterable<unknown>[], direct);
  if (nSync === inputs.length) return combineIterators(inputs as Iterable<unknown>[], direct);
  if (nAsync || nSync)
    throw new TypeError("combineLatest: cannot mix lazy iterator and concrete inputs; pass all iterators or all arrays");
  return combineArrays(inputs, direct);
}

// ---- merge -----------------------------------------------------------------

// N-way index-sorted merge. Mirrors screamer/streams.py `merge`, which rejects
// Node inputs: merge is input routing, not a graph node. Feed streams to a
// Pipeline directly instead.
export function merge(..._nodes: Node[]): Node {
  throw new Error(
    "merge is not supported as a Pipeline graph node (it is input routing; " +
      "feed streams to a Pipeline directly)",
  );
}

// ---- select ----------------------------------------------------------------

// Validate columns (a non-negative int or a sequence of them). Mirrors
// screamer/streams.py `_normalize_columns`.
// An operator setting is a number; a string is a wrong kind, never parsed (R-a).
// A number setting (with `integer`, an integer setting): a string or a
// boolean is a wrong kind (TypeError). `bools` false leaves a boolean to a
// later check with its own message (maxFill, maxPending, checked in C++).
// `name` is "<op>: <setting>".
function setting(name: string, v: unknown, bools = true, integer = false): void {
  const want = integer ? "an integer" : "a number";
  if (typeof v === "string" || v instanceof String) throw new TypeError(`${name} must be ${want}, got a string`);
  if (bools && (typeof v === "boolean" || v instanceof Boolean))
    throw new TypeError(`${name} must be ${want}, got boolean`);
}

function normalizeColumns(columns: number | number[]): number[] {
  (Array.isArray(columns) ? columns : [columns]).forEach((c) => setting("select: columns", c, true, true));
  const cols = Array.isArray(columns) ? columns.map((c) => Math.trunc(c)) : [Math.trunc(columns)];
  for (const c of cols) {
    if (c < 0) throw new Error(`select: column index must be non-negative, got ${c}`);
  }
  return cols;
}

// Pick columns from a wide stream.
export function select(node: Node, columns: number | number[]): Node {
  if (!isNode(node)) throw new Error("select: first argument must be a Node");
  const cols = normalizeColumns(columns);
  return new Node({ combinator: "select", params: { columns: cols } }, [node]);
}

// ---- dropna ----------------------------------------------------------------

export interface DropnaOpts {
  how?: "any" | "all";
}

// Drop events with NaN. `how: "any"` (default) drops a row with any NaN across
// its inputs; "all" drops only rows that are all NaN.
export function dropna(nodes: Node | Node[], opts: DropnaOpts = {}): Node {
  const arr = Array.isArray(nodes) ? nodes : [nodes];
  if (arr.length === 0 || !arr.every(isNode)) {
    throw new Error("dropna: pass one Node or a non-empty array of Nodes");
  }
  return new Node({ combinator: "dropna", params: { howAll: opts.how === "all" } }, arr);
}

// ---- filter ----------------------------------------------------------------

// 2-input mask gate: keep each `data` value whose aligned `mask` is nonzero
// (zero or NaN drops). Mirrors screamer/streams.py `Filter`.
export function filter(data: Node, mask: Node): Node {
  if (!isNode(data) || !isNode(mask)) {
    throw new Error("filter: both data and mask must be Nodes");
  }
  return new Node({ combinator: "filter", params: {} }, [data, mask]);
}

// ---- delay -----------------------------------------------------------------

// Re-stamp each event's index by `k` index units; values unchanged.
export function delay(node: Node, k: number): Node {
  if (!isNode(node)) throw new Error("delay: first argument must be a Node");
  setting("delay: duration", k, true, true);
  return new Node({ combinator: "delay", params: { duration: Math.trunc(k) } }, [node]);
}

// ---- resample --------------------------------------------------------------

export interface ResampleOpts {
  // Bucketing controls (mirror streams.py Resample kwargs).
  every?: number; // index-window width (ByIndex)
  count?: number; // arrivals per bar (ByCount)
  threshold?: number; // cumulative driver threshold (ByCumulative)
  clock?: boolean; // clock-driven close (ByClock)
  mode?: keyof typeof RESAMPLE_MODE_CODE; // explicit mode override
  origin?: number;
  label?: "left" | "right";
  fill?: keyof typeof RESAMPLE_FILL_CODE;
  maxAge?: number;
  // The most empty bars one gap may fill (fill "nan" or "carry"): an event or
  // an advance(now) that would close more empty buckets at once throws a
  // RangeError before it builds any bar. Default 1,000,000; an integer from 1
  // to 2**53. Python's max_fill.
  maxFill?: number;
  // Aggregation: a builtin string agg, or a functor reducer as a functor Node
  // (an op applied to a placeholder). `reducer` is the explicit functor slot.
  agg?: string;
  reducer?: Node;
  // Optional second stream (clock or cumulative driver) as a Node.
  driver?: Node;
}

// Resolved numeric params consumed by pipeline.ts compile -> addResample.
export interface ResampleParams {
  modeCode: number;
  aggCode: number;
  labelCode: number;
  fillCode: number;
  width: number;
  origin: number;
  count: number;
  threshold: number;
  maxAge: number;
  maxFill?: number;   // unset: the engine's default; checked at compile
  plan: Array<[number, number]> | null;
  reducer: Node | null;
}

// Causal windowed downsample. Translates opts -> codes, mirroring dag.py's
// Resample compile branch. A builtin agg reduces every column of a wide stream
// on its own; "ohlc" on X columns gives 4 * X columns in column-major field
// order [o0, h0, l0, c0, o1, ...]. A future eager (non-Node) form must return
// [T', X, 4], as Python's eager call does.
export function resample(node: Node, opts: ResampleOpts = {}): Node {
  if (!isNode(node)) throw new Error("resample: first argument must be a Node");

  const threshold = opts.threshold;
  const count = opts.count;
  const clock = opts.clock ?? false;

  // Mode: explicit override, else derive as dag.py does.
  let modeCode: number;
  if (opts.mode !== undefined) {
    if (!(opts.mode in RESAMPLE_MODE_CODE)) throw new Error(`resample: unknown mode '${opts.mode}'`);
    modeCode = RESAMPLE_MODE_CODE[opts.mode];
  } else if (clock) {
    modeCode = RESAMPLE_MODE_CODE.by_clock; // 3
  } else if (threshold !== undefined && threshold !== null) {
    modeCode = RESAMPLE_MODE_CODE.by_cumulative; // 2
  } else if (count !== undefined && count !== null) {
    modeCode = RESAMPLE_MODE_CODE.by_count; // 1
  } else {
    modeCode = RESAMPLE_MODE_CODE.by_index; // 0
  }

  for (const k of ["threshold", "every", "count", "origin", "maxAge", "maxFill"] as const)
    setting(`resample: ${k}`, (opts as any)[k], k !== "maxFill", k !== "threshold");
  const labelCode = RESAMPLE_LABEL_CODE[opts.label ?? "left"]; // right -> 1 else 0
  const width = opts.every !== undefined && opts.every !== null ? Math.trunc(opts.every) : 1;
  const origin = Math.trunc(opts.origin ?? 0);
  const countVal = count !== undefined && count !== null ? Math.trunc(count) : 1;
  const fillCode = RESAMPLE_FILL_CODE[opts.fill ?? "skip"];
  const thresholdVal = threshold !== undefined && threshold !== null ? Number(threshold) : 0.0;
  const maxAgeVal = opts.maxAge !== undefined && opts.maxAge !== null ? Math.trunc(opts.maxAge) : -1;

  const agg = opts.agg ?? "last";
  let aggCode: number;
  let plan: Array<[number, number]> | null = null;
  let reducer: Node | null = null;

  if (opts.reducer !== undefined && opts.reducer !== null) {
    // Functor reducer supplied explicitly: agg code ignored (0).
    if (!isNode(opts.reducer) || (opts.reducer.op as any)?.functor === undefined) {
      throw new Error("resample: reducer must be a functor Node (an op applied to a placeholder)");
    }
    aggCode = 0;
    reducer = opts.reducer;
  } else if (agg in BAR_AGG_FIXED_PLANS) {
    // Multi-column bar agg with a fixed plan (ohlc_bars, ohlcv, ohlcv2).
    aggCode = 0;
    plan = BAR_AGG_FIXED_PLANS[agg];
  } else if (agg === "ohlcv_bars") {
    // Dynamic plan built in C++ from the input width; empty plan, agg=10.
    aggCode = RESAMPLE_AGG_CODE.ohlcv_bars; // 10
  } else if (agg in RESAMPLE_AGG_CODE) {
    aggCode = RESAMPLE_AGG_CODE[agg];
  } else {
    throw new Error(`resample: unknown agg '${agg}'`);
  }

  const params: ResampleParams = {
    modeCode, aggCode, labelCode, fillCode,
    width, origin, count: countVal, threshold: thresholdVal, maxAge: maxAgeVal,
    plan, reducer,
  };
  if (opts.maxFill !== undefined && opts.maxFill !== null) params.maxFill = opts.maxFill;

  // A driver is the node's second input. For a cumulative resample the graph
  // builder (C++) joins value and driver into one [value, driver] row first; a
  // clock driver goes to the node's own clock port.
  const inputs = opts.driver && isNode(opts.driver) ? [node, opts.driver] : [node];
  return new Node({ combinator: "resample", params }, inputs);
}
