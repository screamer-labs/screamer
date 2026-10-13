// Combinator factories for the define-then-bind Pipeline API. Each returns a
// symbolic Node whose `op` records `{ combinator, params }`; pipeline.ts compile
// dispatches on `combinator` and calls the matching GraphBuilder.add* method.
//
// The resample factory translates its user opts into the integer codes from
// codes.ts, mirroring the kwarg handling in screamer/streams.py's Resample and
// the graph-node compile path in screamer/dag.py (Pipeline._compile_cpp).

import { Input, Node, isNode } from "./node.js";
import type { NdArray } from "./ndarray.js";
import { aKind, kindName, throwMessage } from "./kinds.js";
import {
  block, checkKeys, combineArrays, combineAsync, combineIterators, indexArray, isPlainObject,
  type CombineEvent, type CombineInput, type CombineLayout, type CombineRow, type CombineValues,
} from "./combine.js";
import {
  maxPendingCount, Pipeline, runLazy, setWho,
  type LazyOptions, type Output, type PipelineRow,
} from "./pipeline.js";
import * as ops from "./generated/ops.js";
import { current } from "./index.js";
import { mergeArrays, mergeAsync, mergeIterators, type MergeEvent, type MergeResult } from "./merge.js";
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
  const { emit, maxPending } = o as CombineLatestOpts;
  // C++ reads the layout while WASM frames are live, so it gets a plain copy
  // (a Proxy or a getter would run user code there), checked after the copy.
  const given = (o as CombineLatestOpts).layout;
  const layout = Array.isArray(given) ? Array.from(given) : given;
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
    // The node's cap is checked now, as Python's CombineLatest checks it.
    if (o.maxPending !== undefined) maxPendingCount(current(), o.maxPending, "combineLatest: maxPending");
    // `spec`: the node as Python's to_dict writes it (Pipeline.toJSON).
    const spec = { op: "CombineLatest", params: { emit: o.whenAll ? "when_all" : "on_any",
                                                  ...(o.maxPending !== undefined ? { max_pending: o.maxPending } : {}) } };
    return new Node(
      { combinator: "combineLatest", params: { whenAll: o.whenAll, maxPending: o.maxPending }, spec },
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

/**
 * Merge N streams into one in index order; equal indexes keep the input
 * order. Every input has rows of one width (a 1-D stream has width 1).
 *
 * - Arrays (a typed array, an Array, T rows or an NdArray), or indexed
 *   streams `{ values, index }`, the counterpart of Python's `(values, index)`
 *   tuple. Without an index each input's index is its row numbers. Returns
 *   `{ values, sources, index }`: the rows in merged order (a Float64Array,
 *   or an NdArray `[rows, ...row shape]` for wide inputs), the input of each
 *   row (a Uint32Array) and the merged index (null without one).
 * - Iterators, or async iterables, of events: a value (a number, an Array, a
 *   typed array or an NdArray), or `{ value, index }`. Returns an iterator,
 *   or an async iterator, of `{ value, index, source }` events (index null
 *   without one).
 *
 * Python's `merge` gives the same values. A merge is input routing, not a
 * Pipeline node: feed the streams to a Pipeline directly.
 */
export function merge(inputs: MergeInput[]): MergeResult;
export function merge(inputs: AsyncIterable<CombineEvent>[]): AsyncGenerator<MergeEvent, void, undefined>;
export function merge(inputs: Iterable<CombineEvent>[]): Generator<MergeEvent, void, undefined>;
export function merge(inputs: unknown, ...rest: unknown[]): unknown {
  if (isNode(inputs) || rest.some(isNode) || (Array.isArray(inputs) && inputs.some(isNode)))
    throw new RangeError(
      "merge is not supported as a Pipeline graph node (it is input routing; " +
        "feed streams to a Pipeline directly)",
    );
  if (!Array.isArray(inputs))
    throw new TypeError(`merge: pass an array of inputs, got ${aKind(inputs)}`);
  if (rest.length) throw new TypeError(`merge: takes one argument, the array of inputs, got ${rest.length + 1}`);
  if (inputs.length === 0) throw new RangeError("merge: needs at least one stream");
  const nAsync = inputs.filter(isAsyncIterable).length;
  const nSync = inputs.filter((x) => !isAsyncIterable(x) && isIterator(x)).length;
  if (nAsync && nSync)
    throw new TypeError("merge: async and synchronous inputs cannot be mixed: pass every input as an async iterable, or none");
  if (nAsync === inputs.length) return mergeAsync(inputs as AsyncIterable<unknown>[]);
  if (nSync === inputs.length) return mergeIterators(inputs as Iterable<unknown>[]);
  if (nAsync || nSync)
    throw new TypeError("merge: cannot mix lazy iterator and concrete inputs; pass all iterators or all arrays");
  return mergeArrays(inputs);
}

/** One input of the array form of merge: values, or `{ values, index }`. */
export type MergeInput = CombineInput;

// ---- direct forms ----------------------------------------------------------
//
// select, dropna, filter, delay and resample take Nodes (a graph node for a
// Pipeline), arrays (a typed array, an Array, T rows or an NdArray, or an
// indexed stream `{ values, index }`, Python's `(values, index)` tuple), or
// sync iterators of events (`{ value, index }`, Python's `(value, index)`
// pair). The array and iterator forms run the same C++ node through a
// one-node Pipeline, as Python's do, so every form gives the same rows.

// One stream of the array forms: its values and its index (null without one).
interface Stream { values: Float64Array | NdArray; index: Float64Array | null; shape: number[] }

const isStreamObject = (x: unknown): x is Record<string, unknown> => isPlainObject(x) && "values" in x;

// A stream of the array form: values, or { values, index }.
function readStream(x: unknown, who: string, k: number): Stream {
  let v: unknown = x, idx: Float64Array | null = null;
  const name = `input ${k}`;
  if (isPlainObject(x)) {
    checkKeys(x, ["values", "index"], name, who);
    if (!("values" in x)) throw new TypeError(`${who}: ${name} is an object without values; expected { values, index }`);
    v = x.values;
    if (x.index != null) idx = indexArray(x.index, `the index of ${name}`, who);
  }
  const b = block(v, isPlainObject(x) ? `the values of ${name}` : name, who);
  const values = b.shape.length === 1 ? b.data : { data: b.data, shape: b.shape };
  return { values, index: idx, shape: b.shape };
}

// Every stream has an index, or none (rows numbered as in a batch call).
function positionalOf(streams: Stream[], who: string): boolean {
  const n = streams.filter((s) => s.index !== null).length;
  if (n && n !== streams.length)
    throw new RangeError(`${who}: cannot mix positional and indexed streams; give every stream an index, or none`);
  return n === 0;
}

// Runs a one-node Pipeline, build(...inputs), on the streams (each given its
// index, or none). Returns its output; the Pipeline is freed.
function runOnce(build: (...ins: Node[]) => Node, feeds: Array<{ values: Float64Array | NdArray; index?: ArrayLike<number> | null }>,
                 who: string): Output {
  const ins = feeds.map((_, i) => Input(`_${who}${i}`));
  const p = new Pipeline(ins, [build(...ins)]);
  try {
    setWho(p, who);
    const byName: Record<string, any> = {};
    ins.forEach((n, i) => { byName[(n.op as any).input] = feeds[i].index == null ? feeds[i].values : feeds[i]; });
    return p(byName) as Output;
  } finally {
    p.dispose();
  }
}

// The lazy form: a one-node Pipeline over iterators. The Pipeline is freed
// when the run ends (or, abandoned, when it is collected).
function runIterators(build: (...ins: Node[]) => Node, srcs: Iterable<unknown>[], who: string,
                      o: LazyOptions): Generator<PipelineRow, void, undefined> {
  const ins = srcs.map((_, i) => Input(`_${who}${i}`));
  const p = new Pipeline(ins, [build(...ins)]);
  setWho(p, who);
  const run = runLazy(p, srcs, o);
  return (function* () {
    try { yield* run; } finally { p.dispose(); }
  })();
}

// Whether every input is an iterator, none is, or (a TypeError) some are.
function iteratorsOf(xs: unknown[], who: string): boolean {
  const n = xs.filter(isIterator).length;
  if (n && n !== xs.length)
    throw new TypeError(`${who}: cannot mix lazy iterator and concrete inputs; pass all iterators or all arrays`);
  if (!n && xs.some(isAsyncIterable))
    throw new TypeError(`${who}: takes arrays or sync iterators; collect an async source into an array, or iterate it with a live() session`);
  return n > 0;
}

// ---- select ----------------------------------------------------------------

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

// Validate columns (a non-negative int or a sequence of them). Mirrors
// screamer/streams.py `_normalize_columns`.
function normalizeColumns(columns: number | number[]): number[] {
  const given = Array.isArray(columns) ? Array.from(columns) : [columns];
  for (const c of given) {
    setting("select: columns", c, true, true);
    if (typeof c !== "number") throw new TypeError(`select: columns must be integers, got ${aKind(c)}`);
    if (!Number.isFinite(c)) throw new RangeError(`select: column index must be a finite number, got ${c}`);
  }
  const cols = given.map((c) => Math.trunc(c));
  for (const c of cols) {
    if (c < 0) throw new RangeError(`select: column index must be non-negative, got ${c}`);
  }
  return cols;
}

// The width of a stream's events: 1 for a 1-D stream, else its second axis.
const widthOf = (shape: number[]): number => (shape.length === 1 ? 1 : shape[1]);
const checkColumns = (cols: number[], width: number): void => {
  for (const c of cols)
    if (c >= width) throw new RangeError(`select: column ${c} out of range for width ${width}`);
};

/**
 * Pick column(s) of a wide stream: an integer gives one column (a 1-D
 * result), an array of integers those columns in order.
 *
 * - A `Node` builds a graph node for a Pipeline.
 * - Arrays (a typed array, T rows or an NdArray `[T, w]`), or `{ values, index }`:
 *   returns `{ values, index }`, index null without one.
 * - An iterator of events (a value, or `{ value, index }`): returns an
 *   iterator of `{ value, index }` rows.
 *
 * Python's `Select(columns)` gives the same values.
 */
export function select(node: Node, columns: number | number[]): Node;
export function select(values: CombineInput, columns: number | number[]): { values: Float64Array | NdArray; index: Float64Array | null };
export function select(events: Iterable<CombineEvent>, columns: number | number[]): Generator<PipelineRow, void, undefined>;
export function select(x: unknown, columns: number | number[]): unknown {
  if (isNode(x)) {
    const cols = normalizeColumns(columns);
    const spec = { op: "Select", params: { columns: Array.isArray(columns) ? [...columns] : columns } };
    return new Node({ combinator: "select", params: { columns: cols }, spec }, [x]);
  }
  const cols = normalizeColumns(columns);
  const scalar = !Array.isArray(columns);
  const arg = scalar ? cols[0] : cols;
  // Every form gives the one-node Pipeline's rows (Python's Select does too):
  // a width-1 output is a number per row, duplicate columns are kept.
  if (iteratorsOf([x], "select")) {
    return runIterators((n) => select(n, arg), [x as Iterable<unknown>], "select", {
      positional: true,
      shape: (first) => {
        const w = typeof first === "number" ? 1 : (first as any).data ? (first as NdArray).shape[(first as NdArray).shape.length - 1] : (first as Float64Array).length;
        checkColumns(cols, w);
        return null;
      },
    });
  }
  const s = readStream(x, "select", 0);
  if (s.shape.length > 2) throw new RangeError(`select: the values have shape [${s.shape}], expected (T,) or (T, w)`);
  checkColumns(cols, widthOf(s.shape));
  const out = runOnce((n) => select(n, arg), [s], "select");
  return { values: out.values, index: s.index === null ? null : out.index };
}

// ---- dropna ----------------------------------------------------------------

export interface DropnaOpts {
  how?: "any" | "all";
}

// The options of dropna: a plain object of known keys, `how` "any" or "all".
function dropnaHow(opts: unknown): boolean {
  const o = opts ?? {};
  const proto = typeof o === "object" ? Object.getPrototypeOf(o) : undefined;
  if (proto !== Object.prototype && proto !== null)
    throw new TypeError(`dropna: the options are ${aKind(opts)}, expected an object such as { how: "all" }`);
  for (const key of Object.keys(o)) if (key !== "how") throw new TypeError(`dropna: unknown option '${key}'`);
  const how = (o as DropnaOpts).how;
  if (how !== undefined && how !== "any" && how !== "all") throw new RangeError('dropna: how must be "any" or "all"');
  return how === "all";
}

/**
 * Drop events with NaN. `how: "any"` (default) drops a row with any NaN,
 * "all" only a row that is all NaN.
 *
 * - A `Node`, or an array of Nodes (several streams), builds a graph node.
 * - Arrays, or `{ values, index }`: returns `{ values, index }`, index null
 *   without one. Several streams are an array of `{ values, index }` objects
 *   (`{ values }` for a positional one): their events merge in index order
 *   (ties in input order) and each is dropped on its own.
 * - An iterator of events, or an array of iterators: returns an iterator of
 *   `{ value, index }` rows.
 *
 * Python's `Dropna(how)` gives the same values.
 */
export function dropna(nodes: Node | Node[], opts?: DropnaOpts): Node;
export function dropna(values: CombineInput | Array<{ values: CombineValues; index?: ArrayLike<number> | null }>, opts?: DropnaOpts): { values: Float64Array | NdArray; index: Float64Array | null };
export function dropna(events: Iterable<CombineEvent> | Iterable<CombineEvent>[], opts?: DropnaOpts): Generator<PipelineRow, void, undefined>;
export function dropna(x: unknown, opts: DropnaOpts = {}): unknown {
  const howAll = dropnaHow(opts);
  const how = howAll ? "all" : "any";
  // Several streams: an array of Nodes, of { values, index } objects, or of
  // iterators. Any other array is one stream's values.
  const several = Array.isArray(x) && x.length > 0 &&
    (x.every(isNode) || x.every(isStreamObject) || x.every(isIterator));
  const xs: unknown[] = several ? (x as unknown[]) : [x];
  if (xs.some(isNode)) {
    if (!xs.every(isNode)) throw new TypeError("dropna: pass every stream as a Node, or none");
    const spec = { op: "Dropna", params: { how } };
    return new Node({ combinator: "dropna", params: { howAll }, spec }, xs as Node[]);
  }
  const build = (...ins: Node[]) => dropna(ins, { how });
  if (iteratorsOf(xs, "dropna"))
    return runIterators(build, xs as Iterable<unknown>[], "dropna", { positional: true });
  const streams = xs.map((s, k) => readStream(s, "dropna", k));
  const positional = positionalOf(streams, "dropna");
  const out = runOnce(build, streams, "dropna");
  return { values: out.values, index: positional ? null : out.index };
}

// ---- filter ----------------------------------------------------------------

/**
 * 2-input mask gate: keep each `data` value whose aligned `mask` is nonzero
 * (zero or NaN drops).
 *
 * - `Node`s build a graph node for a Pipeline.
 * - Arrays, or `{ values, index }`: returns `{ values, index }` (row numbers
 *   for positional streams), as a Pipeline call does.
 * - Iterators of `{ value, index }` events: returns an iterator of
 *   `{ value, index }` rows.
 *
 * Python's `Filter()(data, mask)` gives the same values.
 */
export function filter(data: Node, mask: Node): Node;
export function filter(data: CombineInput, mask: CombineInput): Output;
export function filter(data: Iterable<CombineEvent>, mask: Iterable<CombineEvent>): Generator<PipelineRow, void, undefined>;
export function filter(data: unknown, mask: unknown): unknown {
  if (isNode(data) || isNode(mask)) {
    if (!isNode(data) || !isNode(mask)) throw new TypeError("filter: pass both data and mask as Nodes, or neither");
    return new Node({ combinator: "filter", params: {}, spec: { op: "Filter", params: {} } }, [data, mask]);
  }
  const build = (d: Node, m: Node) => filter(d, m);
  if (iteratorsOf([data, mask], "filter"))
    return runIterators(build, [data as Iterable<unknown>, mask as Iterable<unknown>], "filter", {});
  const streams = [readStream(data, "filter", 0), readStream(mask, "filter", 1)];
  return runOnce(build, streams, "filter");
}

// ---- delay -----------------------------------------------------------------

/**
 * Re-stamp each event's index by `k` index units; values unchanged.
 *
 * - A `Node` builds a graph node for a Pipeline.
 * - `{ values, index }`: returns `{ values, index }` with the index moved.
 *   A delay is time-based, so the stream needs its index.
 *
 * Python's `Delay(duration)(values, index)` gives the same values.
 */
export function delay(node: Node, k: number): Node;
export function delay(stream: { values: CombineValues; index: ArrayLike<number> }, k: number): Output;
export function delay(x: unknown, k: number): unknown {
  setting("delay: duration", k, true, true);
  const duration = Math.trunc(k);
  if (isNode(x))
    return new Node({ combinator: "delay", params: { duration }, spec: { op: "Delay", params: { duration } } }, [x]);
  const s = isIterator(x) ? null : readStream(x, "delay", 0);
  if (!s || s.index === null)
    throw new TypeError("delay requires an explicit index (duration is time-based); pass { values, index }");
  return runOnce((n) => delay(n, duration), [s], "delay");
}

// ---- resample --------------------------------------------------------------

export interface ResampleOpts {
  // Bucketing controls (mirror streams.py Resample kwargs).
  every?: number; // index-window width (ByIndex)
  count?: number; // arrivals per bar (ByCount)
  threshold?: number; // cumulative driver threshold (ByCumulative)
  clock?: boolean; // clock-driven close (ByClock)
  // The mode the settings above give; given, it must match them.
  mode?: keyof typeof RESAMPLE_MODE_CODE;
  origin?: number;
  label?: "left" | "right";
  fill?: keyof typeof RESAMPLE_FILL_CODE;
  maxAge?: number;
  // The most empty bars one gap may fill (fill "nan" or "carry"): an event or
  // an advance(now) that would close more empty buckets at once throws a
  // RangeError before it builds any bar. Default 1,000,000; an integer from 1
  // to 2**53. Python's max_fill.
  maxFill?: number;
  // Aggregation: a builtin name ("high", "low", "open" and "close" read as
  // "max", "min", "first" and "last"; "std", "var", "prod", "skew" and "kurt"
  // as the Expanding* reducer), or a functor reducer as a functor Node (an op
  // applied to a placeholder). `reducer` is the explicit functor slot.
  agg?: string | Node;
  reducer?: Node;
  // The second stream (the clock of clock: true, the driver of threshold):
  // a Node for a graph node, an array or { values, index } for the array
  // form, an iterator for the iterator form.
  driver?: unknown;
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

const RESAMPLE_KEYS = ["every", "count", "threshold", "clock", "mode", "origin", "label", "fill",
                       "maxAge", "maxFill", "agg", "reducer", "driver"];
// Finance aliases and statistical synonyms, as Python resolves them.
const FINANCE_ALIASES: Record<string, string> = { high: "max", low: "min", open: "first", close: "last" };
const STAT_SYNONYMS: Record<string, string> = {
  std: "ExpandingStd", var: "ExpandingVar", prod: "ExpandingProd", skew: "ExpandingSkew", kurt: "ExpandingKurt",
};
// Builtin aggs that reduce each column on its own.
const COLUMN_AGGS = new Set(["first", "last", "min", "max", "sum", "count", "mean", "ohlc"]);
const given = (v: unknown) => v !== undefined && v !== null;
const M0 = (): any => current();

// The checked settings: the options as a plain object of known keys, the
// agg resolved, and the shared C++ rule (dag/resample_check.h, Python's).
function resampleSettings(opts: unknown): { o: ResampleOpts; agg: string; reducer: Node | null } {
  const o = (opts ?? {}) as ResampleOpts;
  const proto = typeof o === "object" ? Object.getPrototypeOf(o) : undefined;
  if (proto !== Object.prototype && proto !== null)
    throw new TypeError(`resample: the options are ${aKind(opts)}, expected an object such as { every: 5 }`);
  for (const key of Object.keys(o))
    if (!RESAMPLE_KEYS.includes(key)) throw new TypeError(`resample: unknown option '${key}'`);
  for (const k of ["threshold", "every", "count", "origin", "maxAge", "maxFill"] as const)
    setting(`resample: ${k}`, (o as any)[k], k !== "maxFill", k !== "threshold");
  for (const k of ["threshold", "every", "count", "origin", "maxAge"] as const) {
    const v = (o as any)[k];
    if (given(v) && typeof v !== "number") throw new TypeError(`resample: ${k} must be a number, got ${aKind(v)}`);
  }
  // The agg: a name (aliases resolved), or a functor reducer.
  let agg: string = "last", reducer: Node | null = null;
  let a: unknown = o.agg ?? "last";
  if (given(o.reducer)) {
    if (o.agg !== undefined) throw new TypeError("resample: pass agg or reducer, not both");
    a = o.reducer;
  }
  if (typeof a === "string") {
    if (a in STAT_SYNONYMS) reducer = (ops as any)[STAT_SYNONYMS[a]]()(Input("_reducer"));
    else agg = FINANCE_ALIASES[a] ?? a;
  } else if (isNode(a) && (a.op as any)?.functor !== undefined) {
    reducer = a;
  } else {
    throw new TypeError("resample: agg must be a name or a functor Node (an op applied to a placeholder)");
  }
  // maxFill now, as Python checks max_fill at the call (unset: the default).
  if (given(o.maxFill)) {
    if (typeof o.maxFill !== "number")
      throwMessage(M0().maxPendingKindMessage(kindName(o.maxFill), "resample: maxFill"));
    throwMessage(M0().maxFillMessage(o.maxFill, "resample: maxFill"));
  }
  const text = (v: unknown, d: string) => (v === undefined ? d : typeof v === "string" ? v : "\u0000");
  const M: any = current();
  const num = (v: unknown) => (given(v) ? (v as number) : 0);
  throwMessage(M.resampleSettingsMessage(
    given(o.every), num(o.every), given(o.count), num(o.count), given(o.threshold), num(o.threshold),
    !!o.clock, reducer !== null, reducer ? "" : agg, text(o.label, "left"), text(o.fill, "skip"),
    given(o.origin), num(o.origin), given(o.maxAge), !given(o.maxAge) || Number.isInteger(o.maxAge), num(o.maxAge)));
  return { o, agg, reducer };
}

// The resample node on `node` (and `driver`, a Node or null), from checked settings.
function resampleNode(node: Node, driver: Node | null, s: { o: ResampleOpts; agg: string; reducer: Node | null }): Node {
  const { o, agg, reducer } = s;
  const clock = !!o.clock;
  const modeCode = clock ? RESAMPLE_MODE_CODE.by_clock
    : given(o.threshold) ? RESAMPLE_MODE_CODE.by_cumulative
    : given(o.count) ? RESAMPLE_MODE_CODE.by_count : RESAMPLE_MODE_CODE.by_index;
  if (o.mode !== undefined) {
    if (typeof o.mode !== "string" || !(o.mode in RESAMPLE_MODE_CODE))
      throw new RangeError(`resample: unknown mode '${String(o.mode)}'`);
    if (RESAMPLE_MODE_CODE[o.mode] !== modeCode)
      throw new RangeError(`resample: mode '${o.mode}' does not match the settings given`);
  }
  const labelCode = RESAMPLE_LABEL_CODE[o.label ?? "left"];
  const width = given(o.every) ? Math.trunc(o.every!) : 1;
  const origin = Math.trunc(o.origin ?? 0);
  const countVal = given(o.count) ? Math.trunc(o.count!) : 1;
  const fillCode = RESAMPLE_FILL_CODE[o.fill ?? "skip"];
  const thresholdVal = given(o.threshold) ? Number(o.threshold) : 0.0;
  const maxAgeVal = given(o.maxAge) ? Math.trunc(o.maxAge!) : -1;

  let aggCode: number;
  let plan: Array<[number, number]> | null = null;
  if (reducer) aggCode = 0;                                   // the functor reduces
  else if (agg in BAR_AGG_FIXED_PLANS) { aggCode = 0; plan = BAR_AGG_FIXED_PLANS[agg]; }
  else aggCode = RESAMPLE_AGG_CODE[agg];                      // ohlcv_bars: 10, a plan built in C++

  const params: ResampleParams = {
    modeCode, aggCode, labelCode, fillCode,
    width, origin, count: countVal, threshold: thresholdVal, maxAge: maxAgeVal,
    plan, reducer,
  };
  if (given(o.maxFill)) params.maxFill = o.maxFill;

  // A driver is the node's second input. For a cumulative resample the graph
  // builder (C++) joins value and driver into one [value, driver] row first; a
  // clock driver goes to the node's own clock port.
  const inputs = driver ? [node, driver] : [node];
  // The node as Python's to_dict writes it (Pipeline.toJSON): the mode's own
  // setting, then agg, origin, label and fill, and the optional ones given.
  const sp: Record<string, unknown> = {};
  if (modeCode === RESAMPLE_MODE_CODE.by_clock) sp.clock = true;
  else if (modeCode === RESAMPLE_MODE_CODE.by_cumulative) sp.threshold = thresholdVal;
  else {
    if (given(o.every)) sp.every = width;
    if (modeCode === RESAMPLE_MODE_CODE.by_count) sp.count = countVal;
  }
  sp.agg = reducer ?? agg;   // a reducer node is written as its functor (toJSON)
  sp.origin = origin;
  sp.label = o.label ?? "left";
  sp.fill = o.fill ?? "skip";
  if (modeCode === RESAMPLE_MODE_CODE.by_clock && maxAgeVal >= 0) sp.max_age = maxAgeVal;
  if (given(o.maxFill)) sp.max_fill = o.maxFill;
  return new Node({ combinator: "resample", params, spec: { op: "Resample", params: sp } }, inputs);
}

// The (*A) part of a (T, *A) or (T, *A, 1) shape (rule 1).
const columnShape = (shape: number[]): number[] => {
  const t = shape.slice(1);
  return t.length >= 2 && t[t.length - 1] === 1 ? t.slice(0, -1) : t;
};
const prod = (xs: number[]) => xs.reduce((a, b) => a * b, 1);
const shapeText = (s: number[]) => `[${s}]`;

// A stream's values as (T,) or (T, w) for a Pipeline feed: `cols` flattens
// the (*A) of a per-column agg.
const asFeed = (s: Stream, cols: number[] | null): Float64Array | NdArray => {
  const data = s.values instanceof Float64Array ? s.values : s.values.data;
  if (s.shape.length === 1) return data;
  const w = cols ? prod(cols) : prod(s.shape.slice(1));
  return { data, shape: [s.shape[0], w] };
};

// A resample result: the values in the rows' shape (T', *A[, 4]) for a
// per-column agg, the bar labels as the index.
function shaped(out: Output, cols: number[] | null, agg: string): { values: Float64Array | NdArray; index: Float64Array } {
  if (!cols) return { values: out.values, index: out.index };
  const data = out.values instanceof Float64Array ? out.values : out.values.data;
  const shape = [out.index.length, ...cols, ...(agg === "ohlc" ? [4] : [])];
  return { values: shape.length === 1 ? data : { data, shape }, index: out.index };
}

/**
 * Causal windowed downsample: bars of `every` index units, `count`
 * arrivals, a cumulative `threshold` of a driver stream, or the ticks of a
 * clock stream (`clock: true`). A builtin agg reduces every column of a
 * wide stream on its own; "ohlc" on X columns gives 4 * X columns in field
 * order [o0, h0, l0, c0, o1, ...] inside a Pipeline, and `[T', X, 4]` from
 * the array form, as in Python.
 *
 * - A `Node` builds a graph node for a Pipeline (`driver` a Node).
 * - Arrays, or `{ values, index }` (`driver` the same): returns
 *   `{ values, index }`, the index the bar labels. Values `(T, *A)` with a
 *   per-column agg give `(T', *A)`, with a last axis of 4 for "ohlc".
 * - An iterator of `{ value, index }` events (`driver` an iterator for a
 *   clock): returns an iterator of `{ value, index }` bars.
 *
 * Python's `Resample(...)` gives the same values.
 */
export function resample(node: Node, opts?: ResampleOpts): Node;
export function resample(values: CombineInput, opts?: ResampleOpts): { values: Float64Array | NdArray; index: Float64Array };
export function resample(events: Iterable<CombineEvent>, opts?: ResampleOpts): Generator<PipelineRow, void, undefined>;
export function resample(x: unknown, opts: ResampleOpts = {}): unknown {
  const s = resampleSettings(opts);
  const { o, agg, reducer } = s;
  const clock = !!o.clock, threshold = given(o.threshold);
  const driver = o.driver;
  if (given(driver) && !clock && !threshold)
    throw new RangeError("resample: a driver requires threshold or clock: true to set the mode");
  const checked = s;   // resampleNode reads the settings, never the driver

  if (isNode(x)) {
    if (clock && !isNode(driver))
      throw new RangeError("resample: clock: true requires a clock Node as the driver");
    if (threshold && given(driver) && !isNode(driver))
      throw new RangeError("resample: threshold requires a driver Node as the driver");
    return resampleNode(x, isNode(driver) ? driver : null, checked);
  }
  if (isNode(driver)) throw new TypeError("resample: pass the values and the driver as Nodes, or neither");
  const perColumn = !reducer && COLUMN_AGGS.has(agg);

  // ---- iterators
  if (iteratorsOf(given(driver) ? [x, driver] : [x], "resample")) {
    if (threshold)
      throw new TypeError("resample: threshold takes arrays; pass the values and the driver as arrays");
    if (clock && !given(driver))
      throw new RangeError("resample: clock: true requires a clock stream as the driver");
    // A per-column agg gives each bar in the event's shape, (*A) or (*A, 4).
    const shape = (first: unknown): number[] | null => {
      if (!perColumn || typeof first === "number") return null;
      const nd = first as any;
      const a: number[] = nd.data ? nd.shape.slice() : [nd.length];
      if (a.length >= 2 && a[a.length - 1] === 1) a.pop();
      return agg === "ohlc" ? [...a, 4] : a;
    };
    const srcs = clock ? [x, driver] : [x];
    const build = (v: Node, c?: Node) => resampleNode(v, c ?? null, checked);
    return runIterators(build, srcs as Iterable<unknown>[], "resample", { shape });
  }

  // ---- arrays
  const vs = readStream(x, "resample", 0);
  const T = vs.shape[0];
  const index = vs.index ?? Float64Array.from({ length: T }, (_, i) => i);
  if (clock) {
    if (!given(driver)) throw new RangeError("resample: clock: true requires a clock stream as the driver");
    if (given(o.maxAge) && vs.shape.length !== 1) throw new RangeError("resample: maxAge requires a 1-D value stream");
    const cs = readStream(driver, "resample", 1);
    const cdata = cs.values instanceof Float64Array ? cs.values : cs.values.data;
    // The clock's values are not read: its first column stands for them.
    const cw = prod(cs.shape.slice(1));
    const clk = cw === 1 ? cdata : Float64Array.from({ length: cs.shape[0] }, (_, i) => cdata[i * cw]);
    const cidx = cs.index ?? Float64Array.from({ length: cs.shape[0] }, (_, i) => i);
    const cols = perColumn && vs.shape.length > 1 ? columnShape(vs.shape) : null;
    const out = runOnce((v, c) => resampleNode(v, c, checked),
      [{ values: asFeed(vs, cols), index }, { values: clk, index: cidx }], "resample");
    return shaped(out, cols, agg);
  }
  if (threshold) {
    if (!given(driver)) throw new RangeError("resample: threshold requires a driver stream as the driver");
    if (vs.shape.length !== 1)
      throw new RangeError(`resample: threshold requires a 1-D value stream; got shape ${shapeText(vs.shape)}`);
    const ds = readStream(driver, "resample", 1);
    if (ds.index !== null)
      throw new RangeError("resample: a threshold driver is paired with the values by position; " +
        "pass the driver without an index");
    if (ds.shape.length !== 1)
      throw new RangeError(`resample: threshold requires a 1-D driver stream; got shape ${shapeText(ds.shape)}`);
    if (ds.shape[0] !== T)
      throw new RangeError(`resample: value and driver must have the same length; got ${T} and ${ds.shape[0]}`);
    // One [value, driver] row per event.
    const v = vs.values as Float64Array, d = ds.values as Float64Array;
    const both = new Float64Array(2 * T);
    for (let i = 0; i < T; i++) { both[2 * i] = v[i]; both[2 * i + 1] = d[i]; }
    const out = runOnce((n) => resampleNode(n, null, checked), [{ values: { data: both, shape: [T, 2] }, index }], "resample");
    return { values: out.values, index: out.index };
  }
  // every= or count=: the bar aggs need their columns.
  if (agg === "ohlcv" || agg === "ohlcv2") {
    if (vs.shape.length !== 2 || vs.shape[1] !== 2)
      throw new RangeError(`resample: agg='${agg}' requires exactly 2 columns [price, ${agg === "ohlcv" ? "volume" : "signed_volume"}]; got shape ${shapeText(vs.shape)}`);
  } else if (agg === "ohlc_bars") {
    if (vs.shape.length !== 2 || vs.shape[1] !== 4)
      throw new RangeError(`resample: agg='${agg}' requires exactly 4 columns [open, high, low, close]; got shape ${shapeText(vs.shape)}`);
  } else if (agg === "ohlcv_bars") {
    if (vs.shape.length !== 2)
      throw new RangeError(`resample: agg='${agg}' requires a 2-D input [open, high, low, close, vol1, ...]; got shape ${shapeText(vs.shape)}`);
  } else if (vs.shape.length !== 1 && !perColumn) {
    throw new RangeError(`resample: a functor agg aggregates one column; got shape ${shapeText(vs.shape)}. ` +
      "Resample each column with its own functor, or use a builtin agg.");
  }
  const cols = perColumn && vs.shape.length > 1 ? columnShape(vs.shape) : null;
  const out = runOnce((n) => resampleNode(n, null, checked), [{ values: asFeed(vs, cols), index }], "resample");
  return shaped(out, cols, agg);
}
