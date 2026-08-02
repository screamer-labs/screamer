// Combinator factories for the define-then-bind Pipeline API. Each returns a
// symbolic Node whose `op` records `{ combinator, params }`; pipeline.ts compile
// dispatches on `combinator` and calls the matching GraphBuilder.add* method.
//
// The resample factory translates its user opts into the integer codes from
// codes.ts, mirroring the kwarg handling in screamer/streams.py's Resample and
// the graph-node compile path in screamer/dag.py (Pipeline._compile_cpp).

import { Node, isNode } from "./node.js";
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
  maxPending?: number;
}

// As-of latest-value join over N Node streams. `emit: "when_all"` (default) emits
// only once every input has produced a value; "on_any" emits on any input event.
export function combineLatest(nodes: Node[], opts: CombineLatestOpts = {}): Node {
  if (!Array.isArray(nodes) || nodes.length === 0 || !nodes.every(isNode)) {
    throw new Error("combineLatest: pass a non-empty array of Node inputs");
  }
  const emit = opts.emit ?? "when_all";
  const maxPending = opts.maxPending ?? 1_000_000;
  return new Node(
    { combinator: "combineLatest", params: { whenAll: emit === "when_all", maxPending } },
    nodes,
  );
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
function normalizeColumns(columns: number | number[]): number[] {
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
  plan: Array<[number, number]> | null;
  reducer: Node | null;
}

// Causal windowed downsample. Translates opts -> codes, mirroring dag.py's
// Resample compile branch.
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

  const inputs = opts.driver && isNode(opts.driver) ? [node, opts.driver] : [node];
  return new Node({ combinator: "resample", params }, inputs);
}
