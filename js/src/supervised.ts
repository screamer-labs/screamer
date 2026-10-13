// forecastPairs: Python's forecast_pairs. The graph is defined once in C++
// (include/screamer/dag/forecast_pairs.h, bound in wasm/dag_embind.h); this
// module checks the options, builds that graph as one Pipeline node, and
// splits the output columns.
import { current } from "./index.js";
import { asNd, type NdArray } from "./ndarray.js";
import { Input, Node, isNode } from "./node.js";
import { Pipeline, type Output } from "./pipeline.js";
import { wasmCall } from "./errors.js";
import { aKind, flattenRows, toF64Checked } from "./kinds.js";

export interface ForecastPairsOptions {
  /** Lag X by this many events. Exactly one of `count` and `duration`. */
  count?: number;
  /** Delay X's index by this many index units and pair each y with the last delayed X. */
  duration?: number;
  /** Drop a row whose shifted features or target is NaN. Default `false`. */
  dropna?: boolean;
}

/** Feature values: a 1-D array, a `[T, K]` NdArray, or T rows of K values. */
export type ForecastValues = Float64Array | ArrayLike<number | null> | NdArray | ArrayLike<ArrayLike<number | null>>;

/** A stream for duration mode: values and their index (row numbers when absent). */
export interface ForecastStream {
  values: ForecastValues;
  index?: ArrayLike<number>;
}

const WHO = "forecastPairs";

// The graph parameters of one call, checked.
interface Horizon { count?: number; duration?: number; dropna: boolean }

// A count or duration: an integer setting (a string or a boolean is a wrong
// kind, as for every setting), truncated as delay() truncates its duration.
function horizonValue(name: string, v: unknown): number | undefined {
  if (v === undefined || v === null) return undefined;
  if (typeof v !== "number")
    throw new TypeError(`${WHO}: ${name} must be an integer, got ${typeof v === "boolean" ? "boolean" : aKind(v)}`);
  return Math.trunc(v);
}

// The options as a plain object of known keys; the engine checks the horizon.
function horizon(opts: unknown): Horizon {
  const o = opts ?? {};
  const proto = typeof o === "object" ? Object.getPrototypeOf(o) : undefined;
  if (proto !== Object.prototype && proto !== null)
    throw new TypeError(`${WHO}: the options are ${aKind(opts)}, expected an object such as { count: 5 }`);
  for (const key of Object.keys(o))
    if (key !== "count" && key !== "duration" && key !== "dropna")
      throw new TypeError(`${WHO}: unknown option '${key}'`);
  const { count: c, duration: d, dropna } = o as ForecastPairsOptions;
  if (dropna !== undefined && typeof dropna !== "boolean")
    throw new TypeError(`${WHO}: dropna must be a boolean, got ${aKind(dropna)}`);
  const count = horizonValue("count", c);
  const duration = horizonValue("duration", d);
  const M: any = current();
  wasmCall(() => M.checkForecastPairsHorizon(count !== undefined, count ?? 0,
                                             duration !== undefined, duration ?? 0));
  return { count, duration, dropna: dropna ?? false };
}

// One forecastPairs graph node over X (any width) and y.
function pairsNode(X: Node, y: Node, h: Horizon): Node {
  return new Node({ combinator: "forecastPairs", params: { ...h } }, [X, y]);
}

interface Block { data: Float64Array; ndim: number; T: number; K: number; wide: boolean }

// Values as row-major data: a flat array (null is NaN, a string a TypeError),
// an NdArray, or an array of rows (flattenRows). The engine checks the rank.
function block(v: unknown, where: string): Block {
  const nd = asNd(v);
  if (nd) {
    const ndim = nd.shape.length;
    return { data: nd.data, ndim, T: nd.shape[0] ?? 0, K: ndim === 2 ? nd.shape[1] : 1, wide: ndim === 2 };
  }
  if (Array.isArray(v) && v.length > 0 && v[0] !== null && typeof v[0] === "object") {
    const { data, T, K } = flattenRows(v, WHO, where);
    return { data, ndim: 2, T, K, wide: true };
  }
  if (Array.isArray(v) || (ArrayBuffer.isView(v) && !(v instanceof DataView))) {
    const data = toF64Checked(v as ArrayLike<unknown>, WHO, (i) => `item ${i} of ${where}`, true);
    return { data, ndim: 1, T: data.length, K: 1, wide: false };
  }
  throw new TypeError(`${WHO}: ${where} is ${aKind(v)}, expected an array, a typed array or an NdArray`);
}

const isStream = (v: unknown): v is ForecastStream =>
  v !== null && typeof v === "object" && !Array.isArray(v) && !ArrayBuffer.isView(v) &&
  asNd(v) === null && "values" in (v as object);

function asFeed(b: Block, index?: ArrayLike<number>) {
  const values = b.wide ? ({ data: b.data, shape: [b.T, b.K] } as NdArray) : b.data;
  return index === undefined ? values : { values, index: index as Float64Array | number[] };
}

/**
 * Pair features with a target `count` events (or `duration` index units) ahead.
 *
 * Row t of the result holds the features from `count` events before t and the
 * target at t, so a model learns to predict `count` ahead. X is delayed and y
 * passes through. The first `count` rows of the shifted features are NaN;
 * `dropna: true` drops every row with a NaN. Pass exactly one of `count` and
 * `duration`.
 *
 * - Count mode: X is a 1-D array, a `[T, K]` NdArray or T rows, and y one
 *   column of the same length.
 * - Duration mode: X and y are `{ values, index }` streams; each y is paired
 *   with the last X whose index plus `duration` is at or before y's index.
 * - With `Input` nodes, returns a Node whose output row is `[X_shifted..., y]`,
 *   for a Pipeline (and its live sessions). X may have any width.
 *
 * Returns `[X_shifted, y]`: X_shifted has X's rank (a Float64Array, or an
 * NdArray of shape `[rows, K]`). Python's `forecast_pairs` gives the same values.
 */
export function forecastPairs(X: Node, y: Node, opts: ForecastPairsOptions): Node;
export function forecastPairs(
  X: ForecastValues | ForecastStream, y: ForecastValues | ForecastStream, opts: ForecastPairsOptions,
): [Float64Array | NdArray, Float64Array];
export function forecastPairs(X: unknown, y: unknown, opts: ForecastPairsOptions): unknown {
  const h = horizon(opts);
  if (isNode(X) || isNode(y)) {
    if (!(isNode(X) && isNode(y)))
      throw new TypeError(`${WHO}: pass X and y both as Nodes to build a graph`);
    return pairsNode(X, y, h);
  }
  const M: any = current();
  let bx: Block, by: Block, ix: ArrayLike<number> | undefined, iy: ArrayLike<number> | undefined;
  if (h.duration !== undefined) {
    if (!isStream(X) || !isStream(y))
      throw new TypeError(`${WHO}: duration mode needs X and y as { values, index } streams`);
    bx = block(X.values, "the values of X");
    by = block(y.values, "the values of y");
    ix = X.index;
    iy = y.index;
  } else {
    if (isStream(X) || isStream(y))
      throw new TypeError(`${WHO}: count mode takes X and y as values (an array, a typed array or an NdArray)`);
    bx = block(X, "X");
    by = block(y, "y");
  }
  wasmCall(() => M.checkForecastPairsInputs(h.count !== undefined, bx.ndim, bx.T, bx.K,
                                            by.ndim, by.T, by.K));
  const xi = Input("X"), yi = Input("y");
  const p = new Pipeline([xi, yi], [pairsNode(xi, yi, h)]);
  let out: Output;
  try {
    out = p({ X: asFeed(bx, ix), y: asFeed(by, iy) }) as Output;
  } finally {
    p.dispose();
  }
  return split(out, bx.K, bx.wide);
}

// The output [X_shifted..., y] as [X_shifted, y].
function split(out: Output, K: number, wide: boolean): [Float64Array | NdArray, Float64Array] {
  const nd = asNd(out.values);
  const data = nd ? nd.data : (out.values as Float64Array);
  const W = nd ? nd.shape[1] : 1;
  const rows = out.index.length;
  const xs = new Float64Array(rows * K), ys = new Float64Array(rows);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < K; c++) xs[r * K + c] = data[r * W + c];
    ys[r] = data[r * W + K];
  }
  return [wide ? ({ data: xs, shape: [rows, K] } as NdArray) : xs, ys];
}
