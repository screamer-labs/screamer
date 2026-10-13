// split: Python's split (screamer/streams.py). The rules and the partition are
// C++ (include/screamer/streams/split.h, bound in wasm/split_bindings.cpp),
// shared with Python; this module reads the arguments into WASM buffers and
// cuts the result into one pair per source.
import { current } from "./index.js";
import { asNd, type NdArray } from "./ndarray.js";
import { wasmCall } from "./errors.js";
import { aKind, flattenRows, throwWrongKind, toF64Checked } from "./kinds.js";

export interface SplitOptions {
  /** The index of each event, partitioned with the values. Absent, each output's index is null. */
  index?: ArrayLike<number>;
  /** The number of outputs. Default max(sources) + 1; set it to include sources with no events. */
  n?: number;
}

const WHO = "split";

// An array argument as float64 with its number of dimensions: an NdArray as it
// is, an array of rows flattened (C++ then rejects the second dimension, as
// Python does), anything else array-like item by item (null is NaN when
// `values`, a string a TypeError).
function column(x: unknown, where: string, values: boolean): { data: Float64Array; ndim: number } {
  const nd = asNd(x);
  if (nd) return { data: nd.data, ndim: nd.shape.length };
  if (Array.isArray(x) && x.length > 0 && x[0] !== null && typeof x[0] === "object" && !(x[0] instanceof Date)) {
    const { data } = flattenRows(x, WHO, where);
    return { data, ndim: 2 };
  }
  if (Array.isArray(x) || (ArrayBuffer.isView(x) && !(x instanceof DataView)))
    return { data: toF64Checked(x as ArrayLike<unknown>, WHO, (i) => `item ${i} of ${where}`, values), ndim: 1 };
  throwWrongKind(WHO, where, x, "an array, a typed array or an NdArray");
}

// The options as a plain object of known keys; n a number (a boolean as 0 or
// 1, as Python takes a bool). C++ checks that n is a whole number.
function options(opts: unknown): { index?: ArrayLike<number>; n?: number } {
  const o = opts ?? {};
  const proto = typeof o === "object" ? Object.getPrototypeOf(o) : undefined;
  if (proto !== Object.prototype && proto !== null)
    throw new TypeError(`${WHO}: the options are ${aKind(opts)}, expected an object such as { n: 3 }`);
  for (const key of Object.keys(o))
    if (key !== "index" && key !== "n") throw new TypeError(`${WHO}: unknown option '${key}'`);
  const { index, n } = o as SplitOptions;
  if (n !== undefined && n !== null && typeof n !== "number" && typeof n !== "boolean")
    throw new TypeError(`${WHO}: n must be a number, got ${aKind(n)}`);
  return { index: index ?? undefined, n: n == null ? undefined : Number(n) };
}

/**
 * Partition a tagged stream into one stream per source, the inverse of a merge.
 *
 * Event i goes to output `sources[i]`, which holds its value (and its index
 * when `index` is given) in event order. A source is a whole number from 0 to
 * 2**32 - 1; any other source is a TypeError. There are `n` outputs, by
 * default max(sources) + 1; an `n` that would drop an event (n <= max(sources))
 * is a RangeError.
 *
 * Returns `[values, index]` pairs: `values` a Float64Array, `index` a
 * Float64Array, or null without an index. Python's `split` gives the same
 * values.
 */
export function split(
  values: Float64Array | ArrayLike<number | null> | NdArray,
  sources: ArrayLike<number> | NdArray,
  opts: SplitOptions | null = {},
): Array<[Float64Array, Float64Array | null]> {
  const M = current();
  const o = options(opts);
  const v = column(values, "values", true);
  const s = column(sources, "sources", true);
  const kn = asNd(o.index);
  const k = o.index === undefined ? null : kn ? kn.data
    : toF64Checked(o.index as ArrayLike<unknown>, WHO, (i) => `item ${i} of index`);
  const total = v.data.length;
  const ptrs: number[] = [];
  const buffer = (data: Float64Array | null) => {
    const p = M.allocF64(Math.max(1, data ? data.length : total));
    ptrs.push(p);
    if (data) M.viewF64(p, data.length).set(data);
    return p;
  };
  try {
    const vp = buffer(v.data), sp = buffer(s.data), kp = k ? buffer(k) : 0;
    const outV = buffer(null), outK = k ? buffer(null) : 0;
    const counts = wasmCall(() => M.splitInto(
      vp, v.ndim, total, sp, s.ndim, s.data.length, k !== null, kn ? kn.shape.length : 1, kp,
      k ? k.length : 0,
      o.n !== undefined, o.n ?? 0, outV, outK)) as Float64Array;
    // Read the heap after the call: the views are valid until the next allocation.
    const allV = M.viewF64(outV, total), allK = k ? M.viewF64(outK, total) : null;
    const out: Array<[Float64Array, Float64Array | null]> = [];
    let at = 0;
    for (let i = 0; i < counts.length; i++) {
      const c = counts[i];
      out.push([allV.slice(at, at + c), allK ? allK.slice(at, at + c) : null]);
      at += c;
    }
    return out;
  } finally {
    for (const p of ptrs) M.freeBuf(p);
  }
}
