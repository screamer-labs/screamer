// split: Python's split (screamer/streams.py). The rules and the partition are
// C++ (include/screamer/streams/split.h, bound in wasm/split_bindings.cpp),
// shared with Python; this module reads the arguments into WASM buffers and
// cuts the result into one pair per source.
import { current } from "./index.js";
import { asNd, type NdArray } from "./ndarray.js";
import { wasmCall } from "./errors.js";
import { aKind, fillF64Checked, flattenRows, throwWrongKind } from "./kinds.js";

export interface SplitOptions {
  /** The index of each event, partitioned with the values. Absent, each output's index is null. */
  index?: ArrayLike<number>;
  /** The number of outputs. Default max(sources) + 1; set it to include sources with no events. */
  n?: number;
}

/**
 * One output: its rows of values (an NdArray when values have rows), and its
 * index or null. An indexed stream in the shape combineLatest and a Pipeline
 * take, the counterpart of Python's `(values, index)` tuple.
 */
export interface SplitOutput { values: Float64Array | NdArray; index: Float64Array | null }

const WHO = "split";

// An argument read for the heap: float64 data to copy, or items to convert
// straight into the heap (one copy fewer), and the shape.
interface Arg { data: Float64Array | null; items: ArrayLike<unknown> | null; shape: number[] }

const isRows = (x: unknown[]) =>
  x.length > 0 && x[0] !== null && typeof x[0] === "object" && !(x[0] instanceof Date);

// An NdArray as it is, an array of rows flattened (C++ checks the rank: rows
// are a shape error for sources and the index), a Float64Array as it is, and
// any other array or typed array item by item, later.
function read(x: unknown, where: string): Arg {
  const nd = asNd(x);
  if (nd) return { data: nd.data, items: null, shape: nd.shape };
  if (Array.isArray(x) && isRows(x)) {
    const { data, T, K } = flattenRows(x, WHO, where);
    return { data, items: null, shape: [T, K] };
  }
  if (x instanceof Float64Array) return { data: x, items: null, shape: [x.length] };
  if (Array.isArray(x) || (ArrayBuffer.isView(x) && !(x instanceof DataView))) {
    const items = x as ArrayLike<unknown>;
    return { data: null, items, shape: [items.length] };
  }
  throwWrongKind(WHO, where, x, "an array, a typed array or an NdArray");
}

const size = (a: Arg) => (a.data ? a.data.length : a.items!.length);

// The sources item by item, each kind checked before it becomes a number, as
// Python reads a list: a number goes to C++ (which checks that it is whole and
// in range); a boolean, null or any other kind is a TypeError.
function fillSources(items: ArrayLike<unknown>, out: Float64Array): void {
  for (let i = 0; i < items.length; i++) {
    const v = items[i];
    if (typeof v === "number") { out[i] = v; continue; }
    const kind = typeof v === "boolean" ? "a bool" : v === null ? "null" : aKind(v);
    throw new TypeError(`${WHO}: a source must be a whole number, got ${kind}`);
  }
}

// The options as a plain object of known keys; n a number (a boolean is a
// wrong kind, as in Python). C++ checks that n is a whole number.
function options(opts: unknown): { index?: unknown; n?: number } {
  const o = opts ?? {};
  const proto = typeof o === "object" ? Object.getPrototypeOf(o) : undefined;
  if (proto !== Object.prototype && proto !== null)
    throw new TypeError(`${WHO}: the options are ${aKind(opts)}, expected an object such as { n: 3 }`);
  for (const key of Object.keys(o))
    if (key !== "index" && key !== "n") throw new TypeError(`${WHO}: unknown option '${key}'`);
  const { index, n } = o as SplitOptions;
  if (n !== undefined && n !== null && typeof n !== "number")
    throw new TypeError(`${WHO}: n must be an integer, got ${typeof n === "boolean" ? "boolean" : aKind(n)}`);
  return { index: index ?? undefined, n: n == null ? undefined : n };
}

/**
 * Partition a tagged stream into one stream per source, the inverse of a merge.
 *
 * Event i goes to output `sources[i]`, which holds its value (and its index
 * when `index` is given) in event order. `values` may have rows: an NdArray
 * of shape `[T, ...]` or T rows, each event one row. A source is a whole
 * number (a boolean, a fractional or a non-finite source is a TypeError) from
 * 0 to 2147483646 (else a RangeError). There are `n` outputs, by default
 * max(sources) + 1; an `n` that would drop an event (n <= max(sources)) is a
 * RangeError.
 *
 * Returns `{ values, index }` objects: `values` a Float64Array for 1-D values, or
 * an NdArray of shape `[rows, ...]` that keeps the row shape; `index` a
 * Float64Array, or null without an index. Python's `split` gives the same
 * values.
 */
export function split(
  values: Float64Array | ArrayLike<number | null> | NdArray | ArrayLike<ArrayLike<number | null>>,
  sources: ArrayLike<number> | NdArray,
  opts: SplitOptions | null = {},
): SplitOutput[] {
  const M = current();
  const o = options(opts);
  const v = read(values, "values");
  const s = read(sources, "sources");
  const k = o.index === undefined ? null : read(o.index, "index");
  const rows = v.shape[0] ?? 0;
  const rowShape = v.shape.slice(1);
  const width = rowShape.reduce((a, b) => a * b, 1);
  const ptrs: number[] = [];
  const alloc = (n: number) => { const p = M.allocF64(Math.max(1, n)); ptrs.push(p); return p; };
  try {
    // Every buffer first: an allocation may grow the heap, which replaces its views.
    const vp = alloc(size(v)), sp = alloc(size(s)), kp = k ? alloc(size(k)) : 0;
    const outV = alloc(rows * width), outK = k ? alloc(rows) : 0;
    if (v.data) M.viewF64(vp, v.data.length).set(v.data);
    else fillF64Checked(v.items!, M.viewF64(vp, size(v)), WHO, (i) => `item ${i} of values`, true);
    if (s.data) M.viewF64(sp, s.data.length).set(s.data);
    else fillSources(s.items!, M.viewF64(sp, size(s)));
    if (k) {
      if (k.data) M.viewF64(kp, k.data.length).set(k.data);
      else fillF64Checked(k.items!, M.viewF64(kp, size(k)), WHO, (i) => `item ${i} of index`);
    }
    const counts = wasmCall(() => M.splitInto(
      vp, v.shape.length, rows, width, sp, s.shape.length, size(s),
      k !== null, k ? k.shape.length : 1, kp, k ? size(k) : 0,
      o.n !== undefined, o.n ?? 0, outV, outK)) as Float64Array;
    // Read the heap after the call: the views are valid until the next allocation.
    const allV = M.viewF64(outV, rows * width), allK = k ? M.viewF64(outK, rows) : null;
    const out: SplitOutput[] = [];
    let at = 0;
    for (let i = 0; i < counts.length; i++) {
      const c = counts[i];
      const data = allV.slice(at * width, (at + c) * width);
      out.push({ values: v.shape.length > 1 ? { data, shape: [c, ...rowShape] } : data,
        index: allK ? allK.slice(at, at + c) : null });
      at += c;
    }
    return out;
  } finally {
    for (const p of ptrs) M.freeBuf(p);
  }
}
