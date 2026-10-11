import { current } from "./index.js";
import { normalizeError } from "./errors.js";
import { checkNd, type NdArray } from "./ndarray.js";

/**
 * Reshape data by a layout string: `T` (time), `A` (asset), `F` (field), with
 * an optional `->` and the wanted result, as in Python's `screamer.reshape`.
 * Returns the same buffer with a new shape, or a copy when axes are reordered.
 */
export function reshape(x: NdArray | Float64Array | number[], layout: string): NdArray {
  const M = current();
  const nd: NdArray = (x as NdArray).shape
    ? (x as NdArray) : { data: Float64Array.from(x as ArrayLike<number>), shape: [(x as ArrayLike<number>).length] };
  checkNd(nd);
  let plan: { source: number[]; shape: number[] };
  try { plan = M.layoutPlan(layout, nd.shape); } catch (e) { throw normalizeError(e); }
  const kept = plan.source.filter((s) => s >= 0);
  if (kept.every((s, i) => i === 0 || s > kept[i - 1])) return { data: nd.data, shape: plan.shape };
  const inStride = nd.shape.map((_, d) => nd.shape.slice(d + 1).reduce((a, b) => a * b, 1));
  const stride = plan.source.map((s) => (s >= 0 ? inStride[s] : 0));
  const out = new Float64Array(nd.data.length);
  const idx = new Array(plan.shape.length).fill(0);
  for (let o = 0; o < out.length; o++) {
    let off = 0;
    for (let d = 0; d < idx.length; d++) off += idx[d] * stride[d];
    out[o] = nd.data[off];
    for (let d = idx.length - 1; d >= 0; d--) { if (++idx[d] < plan.shape[d]) break; idx[d] = 0; }
  }
  return { data: out, shape: plan.shape };
}
