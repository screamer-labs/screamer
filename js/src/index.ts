import { init as loadInit, type Screamer } from "./loader.js";

let M: Screamer | null = null;

export function current(): Screamer {
  if (!M) throw new Error("call `await ready()` before constructing ops");
  return M;
}

export async function ready(): Promise<void> {
  M = await loadInit();
}

export { init } from "./loader.js";
export type { ScreamerOp } from "./runtime.js";
export { toNested } from "./ndarray.js";
export type { NdArray } from "./ndarray.js";
export { Node, Input, isNode } from "./node.js";
export { Pipeline } from "./pipeline.js";
export type { Output } from "./pipeline.js";
export {
  combineLatest,
  resample,
  select,
  dropna,
  filter,
  delay,
  merge,
} from "./combinators.js";
export type {
  CombineLatestOpts,
  DropnaOpts,
  ResampleOpts,
  ResampleParams,
} from "./combinators.js";
export {
  RESAMPLE_AGG_CODE,
  RESAMPLE_MODE_CODE,
  RESAMPLE_FILL_CODE,
  RESAMPLE_LABEL_CODE,
  PLAN_AGG_CODE,
  BAR_AGG_FIXED_PLANS,
  CODES,
} from "./codes.js";
export * from "./generated/ops.js";
