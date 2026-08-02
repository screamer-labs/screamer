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
export * from "./generated/ops.js";
