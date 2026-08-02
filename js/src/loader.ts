// Loads the Phase 2 Embind module once. The .mjs + .wasm are copied into
// src/generated/ by `npm run build:wasm`.
import initModule from "./generated/screamer.mjs";

export interface RawOp {
  nIn(): number; nOut(): number;
  evalInto(inPtr: number, outPtr: number): void;
  reset(): void; delete(): void;
}
export interface Screamer {
  allocF64(n: number): number; freeBuf(p: number): void;
  viewF64(p: number, n: number): Float64Array;
  VectorDouble: new () => { push_back(x: number): void; delete(): void };
  [name: string]: any;
}
let cached: Promise<Screamer> | null = null;
export function init(): Promise<Screamer> {
  if (!cached) cached = initModule() as Promise<Screamer>;
  return cached;
}
