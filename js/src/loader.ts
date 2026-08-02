// Loads the Phase 2 Embind module once. `npm run build:wasm` copies the
// dev (separate-file) screamer.mjs + screamer.wasm into src/generated/;
// `npm run build:wasm:single` instead copies the self-contained
// screamer.single.mjs (wasm embedded as base64) to this same path, and
// `npm run build` copies whichever is currently there into dist/generated/
// so the packaged tarball needs no separate .wasm asset to resolve.
import initModule from "./generated/screamer.mjs";
import { normalizeError } from "./errors.js";

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

// The Embind module, when built without WASM exception-catching support for a
// given throw site, surfaces a C++ exception as a raw numeric pointer instead
// of a JS Error. `wrapOp`'s `call()` catches and normalizes that for *eval*
// errors (invalid input during a push), but ctor-time validation errors (e.g.
// `new M.RollingMean(0, ...)` rejecting an out-of-range window size) happen
// before a ScreamerOp even exists, outside any try/catch the generated
// factories provide. Wrap every Embind class constructor once, here, so a
// raw pointer thrown mid-construction is normalized the same way.
function wrapCtors(M: Screamer): Screamer {
  for (const key of Object.keys(M)) {
    if (!/^[A-Z]/.test(key)) continue; // skip plain helper functions (allocF64, ...)
    const Ctor = M[key];
    if (typeof Ctor !== "function") continue;
    M[key] = new Proxy(Ctor, {
      construct(target, args) {
        try {
          return Reflect.construct(target, args);
        } catch (e) {
          throw normalizeError(e);
        }
      },
    });
  }
  return M;
}

let cached: Promise<Screamer> | null = null;
export function init(): Promise<Screamer> {
  if (!cached) cached = (initModule() as Promise<Screamer>).then(wrapCtors);
  return cached;
}
