// Loads the Embind module once. `npm run build:wasm` copies the dev
// (separate-file) screamer.mjs + screamer.wasm into src/generated/;
// `npm run build:wasm:single` instead copies the self-contained
// screamer.single.mjs (wasm embedded as base64) to this same path, and
// `npm run build` copies whichever is currently there into dist/generated/,
// which both the ES and the CommonJS build load (wasm_module.ts), so the
// packaged tarball needs no separate .wasm asset to resolve.
import { normalizeError, setExceptionHelpers, OUT_OF_MEMORY } from "./errors.js";
import { loadModule } from "./wasm_module.js";

export interface RawOp {
  nIn(): number; nOut(): number;
  evalInto(inPtr: number, outPtr: number): void;
  reset(): void; delete(): void;
  // Present only on dynamic-width reducers (see wrapReducerOp): one event
  // reduces an (assets, nIn) block, so the asset count travels per call.
  reduceInto?(inPtr: number, outPtr: number, groups: number): void;
  reduceBatchInto?(inPtr: number, outPtr: number, rows: number, groups: number): void;
  // The same with one price per group (groups doubles per event at pricePtr).
  reducePricedInto?(inPtr: number, pricePtr: number, outPtr: number, groups: number): void;
  reducePricedBatchInto?(inPtr: number, pricePtr: number, outPtr: number, rows: number, groups: number): void;
  // Present only on row ops (see wrapRowOp): one event is a row of k assets
  // with fieldsIn() values each, so k travels per call.
  mapRowInto?(inPtr: number, outPtr: number, k: number): void;
  mapRowsInto?(inPtr: number, outPtr: number, rows: number, k: number, packed: boolean): void;
  fieldsIn?(): number;
  fieldsOut?(): number;
}
export interface Screamer {
  allocF64(n: number): number; freeBuf(p: number): void;
  /**
   * Bytes the WASM allocator has handed out and not taken back, for leak
   * tests. It walks the allocator's free lists, so its cost grows with the
   * number of free blocks: keep it off hot paths.
   * @internal
   */
  heapInUse(): number;
  viewF64(p: number, n: number): Float64Array;
  VectorDouble: new () => { push_back(x: number): void; delete(): void };
  [name: string]: any;
}

// A C++ constructor can reject its arguments (e.g. `new M.RollingMean(0, ...)`
// rejects a window size of 0) before a ScreamerOp exists, outside the narrow
// try blocks around the op's own calls into WASM. Wrap every Embind class
// constructor once, here, so the construction alone is normalized.
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

// The most doubles one allocation may ask for: the 4 GB wasm32 address space.
const MAX_DOUBLES = 2 ** 29 - 1;

let cached: Promise<Screamer> | null = null;
export function init(): Promise<Screamer> {
  if (!cached) {
    cached = loadModule()
      .then((m) => m.default())
      .then((M: Screamer) => {
        setExceptionHelpers(M as any);
        // A view of the WASM heap, made in JavaScript: the Embind viewF64 makes
        // a memory view through emval on every call, which costs a handle per
        // event on the streaming path. HEAPF64 is replaced when memory grows,
        // so it is read on every call; like any heap view, the result is valid
        // until the next call that may allocate.
        M.viewF64 = (p: number, n: number) => new Float64Array((M as any).HEAPF64.buffer, p, n);
        // n doubles of heap. A count past what the heap can hold is the
        // out-of-memory RangeError before the call (Embind would wrap a
        // number past 2**32); a failed malloc is std::bad_alloc in C++.
        const alloc = M.allocF64;
        M.allocF64 = (n: number): number => {
          if (!(n >= 0 && n <= MAX_DOUBLES)) throw new RangeError(OUT_OF_MEMORY);
          try { return alloc(n); } catch (e) { throw normalizeError(e); }
        };
        return wrapCtors(M);
      });
  }
  return cached;
}
