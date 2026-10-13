// Loads the Phase 2 Embind module once. `npm run build:wasm` copies the
// dev (separate-file) screamer.mjs + screamer.wasm into src/generated/;
// `npm run build:wasm:single` instead copies the self-contained
// screamer.single.mjs (wasm embedded as base64) to this same path, and
// `npm run build` copies whichever is currently there into dist/generated/
// (and dist/cjs/generated/ for the CJS build) so the packaged tarball needs
// no separate .wasm asset to resolve.
import { normalizeError, setExceptionHelpers } from "./errors.js";

// screamer.mjs is single-file ESM (`export default Module`), in both the
// ESM and CJS build. A literal `import(...)` here would be fine for the ESM
// build, but tsc's CommonJS emit downlevels a literal dynamic `import()`
// into a `require()` wrapped in a promise, and `require()` of an ESM-only
// module throws `ERR_REQUIRE_ESM`. Indirecting through `Function` hides the
// import expression from tsc's static downlevel transform, so both builds
// execute a real dynamic `import()` at runtime; Node has supported
// `import()` from CommonJS modules since 12.x. The specifier still resolves
// relative to this file (loader.js / loader.cjs), not the caller's cwd.
const dynamicImport = new Function("specifier", "return import(specifier)") as (
  specifier: string,
) => Promise<{ default: () => Promise<Screamer> }>;

export interface RawOp {
  nIn(): number; nOut(): number;
  evalInto(inPtr: number, outPtr: number): void;
  // One C++ pass over `rows` row-major events (each row is nIn doubles at
  // inPtr, writing nOut at outPtr): the whole array is processed without
  // recrossing the JS/WASM boundary per element.
  evalBatchInto(inPtr: number, outPtr: number, rows: number): void;
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

let cached: Promise<Screamer> | null = null;
export function init(): Promise<Screamer> {
  if (!cached) {
    cached = dynamicImport("./generated/screamer.mjs")
      .then((m) => m.default())
      .then((M: Screamer) => {
        setExceptionHelpers(M as any);
        // A view of the WASM heap, made in JavaScript: the Embind viewF64 makes
        // a memory view through emval on every call, which costs a handle per
        // event on the streaming path. HEAPF64 is replaced when memory grows,
        // so it is read on every call; like any heap view, the result is valid
        // until the next call that may allocate.
        M.viewF64 = (p: number, n: number) => new Float64Array((M as any).HEAPF64.buffer, p, n);
        return wrapCtors(M);
      });
  }
  return cached;
}
