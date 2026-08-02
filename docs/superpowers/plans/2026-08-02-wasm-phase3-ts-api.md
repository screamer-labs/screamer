# screamer.js Phase 3: Ergonomic TypeScript API - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A typed, ergonomic TypeScript API over the Phase 2 WASM substrate: `Name(...args)` factories returning a callable that dispatches across the four input regimes (scalar, batch, lazy, async), container-preserving, with a `using`/dispose lifecycle - parity-faithful to Python's `Op(config)(data)`, for all 226 point ops.

**Architecture:** A single generic runtime wrapper turns any raw Embind op (via `nIn`/`nOut`/`evalInto`/`reset`/`delete` + the heap helpers) into a polymorphic callable. Per-op typed factory functions are code-generated from the Phase 2 manifest (C++ ctor types -> TS types) plus arg names/defaults pulled from the Python module signatures. The package loads the WASM module once (`await init()`), exposes the factories, and normalizes errors and object lifetimes.

**Tech Stack:** TypeScript, the Phase 2 `screamer.mjs`/`.wasm` build, Node (`node:test` + tsx or vitest) for tests, the Python screamer package (signatures + parity oracle).

## Global Constraints

- JS-side only. MUST NOT modify any Python binding, operator header, C++ compute code, or the Phase 2 generators/runtime/build. The Python suite stays 6930/2.
- Parity with Python semantics: `Name(...args)` is a call (no `new`); `op(number)` streams statefully; `op(Float64Array | number[])` is a self-contained batch that resets internally; array-in -> array-out, typed-in -> typed-out (container-preserving); iterable -> lazy generator; async-iterable -> async generator. Multi-output batch returns a `{ data: Float64Array, shape: number[] }` ndarray-lite.
- The typed factories are code-generated from `devtools/wasm/wasm_manifest.json` + Python signatures; they are not hand-maintained. A freshness gate enforces this.
- Package name `@screamer-labs/screamer`, ESM-first with a CJS build and shipped `.d.ts`. Shared versioning with Python (do not set a version here; packaging/publish is Phase 6).
- No em-dashes in prose or comments (ASCII hyphens). No version file edits.

## The Phase 2 substrate (what this wraps)

`import init from "screamer.mjs"; const M = await init();` gives a module where every op is `new M[Name](...args)` (positional args; NaN sentinel for a missing `optional<double>` slot; `M.VectorDouble` for a `vector<double>` ctor). Each instance exposes `nIn(): number`, `nOut(): number`, `evalInto(inPtr, outPtr): void`, `reset(): void`, `delete(): void`. Heap: `M.allocF64(n): ptr`, `M.freeBuf(ptr)`, `M.viewF64(ptr, n): Float64Array` (a view; `.set([...])` writes). One event = write `nIn` doubles at inPtr, `evalInto`, read `nOut` doubles at outPtr. (See `wasm/smoke/smoke.mjs` for the exact calling convention.)

## File Structure

- Create `js/package.json`, `js/tsconfig.json` - the TS package (name `@screamer-labs/screamer`, ESM + CJS, `exports` map).
- Create `js/src/loader.ts` - `init()`: loads the WASM module (default separate `.wasm` via `import.meta.url`), returns the bound namespace; caches the module.
- Create `js/src/runtime.ts` - `wrapOp(M, RawClass, ctorArgs)`: the generic polymorphic callable (the four regimes, container-preserving, ndarray-lite, lifecycle). The core of the phase.
- Create `js/src/ndarray.ts` - the `{ data, shape }` ndarray-lite type + `toNested()`.
- Create `js/src/errors.ts` - map raw Embind/kernel errors to `TypeError`/`RangeError` with parity messages.
- Create `devtools/wasm/gen_ts_api.py` - emits `js/src/generated/ops.ts` (typed factories) + `js/src/generated/ops.d.ts` from the manifest + Python signatures.
- Create `js/src/generated/ops.ts` + `ops.d.ts` - generated, committed.
- Create `js/src/index.ts` - re-exports `init` + all factories.
- Create `js/test/*.test.ts` - the JS suite (regimes, container-preserving, lifecycle, errors, parity).
- Create `tests/test_ts_api_fresh.py` - freshness gate for the generated TS.

---

### Task 1: TS package skeleton + WASM loader

**Files:**
- Create: `js/package.json`, `js/tsconfig.json`, `js/src/loader.ts`, `js/src/index.ts`
- Test: `js/test/loader.test.ts`

**Interfaces:**
- Produces: `init(): Promise<Screamer>` where `Screamer` is the module handle exposing the raw op classes and heap helpers. `index.ts` re-exports `init`.

- [ ] **Step 1: Write `js/package.json`**

```json
{
  "name": "@screamer-labs/screamer",
  "type": "module",
  "exports": { ".": { "types": "./dist/index.d.ts", "import": "./dist/index.js", "require": "./dist/index.cjs" } },
  "files": ["dist"],
  "scripts": {
    "build:wasm": "bash ../wasm/build-wasm.sh && cp ../wasm/build/screamer.mjs ../wasm/build/screamer.wasm src/generated/",
    "gen": "python3 ../devtools/wasm/gen_ts_api.py",
    "build": "tsc -p tsconfig.json",
    "test": "node --import tsx --test test/*.test.ts"
  },
  "devDependencies": { "typescript": "^5.5.0", "tsx": "^4.0.0" }
}
```

- [ ] **Step 2: Write `js/tsconfig.json`**

```json
{ "compilerOptions": { "target": "ES2022", "module": "ESNext", "moduleResolution": "bundler",
  "declaration": true, "outDir": "dist", "strict": true, "lib": ["ES2022"], "types": ["node"] },
  "include": ["src"] }
```

- [ ] **Step 3: Write `js/src/loader.ts`**

```ts
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
```

- [ ] **Step 4: Copy the built module in and test load**

Run: `cd js && npm install && npm run build:wasm && node --import tsx --test test/loader.test.ts`
`loader.test.ts`: `import { init } from "../src/index.ts"; ... const M = await init(); assert(typeof M.RollingMean === "function"); assert(typeof M.allocF64 === "function");`
Expected: pass. (`src/generated/screamer.mjs`/`.wasm` are build artifacts - gitignore them; they are regenerated by `build:wasm`.)

- [ ] **Step 5: Commit**

```bash
git add js/package.json js/tsconfig.json js/src/loader.ts js/src/index.ts js/test/loader.test.ts js/.gitignore
git commit -m "feat(js): TS package skeleton + WASM module loader"
```

---

### Task 2: The generic polymorphic runtime wrapper

**Files:**
- Create: `js/src/ndarray.ts`, `js/src/errors.ts`, `js/src/runtime.ts`
- Test: `js/test/runtime.test.ts`

**Interfaces:**
- Consumes: `Screamer`, `RawOp` from loader.
- Produces: `wrapOp(M: Screamer, raw: RawOp): ScreamerOp` where `ScreamerOp` is a callable with signatures for each regime, plus `.reset()`, `.dispose()`, `[Symbol.dispose]()`.

- [ ] **Step 1: `js/src/ndarray.ts`**

```ts
export interface NdArray { data: Float64Array; shape: number[]; }
export function toNested(a: NdArray): number[] | number[][] {
  if (a.shape.length <= 1) return Array.from(a.data);
  const [rows, cols] = a.shape;
  const out: number[][] = [];
  for (let r = 0; r < rows; r++) out.push(Array.from(a.data.subarray(r * cols, r * cols + cols)));
  return out;
}
```

- [ ] **Step 2: `js/src/errors.ts`**

```ts
// Kernel ctor/eval errors surface from Embind as generic Error objects whose
// message carries the C++ what(). Map the common shapes to idiomatic JS errors.
export function normalizeError(e: unknown): Error {
  const msg = e instanceof Error ? e.message : String(e);
  if (/must be|at least|invalid_argument|out of range|>=|<=/.test(msg)) return new RangeError(msg);
  return e instanceof Error ? e : new Error(msg);
}
```

- [ ] **Step 3: `js/src/runtime.ts` - the wrapper (the core)**

Write `wrapOp`. It owns one `RawOp` and produces a callable. The event helper writes `nIn` inputs to a scratch heap buffer, calls `evalInto`, reads `nOut` outputs. Dispatch on the argument:
- N numbers (arity `nIn`) -> one event, return `number` if `nOut===1` else `number[]`.
- a single `Float64Array` or `number[]` (only valid when `nIn===1`) -> batch: `reset()`, loop events, return same container type (`Float64Array`->`Float64Array`, `number[]`->`number[]`) when `nOut===1`, else an `NdArray {data, shape:[rows,nOut]}`.
- N arrays (arity `nIn`) -> columnar batch, same return rule.
- a single sync iterable (has `Symbol.iterator`, not a typed array) -> a generator that `reset()`s then yields one output per pulled event.
- a single async iterable (has `Symbol.asyncIterator`) -> an async generator, same.
- `.reset()` delegates; `.dispose()`/`[Symbol.dispose]` free the scratch buffers and call `raw.delete()`. A `FinalizationRegistry` on the wrapper calls `raw.delete()` if the user forgets.
Full skeleton:

```ts
import type { Screamer, RawOp } from "./loader.js";
import { normalizeError } from "./errors.js";
import type { NdArray } from "./ndarray.js";

const REG = new FinalizationRegistry<() => void>((free) => free());

export type ScreamerOp = {
  (...args: any[]): any;
  reset(): void;
  dispose(): void;
  [Symbol.dispose](): void;
};

export function wrapOp(M: Screamer, raw: RawOp): ScreamerOp {
  const nIn = raw.nIn(), nOut = raw.nOut();
  const inBuf = M.allocF64(nIn), outBuf = M.allocF64(nOut);
  let disposed = false;
  const free = () => { if (!disposed) { disposed = true; M.freeBuf(inBuf); M.freeBuf(outBuf); raw.delete(); } };

  const event = (inputs: ArrayLike<number>): number | number[] => {
    M.viewF64(inBuf, nIn).set(inputs as number[]);
    raw.evalInto(inBuf, outBuf);
    const o = M.viewF64(outBuf, nOut);
    return nOut === 1 ? o[0] : Array.from(o);
  };

  const isTyped = (x: any) => x instanceof Float64Array;
  const isNumArr = (x: any) => Array.isArray(x) && (x.length === 0 || typeof x[0] === "number");
  const isSyncIter = (x: any) => x != null && typeof x[Symbol.iterator] === "function" && !isTyped(x) && !isNumArr(x);
  const isAsyncIter = (x: any) => x != null && typeof x[Symbol.asyncIterator] === "function";

  function batch1(arr: ArrayLike<number>, typed: boolean): any {
    raw.reset();
    const rows = arr.length;
    if (nOut === 1) {
      const out = typed ? new Float64Array(rows) : new Array(rows);
      for (let i = 0; i < rows; i++) (out as any)[i] = event([arr[i]]);
      raw.reset();
      return out;
    }
    const data = new Float64Array(rows * nOut);
    for (let i = 0; i < rows; i++) { const o = event([arr[i]]) as number[]; data.set(o, i * nOut); }
    raw.reset();
    return { data, shape: [rows, nOut] } as NdArray;
  }

  function* gen(src: Iterable<number>) {
    raw.reset();
    for (const v of src) yield event([v]);
  }
  async function* agen(src: AsyncIterable<number>) {
    raw.reset();
    for await (const v of src) yield event([v]);
  }

  const call = (...args: any[]) => {
    try {
      if (nIn === 1) {
        const a = args[0];
        if (typeof a === "number") return event([a]);
        if (isTyped(a)) return batch1(a, true);
        if (isNumArr(a)) return batch1(a, false);
        if (isAsyncIter(a)) return agen(a);
        if (isSyncIter(a)) return gen(a);
        throw new TypeError(`unsupported input for a 1-input op: ${typeof a}`);
      }
      // nIn > 1: N scalars -> one event; N arrays -> columnar batch.
      if (args.length === nIn && args.every((x) => typeof x === "number")) return event(args);
      if (args.length === nIn && args.every((x) => isTyped(x) || isNumArr(x))) {
        raw.reset();
        const rows = (args[0] as ArrayLike<number>).length;
        const single = nOut === 1;
        const out = single ? new Float64Array(rows) : new Float64Array(rows * nOut);
        for (let i = 0; i < rows; i++) {
          const o = event(args.map((c) => c[i]));
          if (single) (out as Float64Array)[i] = o as number; else (out as Float64Array).set(o as number[], i * nOut);
        }
        raw.reset();
        return single ? out : ({ data: out, shape: [rows, nOut] } as NdArray);
      }
      throw new TypeError(`expected ${nIn} numeric inputs`);
    } catch (e) { throw normalizeError(e); }
  };

  const op = call as ScreamerOp;
  op.reset = () => raw.reset();
  op.dispose = free;
  (op as any)[Symbol.dispose] = free;
  REG.register(op, free);
  return op;
}
```

- [ ] **Step 4: Test the wrapper against a raw op**

`runtime.test.ts`: build the module, `const M = await init(); const raw = new M.RollingMean(3, "strict"); const op = wrapOp(M, raw);` then assert: `op(1)` returns a number; `op(new Float64Array([1,2,3,4,5]))` returns a Float64Array of length 5 with `[NaN,NaN,2,3,4]`; `op([1,2,3])` returns a `number[]`; `[...op([1,2,3,4,5][Symbol.iterator]())]` streams; and a multi-out op (`new M.RollingMinMax(3)`) batch returns an `{data,shape:[n,2]}`. Free with `op.dispose()`.

Run: `cd js && npm run build:wasm && node --import tsx --test test/runtime.test.ts`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add js/src/ndarray.ts js/src/errors.ts js/src/runtime.ts js/test/runtime.test.ts
git commit -m "feat(js): generic polymorphic op wrapper (4 regimes, lifecycle, ndarray-lite)"
```

---

### Task 3: Typed factory codegen

**Files:**
- Create: `devtools/wasm/gen_ts_api.py`
- Create: `js/src/generated/ops.ts`, `js/src/generated/ops.d.ts` (generated, committed)
- Modify: `js/src/index.ts` (re-export the generated factories)

**Interfaces:**
- Consumes: `devtools/wasm/wasm_manifest.json`, the Python screamer package (for ctor arg names + defaults), `wrapOp` + `init`.
- Produces: for each op a typed factory `export function RollingMean(windowSize: number, startPolicy?: string): ScreamerOp`.

- [ ] **Step 1: Write `gen_ts_api.py`**

Load the manifest. For each op, get arg NAMES + DEFAULTS from the Python signature via `poetry run python`: `import inspect, screamer; sig = inspect.signature(getattr(screamer, name).__init__)` (fall back to positional `arg0..argN` if a signature is unavailable). Map C++ ctor types to TS types: `int`/`double`/`std::optional<double>` -> `number`; `std::string` -> `string`; `std::vector<double>` -> `number[]`. Emit `ops.ts`:

```ts
import { init } from "../loader.js";
import { wrapOp, type ScreamerOp } from "../runtime.js";
export function RollingMean(windowSize: number, startPolicy: string = "strict"): ScreamerOp {
  // init() must have resolved; callers `await ready()` first (see index.ts).
  const M = current(); return wrapOp(M, new M.RollingMean(windowSize, startPolicy));
}
// ... one per op ...
```

where `current()` returns the resolved module (set by `index.ts` after `await init()`), and for `ew_optional` args a `number` param defaulting to `NaN` maps a missing optional; for a `vector<double>` param, materialize `M.VectorDouble` inside the factory (copy the pattern from `smoke.mjs` construct()). Also emit `ops.d.ts` with the same signatures.

- [ ] **Step 2: Handle the sync-init ergonomics in `index.ts`**

```ts
import { init as loadInit, type Screamer } from "./loader.js";
let M: Screamer | null = null;
export function current(): Screamer { if (!M) throw new Error("call `await ready()` before constructing ops"); return M; }
export async function ready(): Promise<void> { M = await loadInit(); }
export * from "./generated/ops.js";
```

(The factories are synchronous like Python; `await ready()` once at startup resolves the module. This keeps `RollingMean(3)` a plain call, matching Python parity.)

- [ ] **Step 3: Generate + coverage check**

`gen_ts_api.py --check`: assert every manifest op has a factory in `ops.ts` and a declaration in `ops.d.ts`. Run: `poetry run python devtools/wasm/gen_ts_api.py && poetry run python devtools/wasm/gen_ts_api.py --check`
Expected: `TS API OK: 226 factories`.

- [ ] **Step 4: Type-check**

Run: `cd js && npm run build:wasm && npx tsc -p tsconfig.json --noEmit`
Expected: no type errors (the generated `ops.ts` compiles against `runtime.ts`/`loader.ts`).

- [ ] **Step 5: Commit**

```bash
git add devtools/wasm/gen_ts_api.py js/src/generated/ops.ts js/src/generated/ops.d.ts js/src/index.ts
git commit -m "feat(js): typed factory codegen for all 226 ops"
```

---

### Task 4: JS parity + regime + lifecycle test suite

**Files:**
- Create: `js/test/regimes.test.ts`, `js/test/parity.test.ts`, `js/test/lifecycle.test.ts`

**Interfaces:**
- Consumes: the public API (`ready` + factories), `wasm/smoke/oracle.json` (Phase 2's Python oracle).

- [ ] **Step 1: `regimes.test.ts`**

For `RollingMean(3)`: assert `op(2)` is a number; `op(new Float64Array([...]))` is a Float64Array (container-preserving); `op([...])` is a `number[]`; `[...op(genOf([1,2,3,4,5]))]` matches the batch; `for await` over an async source yields the same. For `RollingMinMax(3)`: multi-out batch is `{data, shape:[n,2]}` and `toNested()` gives `number[][]`.

- [ ] **Step 2: `parity.test.ts`**

Load `wasm/smoke/oracle.json`; for each entry drive the events through the ergonomic API (`op(x)` per event) and assert the checked output matches within 1e-9. This proves the ergonomic layer preserves the Phase 2 parity.

- [ ] **Step 3: `lifecycle.test.ts`**

Assert `using op = RollingMean(3)` disposes at scope end (wrap in a block, then assert a follow-up use throws or the handle is freed); assert explicit `.dispose()` is idempotent; assert constructing + disposing 1000 ops does not grow `M.HEAP` unboundedly (sample heap via the module if exposed, else assert no throw across the loop). Assert `RollingMean(0)` throws a `RangeError` (kernel `invalid_argument`) and `op("x" as any)` throws a `TypeError`.

- [ ] **Step 4: Run the suite**

Run: `cd js && npm run build:wasm && node --import tsx --test test/*.test.ts`
Expected: all pass. A regime/parity/lifecycle failure is a real finding (report which).

- [ ] **Step 5: Commit**

```bash
git add js/test/regimes.test.ts js/test/parity.test.ts js/test/lifecycle.test.ts
git commit -m "test(js): regimes, parity vs oracle, lifecycle + error mapping"
```

---

### Task 5: TS codegen freshness gate + build

**Files:**
- Create: `tests/test_ts_api_fresh.py`
- Modify: `js/package.json` (ensure `build` emits ESM + CJS + d.ts to `dist/`)

**Interfaces:**
- Consumes: `gen_ts_api.py`.

- [ ] **Step 1: Freshness gate**

`tests/test_ts_api_fresh.py`: regenerate `ops.ts` + `ops.d.ts` via `gen_ts_api.py --stdout` and assert byte-identical to the committed files (so a new Python op without regeneration fails CI). Uses `poetry run python` (needs the screamer import for signatures).

- [ ] **Step 2: Run it + confirm Python suite**

Run: `poetry run pytest tests/test_ts_api_fresh.py -q` -> pass.
Run: `poetry run pytest -q 2>&1 | tail -3` -> 6931 passed, 2 skipped (6930 + 1 new).

- [ ] **Step 3: Verify the dist build**

Run: `cd js && npm run build && ls dist/index.js dist/index.d.ts`
Expected: `tsc` emits the ESM build + declarations with no errors. (CJS output via a second tsconfig or a bundler is acceptable; if `tsc` alone cannot emit `.cjs`, emit ESM `.js` + `.d.ts` now and note CJS as a Phase 6 packaging detail.)

- [ ] **Step 4: Commit**

```bash
git add tests/test_ts_api_fresh.py js/package.json
git commit -m "test(js): TS codegen freshness gate + dist build"
```

---

## Notes for the executor

- The core is Task 2 (`wrapOp`). Get the four regimes and container preservation exactly right; the Task 4 parity test against the Phase 2 oracle is the objective gate that it matches Python.
- Do not hand-edit `js/src/generated/*`; regenerate via `gen_ts_api.py`. The freshness gate enforces it.
- `src/generated/screamer.mjs`/`.wasm` are build artifacts (from `npm run build:wasm`), gitignored; only the generated `ops.ts`/`ops.d.ts` and the hand-written `src/*.ts` are committed.
- The batch path replays `evalInto` per event (the `evalBatchInto` runtime helper is an equivalent optimization available if a single boundary crossing per array is wanted later); correctness follows from the batch==stream invariant, checked by the parity test.
- Everything is additive under `js/` and `devtools/wasm/` plus one new `tests/` file. No Python binding, operator, or compute change; no Phase 2 generator/runtime/build change.
