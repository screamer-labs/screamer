// Turns an error thrown by a call into WASM into an idiomatic JS error. Every
// caller wraps only the call into WASM (a method of an Embind object, an `M.*`
// function, or `new M.X(...)`), so an error thrown by the caller's own code,
// such as an iterator or a getter, never reaches this function and is rethrown
// unchanged.
//
// The module is built with native WebAssembly exceptions (wasm/CMakeLists.txt),
// so a C++ exception reaches JavaScript as a WebAssembly.Exception. The C++
// core validates arguments with `throw std::invalid_argument(...)`, so such an
// exception becomes a RangeError carrying its what() text, read with
// Emscripten's getExceptionMessage, which returns [type, message]. Embind's own
// Error objects (a wrong argument type, for example) keep their class unless
// the message reads like a validation error.
//
// An error that leaves WASM does not unwind the WASM stack pointer, so every
// caught one would leak the frames it unwound, and after a few thousand the
// stack would overrun the heap. On a caught error the stack pointer goes back
// to the value it had when the module loaded.
//
// The reset goes to `stackBase`: the stack pointer when the module loaded, so
// with no WASM frame live. C++ calls back into user JavaScript from the
// randomization tests (validation.ts: strategyTest, positionTest, and
// walkForward, through strategyTestRun, positionTestRun, and walkForwardRun),
// and every such callback runs inside
// callbackScope, which moves the base to the stack pointer at the callback's
// entry for its duration. An error caught inside the callback then frees only
// the frames above that entry and keeps the C++ frames below it. Arguments
// read in C++ through a val (layout plans, string parameters) are plain
// copies or primitive strings, checked before the call.
interface ExceptionHelpers {
  getExceptionMessage(e: unknown): [string, string | undefined];
  decrementExceptionRefcount(e: unknown): void;
  stackSave(): number;
  stackRestore(sp: number): void;
}
let helpers: ExceptionHelpers | null = null;
let stackBase = 0;

// Called once, when the module has loaded and no WASM frame is live.
export function setExceptionHelpers(h: ExceptionHelpers): void {
  helpers = h;
  stackBase = h.stackSave();
}

const WasmException: (new (...a: any[]) => unknown) | undefined = (globalThis as any).WebAssembly?.Exception;

/** The text of the RangeError for a call that needs more WASM heap than there is
 * (Python raises MemoryError). */
export const OUT_OF_MEMORY = "out of memory: the call needs more memory than is available";

export function normalizeError(e: unknown): unknown {
  // Any error from a call into WASM may have left WASM frames (an Embind Error
  // thrown while C++ converts a value, for example), so the reset is for all.
  helpers?.stackRestore(stackBase);
  if (WasmException && e instanceof WasmException && helpers) {
    let type: string | undefined, msg: string | undefined;
    // The caught exception holds the last reference to the C++ object; drop
    // it once the message is read, or the object leaks. Reading it copies the
    // text on the heap, which fails when the heap is exhausted.
    try { [type, msg] = helpers.getExceptionMessage(e); } catch { type = undefined; } finally {
      try { helpers.decrementExceptionRefcount(e); } catch { /* the heap is exhausted */ }
    }
    // An allocation that failed: std::bad_alloc, or a container's
    // std::length_error ("vector"); also an exception whose text could not
    // be read (no heap left to copy it).
    if (!type || /bad_alloc|length_error/.test(type)) return new RangeError(OUT_OF_MEMORY);
    // A custom metric's result of the wrong kind, an index or a setting of the
    // wrong kind is a TypeError, as in Python (MetricTypeError, IndexTypeError,
    // SettingTypeError; the type name may be mangled).
    if (/MetricTypeError|IndexTypeError|SettingTypeError/.test(type)) return new TypeError(msg);
    return new RangeError(msg ?? `C++ exception of type ${type}`);
  }
  if (e instanceof Error && /must be|at least|invalid_argument|out of range|>=|<=/.test(e.message))
    return new RangeError(e.message);
  return e;
}

// Runs one call into WASM and normalizes what it throws. For paths that are not
// hot; a per-event path inlines the same try block instead of a closure.
export function wasmCall<T>(f: () => T): T {
  try { return f(); } catch (e) { throw normalizeError(e); }
}

// Runs a JavaScript callback that C++ calls while WASM frames are live below
// it. For its duration the stack base is the stack pointer at entry, so an
// error caught inside (by wasmCall or here) restores the stack to that point
// and not below the live frames. The error is rethrown. Not exported:
// wasmCallbacks is the only way to hand functions to C++.
function callbackScope<T>(f: () => T): T {
  const saved = stackBase;
  const base = helpers ? helpers.stackSave() : 0;
  stackBase = base;
  try {
    return f();
  } catch (e) {
    helpers?.stackRestore(base);
    throw e;
  } finally {
    stackBase = saved;
  }
}

// Functions that C++ may call back, and the call into WASM that receives them.
// This is the only path for a JavaScript callback into C++: each function runs
// in callbackScope and never throws into WASM. It returns 0, or 1 when it
// threw; the first error is kept, C++ unwinds on the 1, and `call` rethrows
// that error unchanged (any other error is normalized as by wasmCall).
// js/test/validation.test.ts checks that C++ calls JavaScript only through
// such an object.
export function wasmCallbacks<F extends Record<string, (...args: any[]) => void>>(fns: F): {
  call<T>(f: (callbacks: Record<keyof F, (...args: any[]) => number>) => T): T;
} {
  let failed = false;
  let error: unknown;
  const callbacks = {} as Record<keyof F, (...args: any[]) => number>;
  for (const name of Object.keys(fns) as (keyof F)[]) {
    const f = fns[name];
    callbacks[name] = (...args: any[]) => {
      try {
        callbackScope(() => f(...args));
        return 0;
      } catch (e) {
        if (!failed) { failed = true; error = e; }
        return 1;
      }
    };
  }
  return {
    call<T>(f: (callbacks: Record<keyof F, (...args: any[]) => number>) => T): T {
      try {
        return f(callbacks);
      } catch (e) {
        const normalized = normalizeError(e);
        throw failed ? error : normalized;
      }
    },
  };
}
