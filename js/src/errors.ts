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
// with no WASM frame live. C++ calls back into user JavaScript in one place,
// strategyTest (validation.ts), and every such callback runs inside
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

export function normalizeError(e: unknown): unknown {
  // Any error from a call into WASM may have left WASM frames (an Embind Error
  // thrown while C++ converts a value, for example), so the reset is for all.
  helpers?.stackRestore(stackBase);
  if (WasmException && e instanceof WasmException && helpers) {
    let type = "", msg: string | undefined;
    // The caught exception holds the last reference to the C++ object; drop
    // it once the message is read, or the object leaks.
    try { [type, msg] = helpers.getExceptionMessage(e); } finally { helpers.decrementExceptionRefcount(e); }
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
// and not below the live frames. The error is rethrown.
export function callbackScope<T>(f: () => T): T {
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
