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
// A C++ exception that leaves WASM does not unwind the WASM stack pointer, so
// every caught one would leak the frames it unwound, and after a few thousand
// the stack would overrun the heap. A call into WASM is always made from
// JavaScript with no WASM frame below it (no C++ code calls back into user
// JavaScript), so on a caught exception the stack pointer goes back to the
// value it had when the module loaded.
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
  if (WasmException && e instanceof WasmException && helpers) {
    helpers.stackRestore(stackBase);
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
