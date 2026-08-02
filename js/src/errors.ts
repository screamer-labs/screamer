// Kernel ctor/eval errors usually surface from Embind as generic Error objects
// whose message carries the C++ what(). Map the common shapes to idiomatic JS
// errors.
//
// This WASM build (see wasm/CMakeLists.txt) does not link embind's exception
// message translation, so a C++ exception thrown *during construction* (e.g.
// `new M.RollingMean(0, ...)` rejecting an invalid window size) is not caught
// and converted into a proper Error by Embind at all: it surfaces in JS as a
// bare number (the raw exception pointer), with no recoverable `what()` text.
// `loader.ts`'s `wrapCtors` routes every such throw through this function.
// Across the C++ op sources, constructor-time argument validation is
// overwhelmingly `throw std::invalid_argument(...)` (170/178 non-DAG throws);
// the few exceptions are unreachable from this scalar point-op API (numpy
// dispatch checks, an internal order-statistic-tree bound). So a non-Error
// value here is treated as that same invalid_argument case and mapped to
// RangeError, matching the message-based classification below for the errors
// that do carry text.
export function normalizeError(e: unknown): Error {
  if (!(e instanceof Error)) {
    return new RangeError(
      `invalid argument (opaque WASM exception, no message text available: ${String(e)})`,
    );
  }
  const msg = e.message;
  if (/must be|at least|invalid_argument|out of range|>=|<=/.test(msg)) return new RangeError(msg);
  return e;
}
