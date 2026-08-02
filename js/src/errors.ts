// Kernel ctor/eval errors surface from Embind as generic Error objects whose
// message carries the C++ what(). Map the common shapes to idiomatic JS errors.
export function normalizeError(e: unknown): Error {
  const msg = e instanceof Error ? e.message : String(e);
  if (/must be|at least|invalid_argument|out of range|>=|<=/.test(msg)) return new RangeError(msg);
  return e instanceof Error ? e : new Error(msg);
}
