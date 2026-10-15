# Parity with Python

screamer.js and the Python `screamer` package share one implementation of every function. Both
call the same C++ kernels: JavaScript through a thin binding to a WebAssembly build, Python
through a thin nanobind layer.

That means:

- **Same results.** Every function's output matches the Python package's output on the same
  input to 1e-9, verified by cross-language fixture tests.
- **Same causal semantics.** No function looks ahead, in either language.
- **Same warmup behavior.** `startPolicy` options (`"strict"`, `"expanding"`, `"zero"`, where an
  op exposes them), NaN warmup periods, and window-fill rules match exactly.
- **Same saved pipelines.** A `Pipeline` saved as JSON in one language loads in the other.

The two packages differ in their surface. Op factories have the same names in both
(`RollingMean`), options and plain functions are `camelCase` in JavaScript (`startPolicy`,
`positionTest`) and `snake_case` in Python (`start_policy`, `position_test`), and the calling
conventions follow each language's idioms (see [Calling conventions](./regimes.md) for the
JavaScript side).

## Where to go for the math

This guide covers the JavaScript surface: how to call an op, how to build a pipeline, how to
manage lifetime. It does not re-derive each function's formula. For the full per-function
reference, including the underlying math, parameter semantics, and worked examples, see the
[Python documentation](https://screamer.readthedocs.io/en/latest/). Every op factory in the
API reference here links back to it as well.
