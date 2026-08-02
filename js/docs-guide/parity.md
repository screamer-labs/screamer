# Parity with Python

screamer.js is not a reimplementation of the Python `screamer` package. It compiles the same
C++ operator kernels to WebAssembly and calls them through a thin JS binding layer, the same way
the Python package calls them through a thin pybind11 layer. There is one implementation of
every operator's numerics, shared by both languages.

That means:

- **Same results.** Every operator's output matches the Python package's output on the same
  input to 1e-9, verified by cross-language fixture tests.
- **Same causal semantics.** No operator looks ahead. A batch call and a step-by-step streaming
  call on the same data always agree, in both languages.
- **Same warmup behavior.** `startPolicy` options (`"strict"`, `"expanding"`, `"zero"`, where an
  op exposes them), NaN warmup periods, and window-fill rules match exactly.

What differs between the two packages is the surface, not the math: JS uses `camelCase` op
names and options (`RollingMean`, `startPolicy`) where Python uses `snake_case`
(`rolling_mean`, `start_policy`), and the calling conventions follow each language's idioms (see
[Calling conventions](./regimes.md) for the JS side).

## Where to go for the math

This guide covers the JS surface: how to call an op, how to build a pipeline, how to manage
lifetime. It does not re-derive each operator's formula. For the full per-operator reference,
including the underlying math, parameter semantics, and worked examples, see the
[Python documentation](https://screamer.readthedocs.io/en/latest/). Every op factory in the
API reference here links back to it as well.
