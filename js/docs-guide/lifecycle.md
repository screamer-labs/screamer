# Lifecycle and disposal

Every op is a thin wrapper around WASM-side state: a C++ kernel instance plus a couple of small
input/output buffers on the WASM heap. That memory is not garbage-collected by the JS engine,
so an op needs to be disposed once you are done with it.

## `using` (explicit resource management)

The cleanest way to dispose an op is a `using` declaration. It runs `Symbol.dispose` on the op
automatically at the end of the enclosing block, including when the block exits via an
exception:

```js
{
  using sma = RollingMean(3);
  sma(1);
  sma(2);
  sma(3); // 2
} // sma is disposed here, even if the block above threw
```

## Explicit `.dispose()`

Where `using` is not available (older toolchains, or an op whose lifetime does not follow a
block), call `.dispose()` directly:

```js
const sma = RollingMean(3);
sma(1);
sma.dispose();
```

Calling `.dispose()` twice, or after a `using` scope has already disposed the op, is safe: the
second call is a no-op. Calling the op itself after disposal throws, rather than writing
through freed WASM memory:

```js
sma.dispose();
sma(4); // throws: "operation used after dispose()"
```

## The FinalizationRegistry net

Every op and every `Pipeline` also registers with a `FinalizationRegistry`, so if you drop every
reference to one without disposing it, its WASM memory is freed after the JS engine
garbage-collects the wrapper. A program that builds a new op on every call and drops it keeps a
flat memory footprint on both heaps. A stream keeps its op alive until the stream is finished or
dropped, and a `pipeline.live()` driver keeps its `Pipeline` alive.

The engine runs finalizers between tasks, so a loop that never yields to the event loop holds
every op it dropped until it does: the JS wrapper (about 500 bytes) and the op's C++ state on the
WASM heap. The C++ state grows with the op's parameters; `RollingMean(1000)` keeps a window of
1000 values, about 8 KB. A synchronous parameter sweep that builds hundreds of thousands of such
ops in one loop can fill the WASM heap before any of them is freed. Use `using` there, or call
`.dispose()` on each op:

```js
for (const w of windows) {
  using sma = RollingMean(w);
  results.push(sma(prices));
} // each op is freed at the end of its iteration
```

GC timing is not predictable either. Prefer `using` or an explicit `.dispose()` in code you
control: both free the memory at once.

## A Pipeline owns copies of its ops

A `Pipeline` compiles its own copy of every op in its graph (as `RollingMean(3)` below), made
by the op's factory with the same settings and a fresh state. `pipeline.dispose()` frees the
copies with the graph:

```js
const x = Input("x");
const rm = RollingMean(3);
const pipeline = new Pipeline([x], [Diff(1)(rm(x))]);

rm.dispose();       // the Pipeline runs on its own copy: this does not affect it
pipeline(data);
pipeline.dispose(); // frees the graph and its op copies
```

So two Pipelines built from the same nodes never share state, a direct call of `rm` does not
change the Pipeline, and disposing `rm` at any time is safe.

Reusing one op instance across two different nodes of the same graph still throws: construct
a fresh instance per node.
