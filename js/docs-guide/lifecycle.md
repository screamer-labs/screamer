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
the memory of every op it dropped (a few hundred bytes each) until it does. GC timing is not
predictable either. Prefer `using` or an explicit `.dispose()` in code you control: both free
the memory at once.

## Ops inside a Pipeline are pinned

Once an op appears inside a `Pipeline` graph (as in `RollingMean(3)(x)` below), the `Pipeline`
takes over its lifetime. It retains the op for as long as the compiled graph is alive, and
`pipeline.dispose()` releases that retention:

```js
const x = Input("x");
const y = Diff(1)(RollingMean(3)(x));
const pipeline = new Pipeline([x], [y]);

// Do not call RollingMean's own .dispose() here: the Pipeline owns it now.

pipeline(data);
pipeline.dispose(); // releases every op the graph retained
```

Do not call `.dispose()` on an op you have passed into a live `Pipeline`. Doing so frees state
the compiled graph still points at, and the next call into the graph reads freed memory. Let
`pipeline.dispose()` handle it instead.

A second, independent instance of the same op kind is unaffected by this rule; the constraint
is about the specific op instance backing a specific graph node, not about calling
`RollingMean(3)` more than once. What is not allowed is reusing one op instance across two
different nodes in the same graph: a graph node is stateful, so two nodes sharing one instance
would corrupt each other's state. Construct a fresh instance per node instead.
