# Unified streaming model for functors, stream operators, and Dags

**Status:** design (pre-implementation)
**Date:** 2026-07-09
**Scope:** a breaking, pre-1.0 redesign of how every computation in screamer is
called and streamed. Settles a single call-and-stream surface so a `Dag` behaves
exactly like a function, and renames `resample`'s arguments as a consequence.

## Motivation

Today screamer has three different streaming stories, and in places two
implementations of the same math:

1. **Functors** are called `f(array)` for batch, `f(iterable)` eagerly consumes to a
   list, and there is a separate C++ lazy-iterator type (`LazyIterator`,
   `FunctorIterator`).
2. **Stream operators** have an eager form (`resample(...)`) and a parallel
   `*_iter` Python generator over `(value, index)` events (`resample_iter`,
   `merge_iter`, `combine_latest_iter`, ...), which re-implements the windowing math
   in Python.
3. **Dags** are `dag()` for batch, `dag.stream()` for replay, and `dag.live()` for a
   push session with `push` / `advance` / `flush` / `result`.

Three families, three streaming idioms, two implementations of resample. This is the
"needless variants, works slightly different in different situations" problem. A user
cannot carry one mental model across the library, and a `Dag` (which should just be a
function built from functions) has bespoke methods no function has.

## Goal

**One call-and-stream surface, shared by functors, stream operators, and Dags.** A
`Dag` is called exactly like a functor. There is one streaming concept, reused
everywhere, that covers batch, pull, and push. The index is ordinary data, not a
hidden channel. Async multi-input is handled by one alignment rule, not per-operator
special cases.

### Non-goals

- Not changing the numeric results of any operator (batch output is unchanged).
- Not adding timezone or physical-unit awareness; screamer stays integer-index-space
  and unit-agnostic (see "The index is data").
- Not preserving the current Python API; this is a deliberate pre-1.0 break.

## The unified model

### 1. A computation is one callable, dispatching on input type

Every computation `c` (a functor, a stream operator, or a `Dag`) is a stateful
callable:

```python
c(x)          # scalar(s): push one event; returns whatever became ready
c(array)      # array(s): batch; reset, run the whole array, return an array
c(iterable)   # iterable(s): pull; return a lazy iterator that yields as it consumes
c.reset()     # clear streaming state
```

- **Push** is calling `c` on one event (a scalar, or one value per input). The
  computation retains state across calls, exactly as a functor's incremental use does
  today.
- **Batch** is calling `c` on whole arrays. It resets, runs, and returns arrays.
- **Pull** is calling `c` on iterables. It returns a lazy iterator.

**The output type mirrors the input type** (this is already the parser's design):
value in, value out; array in, array out; lazy iterator in, lazy iterator out. A live
push loop is simply a lazy iterator whose source produces events as they arrive, so
"push" is not a separate return convention, it is the iterator form with a live source.

The same three modes work identically for a functor, an operator, and a `Dag`. There
are no `stream()` / `live()` / `advance()` / `flush()` methods and no `*_iter`
functions; they are all subsumed by this dispatch (see "What retires").

### 2. Output cardinality: map-like vs window-like

Two honest kinds of computation, distinguished by how many outputs one event yields.
This is not an interface inconsistency; it is the same difference Python draws between
`map` and `filter`.

- **Map-like (one out per event):** elementwise functors (`cumsum`, rolling and
  expanding stats, `sign`, ...). A push returns exactly one output.
- **Window-like (zero or more out per event):** `resample`, `dropna`, `filter`. Most
  events yield nothing; a boundary or a kept event yields one; gap-fill can yield
  several at once.

Under type propagation this needs no special return convention; the array and the
iterator carry the cardinality:

- A **map-like** computation is 1:1, so `cumsum(3.0)` returns `6.0`, `cumsum(array)`
  returns an equal-length array, `cumsum(iter)` yields one scalar per event. The pleasant
  functor ergonomics are unchanged.
- A **window-like** computation is fed an **array** (`resample(array)` returns the M
  bars) or an **iterator** (`resample(iter)` yields each bar as it closes, nothing on a
  mid-bar event, several across a filled gap). Because a single event yields zero or
  more outputs, a window operator is fed an array or an iterator, not scalars one at a
  time; the iterator is exactly where "zero or more" lives naturally. A live loop feeds
  a window operator an iterator whose source is live.

A computation knows which kind it is, and the caller knows which operator they are using
(a running sum versus a bar builder), so the shape is predictable, not a surprise.

### 3. Multi-input: positional, tuple (`*args`), or dict (`**kwargs`)

A computation with N inputs is called either with N inputs directly, or with a single
input that carries them:

```python
sub(a, b)            # two inputs, directly
sub(pair)            # one input carrying (a, b); the tuple is unpacked (like *args)
sub(named)           # one input carrying {"a":.., "b":..}; unpacked by name (**kwargs)
```

This holds for each mode: scalars (`sub(1.0, 2.0)` or `sub((1.0, 2.0))`), arrays, and
iterables (`sub(a_it, b_it)` or `sub(iter_of_pairs)`).

**Output symmetry:** a multi-output computation *produces* a stream of tuples (or
dicts), which a downstream multi-input computation *consumes* by unpacking. So
`f(g(...))` composes whenever `g`'s outputs match `f`'s inputs, positionally by tuple
or by name via dict. Example: `resample` emits `(bar_label, bar_value)` pairs, which
drop straight into any two-input function.

### 4. The index is data, not metadata

An index (a timestamp, or any ordering key) is an **ordinary input** to the few
functions that are time-aware, and an **ordinary output** where a function assigns
one. It is not a hidden channel riding on every value, and there is no `Stream.index`.

- `cumsum(values)`: one input, never sees an index.
- `resample(index, values, ...)`: index is input number 0; the operator buckets on it.
- `resample` output is `(bar_label, bar_value)` pairs: the new index (bar labels) is
  ordinary output data that flows on to the next function.

Consequence: the index is *always meaningful when present and simply absent when not*,
which removes today's "the index may or may not be there" ambiguity. There is no
mandatory counter forced onto index-free functions.

Screamer stays **integer-index-space and unit-agnostic**: an index value is an int,
and a bar width is in the same units as the index (if your index is seconds, a width
of 60 is 60 seconds; if nanoseconds, 60 nanoseconds). The library does not attach or
interpret units. Naming and docs make "in index units" explicit; the unit-agnosticism
is deliberate, not an oversight.

### 5. Async multi-input alignment: as-of by default

When a multi-input computation is fed inputs that do not tick together, it aligns
them **as-of by index (last-value-carry), which is exactly `combine_latest`.** A bare
stream's index is its position, so for equal-rate bare streams this is lockstep `zip`.

```python
add(a, b) == add(combine_latest(a, b))     # same numbers, by construction
```

- The implicit default and the explicit `combine_latest` agree: naming the alignment
  never changes the result, it only lets you see it or vary it.
- Mechanically: a k-way merge by index; pull from whichever input has the next index
  (for bare streams, both, in lockstep), keep each input's latest value, and emit once
  per merged index (same-index events coalesce, as `combine_latest` already does).
As-of is the default (and the only join in v1) because every input always has a current
value, nothing is silently dropped, and it is what the time-series case wants. A strict
join (emit only when inputs share an index) is the rare coincidence-only case: for
example, `price` at indices `{1,3,5}` and `volume` at `{2,4}` never coincide, so a
strict join emits nothing while as-of emits at every index. That is niche, so it is out
of scope for v1 and can be added as a named combinator later.

### 6. No separate clock; the index is the clock

There is no clock concept and no `advance()`. "Advancing time" only ever means
"`resample`'s index moved forward", and the index is already an ordinary input.

- **Empty bars between events** are handled by the index gap alone: when the index
  jumps across one or more empty buckets, `fill` decides what those buckets emit. No
  clock needed.
- **Finalizing a bar in real time** before the next event arrives is done by feeding an
  event that advances the index but carries no value: `(index, NaN)`. `resample`
  already ignores NaN for the aggregate but still buckets on the index, so a NaN-valued
  event is a heartbeat. Nothing new is added.
- **End of stream** is the natural exhaustion of a pull iterator (or the end of the
  input array), which flushes the trailing partial window. This replaces `flush()`.

So time and termination are just events (and iterator exhaustion) in the same stream;
no windowing operator needs a bespoke control method or a separate clock input.

## `resample` redesign (falls out of the model)

`resample` becomes a normal computation under the model, which resolves every
complaint about its current signature. The two bad names `every=` / `count=` collapse
into **one argument, `interval`**, whose meaning is read from the index:

- **No index** (`resample(values, interval=N)`): `interval` is a **bin size by count**,
  a bar every `N` events. `resample` keeps its own internal counter.
- **Integer index** (`resample(index, values, interval=W)`): `interval` is a **span in
  index units**, bar `n` is the half-open interval
  `[origin + n*W, origin + (n+1)*W)`.
- **Timestamp index**: `interval` may additionally be a `timedelta`, converted to
  integer index units internally. This is the one bit of unit-awareness, a thin
  optional convenience over the integer core.

So there is no separate "mode" to name and no "index optional maybe there" ambiguity:
providing an index makes `interval` a span, omitting it makes `interval` a count. The
index is just input number 0 when present.

- **Output is `(bar_label, bar_value)` pairs** (or `(bar_label, col_0, ..., col_k)`
  for multi-column bars), so bars flow on to the next function as ordinary tuple data.
- **`fill=`** (empty-bar policy), **`origin=`**, and **`label=`** carry over unchanged.

## Precise dispatch rules

To make "call on a scalar vs an array vs an iterable" unambiguous, especially with
tuple unpacking:

1. A **scalar** argument (a real number) is one event's value. `c(3.0)` is a push.
2. An **array** argument is a batch. `c(np.array([...]))` resets and runs.
3. An **iterable** argument (that is not an array) is a pull; `c` returns a lazy
   iterator.
4. For an **N-input** computation given a **single** argument:
   - a tuple of N scalars is one aligned event, unpacked to the N inputs;
   - an iterable whose elements are N-tuples is a stream, each element unpacked;
   - a dict (or an iterable of dicts) unpacks by input name.
   Arity plus element type disambiguates: a 2-input function reads `(1.0, 2.0)` as one
   event, and `[(1.0, 2.0), (3.0, 4.0)]` as a two-event stream.
5. `c(a, b)` with N positional arguments feeds the N inputs directly, each argument
   independently a scalar, array, or iterable per rules 1 to 3.

## What retires

- `Dag.stream()`, `Dag.live()`, and the live-session methods `push` / `advance` /
  `flush` / `result` (subsumed by the callable, by the index being an input, and by
  iterator exhaustion).
- Every `*_iter` stream operator (`resample_iter`, `merge_iter`,
  `combine_latest_iter`, `dropna_iter`, `filter_iter`, `select_iter`) and their Python
  reimplementations of the math. Pull is the callable on iterables, backed by the C++
  engine.
- **The `Stream` type entirely.** A stream is just a sequence (list or lazy iterator)
  of values, of tuples (`*args`), or of dicts (`**kwargs`), or a 2D array (columns =
  inputs). Named multi-column output is a list of names carried beside the data, not a
  wrapper. This builds on the existing polymorphic `__call__` in `functor_base.h`,
  which already dispatches scalar / array / iterable, accepts N positional args or one
  tuple/list of N, reads a 2D array as N columns, and casts the return to match; the
  work is to make the iterable path return a lazy iterator, add dict unpacking, and let
  operators and Dags share it.
- `every=` / `count=` on `resample`, replaced by the single contextual `interval=`.
- Any separate clock concept (`advance()`, a clock input); the index is the clock.

## Migration

This is a breaking change across the public API. Because screamer is pre-1.0 (0.6.x),
we take the break rather than carry a compatibility shim, but the plan should:

- land the new call surface behind the existing C++ engine (no numeric changes), so
  `batch == pull == push` is provable against the current batch outputs as the oracle;
- update every doc page, docstring, and notebook to the new surface in the same
  change set (the docs are the acceptance test for "one recognizable model");
- provide a short migration table (old call to new call) for the notebooks and the
  changelog.

## Settled decisions (from spec review)

1. **`resample`:** a single `interval=` argument, contextual to the index (count when
   no index, index-span for an integer index, `timedelta` for a timestamp index). No
   separate mode argument.
2. **`Stream` is removed.** Streams are plain sequences of values / tuples / dicts, or
   2D arrays; the polymorphic parser handles it, and names travel beside the data.
3. **No clock.** The index is the clock; a heartbeat is a `(index, NaN)` event.
4. **No strict-join in v1** (YAGNI). Only the as-of default (`combine_latest`). A named
   strict combinator can be added later if a real need appears.
5. **Type propagates** (value in, value out; array in, array out; lazy iterator in,
   lazy iterator out). There is no separate "window-op return type"; cardinality is
   carried by the array or iterator, so window operators are fed arrays or iterators,
   not scalars one at a time.

## Remaining open questions

- Whether the `timedelta` convenience for timestamp indices is in v1 or deferred (the
  integer-`interval` core is v1 regardless).
- The exact spelling of `interval=` (versus `bin=` or `window=`), a naming preference
  only.

## Testing strategy

- **Oracle:** the current batch outputs. Every operator must satisfy
  `batch == pull == push` bit-for-bit (extending today's `batch == stream` guarantee to
  all three modes). Causality (no lookahead) is unchanged and re-verified.
- **Dispatch tests:** the rules in "Precise dispatch rules", including tuple and dict
  unpacking, single-tuple-event versus tuple-stream, and multi-output to multi-input
  composition.
- **Alignment tests:** `add(a, b) == add(combine_latest(a, b))` for bare equal-rate
  (equals `zip`) and for sparse timestamped inputs (as-of carry).
- **Index and heartbeat tests:** an index gap fills empty bars per `fill`; a
  `(index, NaN)` heartbeat closes a time bar with no trade; iterator exhaustion flushes
  the trailing partial; all match the batch oracle.
- All numeric logic stays in C++ (the pull path must drive the C++ engine, not a Python
  reimplementation), so the pure-C++ library and a future WASM binding inherit the
  model.

## Scope note

This is large and cross-cutting: it touches the functor call convention, every stream
operator, the `Dag`, the `*_iter` layer, `Stream`, and `resample`. The implementation
plan should decompose it into sequenced, independently-testable pieces (for example:
callable-dispatch core with type propagation and cardinality; lazy-iterator output;
multi-input tuple/dict unpacking and as-of alignment; `resample` single-`interval`
re-signature and heartbeats; retire `*_iter` and the `Stream` type; docs and
notebooks), each keeping `batch == pull == push` green.
