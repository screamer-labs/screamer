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

Rule: **a push returns the outputs that became ready from that event, and a pull
yields the same lazily.**

- A map-like computation, being always exactly one, **auto-unwraps**: `cumsum(3.0)`
  returns `6.0` (a scalar), `cumsum(array)` returns an array, `cumsum(iter)` yields
  scalars. This keeps the pleasant functor ergonomics.
- A window-like computation returns the ready outputs as a sequence per push
  (possibly empty), and its pull yields each output event: `resample(...)` mid-bar
  yields nothing, at a bar close yields one bar, across a filled gap yields several.

A computation knows which kind it is; the caller knows which operator they are using
(a running sum vs a bar builder), so the output shape is predictable, not a surprise.

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
- A **strict** join (emit only when inputs share an index) is an explicit opt-in for
  the rare case you want coincidence-only. For bare equal-rate streams it is identical
  to the default.

As-of is the default because every input always has a current value, nothing is
silently dropped, and it is what the time-series case wants.

### 6. Clock and end-of-stream are events, not methods

- A **clock tick** is an event with no value on a dedicated clock input (or a valueless
  heartbeat). Feeding it closes any window whose boundary has passed. This replaces
  `advance(now)`: advancing time is pushing a clock event.
- **End of stream** is the natural exhaustion of a pull iterator, or an explicit EOF
  marker on push, which flushes the trailing partial window. This replaces `flush()`.

No windowing operator needs bespoke control methods; time and termination are just
events in the same stream.

## `resample` redesign (falls out of the model)

`resample` becomes a normal computation under the model, which resolves every
complaint about its current signature.

- **The index is input number 0** for time bars, and absent for count bars, so it is
  never "optionally maybe there": time-mode has an index input, count-mode does not.
- **Two modes, clearly separated:**
  - **Time bars** bucket on the index: bar `n` is the half-open interval
    `[origin + n*W, origin + (n+1)*W)`. Needs the index input. `W` is in index units.
  - **Count bars** bucket by arrival order: a bar every `N` events. Index-free; keeps
    an internal counter and can still label bars by count or by the first or last
    event's index if an index input is supplied.
- **Output is `(bar_label, bar_value)` pairs** (or `(bar_label, col_0, ..., col_k)`
  for multi-column bars), so bars flow on to the next function with no `Stream.index`.
- **`fill=`** (empty-bar policy) and **`origin=`** and **`label=`** carry over
  unchanged in behavior.

### Argument naming (open decision, recommendation below)

The bad names `every=` / `count=` are replaced. Recommended:

- Time bars: `resample(index, values, interval=W, origin=0, label="left", fill="skip")`
  where `interval` is the bar width in index units.
- Count bars: `resample(values, size=N, label="left")` where `size` is the number of
  events per bar.

`interval` reads as "a span on the index"; `size` reads as "how many events per bar".
Alternatives considered: `width=` for time (collides with the ohlc "width" notion),
`ticks=` or `n=` for count. See "Open decisions".

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
  `flush` / `result` (subsumed by the callable and by clock and EOF events).
- Every `*_iter` stream operator (`resample_iter`, `merge_iter`,
  `combine_latest_iter`, `dropna_iter`, `filter_iter`, `select_iter`) and their Python
  reimplementations of the math. Pull is the callable on iterables, backed by the C++
  engine.
- `Stream.index` as a metadata channel. A `Stream` (if it survives at all) becomes a
  thin labelled-array view over output tuples, not an index carrier. (Open decision:
  whether `Stream` remains as a convenience.)
- `every=` / `count=` argument names on `resample`.

## Migration

This is a breaking change across the public API. Because screamer is pre-1.0 (0.6.x),
we take the break rather than carry a compatibility shim, but the plan should:

- land the new call surface behind the existing C++ engine (no numeric changes), so
  `batch == pull == push` is provable against the current batch outputs as the oracle;
- update every doc page, docstring, and notebook to the new surface in the same
  change set (the docs are the acceptance test for "one recognizable model");
- provide a short migration table (old call to new call) for the notebooks and the
  changelog.

## Open decisions (recommendations given; confirm on spec review)

1. **`resample` argument names:** `interval=` (time) and `size=` (count), per above.
   Alternatives: `width=` / `ticks=`.
2. **Does `Stream` survive?** Recommendation: keep a thin labelled view for the
   multi-column bar case (named columns), but with columns as tuple positions and
   labels as data, not as a hidden index. Or drop it entirely and return plain tuples
   plus a separate names list. Lean: keep a minimal labelled view.
3. **Clock input shape:** a dedicated named clock input on windowing operators, versus
   a valueless heartbeat event interleaved on an existing input. Lean: a dedicated
   clock input, because it is explicit and composes as ordinary data.
4. **Strict-join surface:** the explicit opt-in name for coincidence-only alignment
   (for example `zip_strict` or a flag on `combine_latest`). Lean: a distinct
   combinator, so the default `combine_latest` has no mode flag.
5. **Push return type for window-like ops:** a tuple, a list, or a small generator of
   the ready outputs. Lean: a tuple (immutable, cheap, and iterable).

## Testing strategy

- **Oracle:** the current batch outputs. Every operator must satisfy
  `batch == pull == push` bit-for-bit (extending today's `batch == stream` guarantee to
  all three modes). Causality (no lookahead) is unchanged and re-verified.
- **Dispatch tests:** the rules in "Precise dispatch rules", including tuple and dict
  unpacking, single-tuple-event versus tuple-stream, and multi-output to multi-input
  composition.
- **Alignment tests:** `add(a, b) == add(combine_latest(a, b))` for bare equal-rate
  (equals `zip`) and for sparse timestamped inputs (as-of carry); strict-join opt-in.
- **Clock and EOF tests:** clock events close empty time bars; EOF flushes the trailing
  partial; both match the batch oracle.
- All numeric logic stays in C++ (the pull path must drive the C++ engine, not a Python
  reimplementation), so the pure-C++ library and a future WASM binding inherit the
  model.

## Scope note

This is large and cross-cutting: it touches the functor call convention, every stream
operator, the `Dag`, the `*_iter` layer, `Stream`, and `resample`. The implementation
plan should decompose it into sequenced, independently-testable pieces (for example:
callable-dispatch core and cardinality; multi-input unpacking and as-of alignment;
clock and EOF events; `resample` re-signature; retire `*_iter` and `Stream.index`;
docs and notebooks), each keeping `batch == pull == push` green.
