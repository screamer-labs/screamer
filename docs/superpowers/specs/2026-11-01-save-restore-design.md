# Save and restore: settings, state and one file format

Status: draft for approval, 2026-11-01.

## Intent

Users build applications around screamer: live trading bots, scheduled daily jobs,
research notebooks that hand a fitted pipeline to a production process. Today they can
save a Pipeline's structure (`Pipeline.to_json`), but they cannot save what a function
has learned from the data so far. A bot that restarts must replay its whole history, and
a strategy cannot move from Python research to a JavaScript front end with its state.

The goal is the screamer equivalent of saving a scikit-learn pipeline or a PyTorch
`state_dict`: any screamer object, simple or composite, can be written to a file and
read back, in C++, Python or JavaScript, and continues exactly where it stopped. This
makes the library complete as a base for building trading applications.

## Decisions made with the user

1. **One interface for everything.** A single function, a Pipeline and a DAG are
   functionally equivalent: each processes a stream. Saving and loading behaves the same
   for all of them; a composite never acts differently from a single function.
2. **Settings and state are separate things.** Settings are what you construct an object
   with (function name and parameters). State is what it has learned from data so far.
   A file holds both, in separate sections, and each can be used on its own.
3. **One format shared by Python, JavaScript and C++.** The C++ core writes and reads it,
   so all three produce the same bytes for the same object.
4. **Files are for people too.** The default encoding is JSON that people read, diff and
   edit by hand. This is the most flexible option for users.
5. **Compatibility across versions, split by kind** (user accepted the recommendation):
   - Settings stay loadable in every future version. They are constructor arguments,
     public API already protected by the compatibility rules; renamed or deprecated
     parameters are mapped on load.
   - State carries a per-function state version that changes only when that function's
     internals change. A matching version loads; a mismatch refuses with a clear message
     naming the function, and one call recovers: load the settings and warm up from data.
     Converters between state versions are optional, added only when cheap and provably
     exact, never promised.

6. **Batch calls can continue, by explicit opt-in.** A batch call keeps starting fresh by
   default (unchanged behavior). An explicit opt-in (for example `f.resume(new_data)` or
   `f(new_data, resume=True)`; the builder picks one, the same in Python and JavaScript)
   continues from the object's current or loaded state, so a nightly batch job can
   continue where yesterday's run stopped.
7. **Panels keep per-asset state.** An object called on a 2D array (many assets) keeps one
   state per column, so a saved object continues a whole multi-asset book. The file holds
   the per-column states; resuming with a different number of columns refuses with a clear
   error. Today's results for a 2D call do not change.

## Goals

- `save` and `load` for every stateful and stateless object: every functor, every
  stream operator (Resample, CombineLatest, Delay, ...), Pipeline graphs and their live
  sessions, backtest engines (including the account, open orders and pending fills).
- **Continuation guarantee:** feeding A then B gives bit-identical results to feeding A,
  saving, loading, then feeding B. This holds for batch and value-by-value use, after a
  save in one language and a load in another, and through the JSON and binary encodings.
- **Exact numbers:** every double reads back bit for bit (shortest round-trip digits;
  NaN, Infinity, -Infinity and -0.0 written explicitly).
- **Full validation on load:** a loaded file goes through the same checks as the
  constructor, plus shape checks on the state. Errors name the exact place, for example
  `nodes[3].settings.window_size: must be >= 1, got 0`.
- **Readable files:** stable key order, indented, parameter names as documented. Saving
  the same object twice gives the same bytes, so a git diff shows only real changes.

## Non-goals

- Pickle or any format that can execute code on load. Loading only constructs registered
  screamer classes.
- Saving user Python or JavaScript callables (lambdas inside a graph). If a graph holds
  one, saving it raises and names the node.
- Saving input data or results. A file holds settings and state only.
- Promising state compatibility across every version (see decision 5).

## Anti-goals (a failure even if it technically works)

- A restored object that silently computes something different from the original.
- Python and JavaScript writing different files for the same object.
- A per-sample cost: state support must add no work and no memory to the hot path.
- Every operator needing a large amount of hand-written save code that can drift from
  its `reset()`; the state of an operator must be described once.

## The file

A file is one recursive shape. A composite's settings list its parts, and each part has
its own settings and state in the same shape a single function uses, so a single
function's file is the smallest case and one node's entry can be cut out and loaded on
its own. Sketch (the builder fixes the exact field names):

```json
{
  "screamer": "save/1",
  "library_version": "3.11.0",
  "object": {
    "type": "RollingMean",
    "settings": {"window_size": 3, "start_policy": "strict"},
    "state": {
      "version": 1,
      "values": {"buffer": [1.0, 2.0, NaN], "head": 2, "count": 3, "sum": NaN}
    }
  }
}
```

- The settings section reuses the existing Pipeline description (graph_json.h:
  `{"functor": name, "params": {...}}`, canonical names, defaults filled in), so the
  current `Pipeline.to_json` output becomes the settings-only case of the new file and
  existing saved graphs keep loading.
- `"state"` is absent (or `null`) for a settings-only file and for stateless functions,
  which save an empty state, so code that saves objects never asks which kind it holds.
- Encodings: JSON (default) and a binary twin (CBOR, RFC 8949) holding the same tree, for
  large states. CBOR is optional and never required to use the feature.

### Hand edits

- Editing settings or state values is allowed. Whatever loads is fully validated.
- Editing a setting that the saved state depends on (for example `window_size` from 20
  to 50 while the state holds 20 values) refuses by default and explains why. An explicit
  option keeps the new settings and starts that function fresh, with a warning.
- Editing state values directly is allowed whenever shapes and invariants still hold
  (the loader checks them, for example a ring's count against its capacity).

## Interface

The same names in Python and JavaScript (JavaScript in its own casing):

- `obj.save(path)` writes settings and state; `screamer.load(path)` returns a
  ready-to-run object of the same kind.
- In memory, without files: `obj.to_dict()` / `screamer.from_dict(...)` and text or bytes
  variants. The builder picks the final names, consistent with the existing
  `Pipeline.to_json` / `from_json`, which keep working.
- Finer control: `obj.settings()`, `obj.state()`, `obj.set_state(...)`, and
  `screamer.from_settings(...)` to build a fresh object with no state (like sklearn's
  `clone`).
- Warm-up after a refused state: load the settings and feed history, as one documented
  call.

**My call, for review:** a Pipeline object itself holds no streaming state (each batch
call resets it, and live sessions run on their own copies of the operators). So saving a
Pipeline saves its settings, and saving a live session saves settings plus state;
loading a session file gives a live session that continues. A single functor instance is
both at once. This keeps the user's rule (the same calls, the same file shape, the same
continuation guarantee) while matching how objects already behave.

## Architecture (what is decided; the rest is the builder's)

- **All logic in C++.** Python and JavaScript are thin bindings. Each operator exposes
  its state through one C++ description that serves save, load and the shape checks.
  This should also be the natural place to make `reset()` agree with the state (a reset
  object equals a freshly constructed one), so state and reset cannot drift.
- **Inspired by Boost.Serialization and cereal:** one `serialize`-style member per
  operator over a generic archive (writer, reader, validator), versioned per class;
  construction data (settings) kept apart from the serialized state, as Boost's
  `save_construct_data` does; a registry by type name (the existing help.json /
  manifest names) for polymorphic loading. Prior art also considered: PyTorch
  `state_dict` with strict loading, Keras `get_config` / weights, Flink savepoints with
  stable node ids and refusal of incompatible state.
- **Settings capture in C++:** today C++ operators do not keep their constructor
  arguments (Python records them in `_functor_params.py`, JavaScript in its factories).
  For a file written by C++ to be identical from both languages, settings need one
  source of truth; the builder decides whether C++ operators store them or the bindings
  keep supplying them through the existing schema.
- **Composite state:** nested operators, Pipeline nodes, stream-operator buffers
  (pending events, watermarks, partial windows) and backtest accounts are all state, and
  serialize recursively. Node identity in a Pipeline file is the graph's stable node id.
- **Existing JSON code:** reuse the locale-independent, shortest-digits JSON writer and
  parser in graph_json.h, extending it as needed (for example -0.0).

## Testing

- Continuation test for every exported operator (generated from the registry, so a new
  operator cannot ship without it): random split point, save, load, continue, compare bit
  for bit with an uninterrupted run; through JSON and CBOR; Python to JavaScript and back.
- Golden files committed per operator and state version, so a change to an operator's
  state layout without a version bump fails CI.
- Fuzzed hand edits: corrupted or edited files must load correctly or fail with a named
  error, never crash or produce a silently wrong object.
- Benchmarks show no change in per-sample speed.

## Calls made after the builder check (mine, for review)

- **Graph structure must match.** State loads into a Pipeline only when its structure
  matches the saved one (node ids are positions in the graph walk). Any structural edit is
  a settings change: refused by default, or the explicit fresh-start option restarts the
  whole graph. Matching state into an edited graph by user-assigned node names is a
  possible later extension, out of scope now.
- **A live session saves everything it owes.** Its file holds operator state plus pending
  events, partial windows, watermarks and outputs produced but not yet collected, so a
  restart loses nothing. A session that has failed (poisoned) cannot be saved; the error
  says so.
- **Scope.** Functors, stream operators, backtest engines, Pipelines and live sessions are
  saveable. A lazy (iterator) call in progress is not; the docs point to a live session
  for that case.
- **`Pipeline.to_json` stays as it is.** Its output and schema (`screamer_dag: 1`) are
  unchanged, and the new file's settings section embeds that description as is, so
  existing files and parsers keep working.
- **C++ stores its own settings.** Each C++ operator keeps its constructor settings, so a
  C++-only user can save, and Python and JavaScript write identical bytes from one source
  of truth. The Python and JavaScript settings capture becomes a thin read of it.

## Left to the builder

Exact field names, the C++ archive design, how the ~200 operators are migrated (in
tranches is fine, with the registry test listing what is missing until complete), CBOR
implementation choice (in-house is preferred over a new dependency), the warm-up call's
exact form, and docs (a Guide page "Save a strategy and resume it later" plus reference).
