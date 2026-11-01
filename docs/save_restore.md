# Save and restore

Every screamer function, `Pipeline` and live session can be written to a file
and read back. The file holds the object's settings (its constructor
arguments) and its state (what it has learned from the data so far), so a
loaded object continues exactly where the saved one stopped. Python,
JavaScript and C++ read and write the same files: the C++ core writes them,
and the bindings pass the text through. For a worked example, a daily job that
saves a portfolio book and resumes it the next day, see [Save a strategy and
resume it later](guide/live/save-and-resume.md).

```python
import screamer

f = screamer.RollingMean(20)
f.resume(history)            # warm up on stored data, keep the end state
f.save("mean.json")

g = screamer.load("mean.json")
g(next_value)                # continues where f stopped
```

## Calls

Every function has the methods below, and the `screamer` module has the
three loaders. JavaScript has the same calls in its own casing.

| Python | JavaScript | What it does |
|:-------|:-----------|:-------------|
| `obj.save(path)` | `await obj.save(path)` | Write the file: CBOR for a path that ends in `.cbor`, JSON otherwise. |
| `screamer.load(path, on_mismatch="error")` | `await load(path, { onMismatch })` | Read a file and return the object it holds. |
| `obj.dumps(format="json")` | `obj.dumps(format)` | The file as a `str` (JSON) or `bytes` (`"cbor"`); JavaScript returns a string or a `Uint8Array`. |
| `screamer.loads(data, on_mismatch="error")` | `loads(data, { onMismatch })` | Read the file from a string or bytes. |
| `obj.settings()` | `obj.settings()` | The constructor arguments as a dict, by name. |
| `obj.state()` | `obj.state()` | The state, `{"version": N, "values": {...}}`. |
| `obj.set_state(d)` | `obj.setState(d)` | Load a state that `state()` returned. |
| `screamer.from_settings(node)` | `fromSettings(node)` | A fresh object from `{"type": T, "settings": {...}}`. |
| `obj.resume(*inputs)` | `obj.resume(...inputs)` | A batch call that starts from the object's state and keeps the end state. |

`load` returns a function, a `Pipeline` or a live session, whichever the file
holds. In JavaScript, `save` and `load` use `node:fs` and work in Node.js; in
a browser, use `dumps` and `loads`.

A `Pipeline` saves its description, the text `Pipeline.to_json` writes, and
loads as a `Pipeline`. A live session from `Pipeline.live()` saves its
description and its state: each operator's state, the events that wait in a
join or a window, the watermarks, and the outputs that `result()` has not
returned yet. Loading the file gives a live session that continues.

## Resume

A default batch call, `f(x)`, starts from a fresh state and leaves the object
fresh, so calling it twice on the same data gives the same result.
`f.resume(x)` starts from the object's current state, which is the loaded
state after `load`, and keeps the end state in the object. On a fresh object,
`resume` is the warm-up call: it runs the history, after which the object can
be saved or fed one value at a time. Feeding `A` and then `B` gives the same
numbers, bit for bit, as feeding `A`, saving, loading, and feeding `B`.

On a `(T, K)` array, `resume` keeps one state per column (a panel), and the
file holds the `K` states. The object then takes only `(T, K)` arrays: a
`resume` with another `K`, or a call with a single event, raises `ValueError`
until `reset()` or a default batch call drops the panel. A row operator (the
cross-section operators and the portfolio engines such as
`BacktestPortfolioOHLCTarget`) and `PortfolioReport` are one object for `K`
assets: their `resume` continues that one state, and rows of another width
raise `ValueError`.

## The file

A file is a JSON object with three fields:

```json
{
  "screamer": "save/1",
  "library_version": "3.10.1",
  "object": {
    "type": "RollingMean",
    "settings": {
      "window_size": 3,
      "start_policy": "strict"
    },
    "state": {
      "version": 1,
      "values": {
        "buffer": [1.0, 2.0, 0.0],
        "index": 2,
        "count": 2,
        "sum": 3.0
      }
    }
  }
}
```

- `screamer` names the file format, `save/1`. A reader refuses another format.
- `library_version` is the version that wrote the file, a note for people.
  The loader does not check it.
- `object` is a node: the `type` (the class name), its `settings` and its
  `state`.

`settings` lists every constructor parameter under its documented name, in
the order of the function's page, with defaults filled in. An optional
parameter that was not given is `null`, so `EwMean(alpha=0.1)` saves
`"com": null, "span": null, "halflife": null, "alpha": 0.1`. Each number is
written in its parameter's type: an integer setting as `3`, a float setting
as `3.0`. A setting that holds a function, such as the `op` of `Segment`, is
a node of its own with `type` and `settings`.

`state` holds a state version and the values, `{"version": N, "values":
{...}}`, or after a `resume` on a `(T, K)` array `{"version": N, "columns":
[{...}, ...]}` with one entry per column. The values are the members the
function's `reset()` sets: buffers oldest first, counters and running sums. A
function with no state saves empty values, and a file with `"state": null`
holds settings only. A `Pipeline` node has the `Pipeline.to_json` description
as its settings and `"state": null`; a live session node (`"type":
"LiveSession"`) has the same settings and a state with an `engine` part and
one entry per node of the graph.

Numbers read back bit for bit. JSON text spells out the values that JSON has
no syntax for: `NaN`, `Infinity`, `-Infinity`, and `-0.0`. Saving the same
object twice gives the same bytes, so a file under version control diffs
cleanly.

## Hand edits

A file is meant to be read and edited. An edited file goes through the
constructor's checks and through the checks of the state: the length of each
array, the type of each field, and invariants such as a ring's count against
its capacity. An edit the state does not depend on loads. An edit the state
depends on, such as a `window_size` changed from 20 to 50 while the state
holds 20 values, raises `StateMismatch` with the path of the field. A live
session's state also records the graph it was saved with, so an edited graph
(a node added, removed or rewired, or a `Resample` given another `freq`)
raises `StateMismatch`.

`on_mismatch="fresh"` (`{ onMismatch: "fresh" }` in JavaScript) keeps the
file's settings, starts each object whose state does not fit fresh, and warns
with a `UserWarning` (`console.warn` in JavaScript) that names it. It applies
to `StateMismatch` and `StateVersionError`; any other error still raises.

## Errors

Every error of a save or a load is a `SaveError`, a subclass of `ValueError`
(of `RangeError` in JavaScript). Its message starts with the path of the place
it is about, such as `RollingMean.buffer` or
`LiveSession.nodes.3.values.op.count`.

| Error | Raised when |
|:------|:------------|
| `SaveError` | The file is not a save file, is malformed JSON or CBOR, or names an unknown type, an unknown setting or a setting the constructor refuses. |
| `StateMismatch` | The state does not fit the settings: a missing or unknown field, a wrong type or length, a broken invariant, or an edited graph. |
| `StateVersionError` | The file's state version is not the one this library reads. The message names the function and both versions. |
| `NotSaveable` | The object cannot be saved: a lazy call in progress (use a live session), a live session that rejected an event, or a `Pipeline` node whose function is not a screamer class, such as a Python subclass. |

A failed `load` or `set_state` changes nothing: the object keeps the state it
had.

## Compatibility

Settings load in every later release. They are constructor arguments, which
the library's compatibility rules already protect, and a renamed parameter is
mapped to its new name on load.

State loads when its version matches. Each function has a state version, which
changes only when that function's internals change, so a file saved by one
release loads in the next unless that function changed. A file with another
state version raises `StateVersionError`. To recover, load the settings with a
fresh state and warm the object up on history:

```python
f = screamer.load("mean.json", on_mismatch="fresh")
f.resume(history)
```

## CBOR

`save("book.cbor")` and `dumps(format="cbor")` write the same tree in CBOR
([RFC 8949](https://www.rfc-editor.org/rfc/rfc8949)), a binary encoding that
is smaller and faster to read for large states. `load` and `loads` read either
encoding. The encoding keeps the order of the fields, writes every float as an
8-byte float64 and every integer as a CBOR integer, and uses no tags and no
indefinite lengths, so any CBOR library can read the file.

## C++

The C++ headers under `include/screamer/serial` hold the format: each
operator lists its settings and its state once, in its `settings` and `state`
members, and the same lists write, check and read a file. A load checks the
whole file before it changes the object. `serial::make_op(type, settings)`
builds an operator from its type name and settings, which is how the
JavaScript loader builds one.
