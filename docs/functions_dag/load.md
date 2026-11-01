---
name: load
title: load
kind: function
short: Read a file that save wrote and return the function, Pipeline or live session it holds.
tags:
- save
- load
- restore
- checkpoint
- persistence
- state
see_also:
- loads
- from_settings
- resume
- Pipeline
topics:
- save-restore
---

# `load`

`load(path)` reads a file that `save` wrote and returns the object it holds,
ready to continue where the saved one stopped: a function with its state, a
`Pipeline`, or a live session from `Pipeline.live()`. The file can be JSON or
CBOR; `load` reads the encoding from its first byte.

Every function has the method `save(path)`, which writes its settings (the
constructor arguments) and its state (what it has learned from the data). The
loaded object is built through its screamer class, so it has the same `repr`
and the same behavior as one built by hand. A file that does not fit raises a
`SaveError` (a `ValueError`) whose message starts with the path of the field,
and loads nothing.

`on_mismatch` decides what happens to a state that does not fit the file's
settings, such as an edited `window_size`, or that another state version
wrote. `"error"`, the default, raises `StateMismatch` or `StateVersionError`.
`"fresh"` keeps the settings, starts that object (or the whole live session)
fresh and warns with a `UserWarning`. In JavaScript, `await load(path, {
onMismatch })` reads the file in Node.js. See [Save and
restore](../save_restore.md) for the file layout and the compatibility rule.

<!-- HELP_END -->

## Signature

`load(path, on_mismatch="error")`

- **`path`** - a `str` or `os.PathLike` of a file that `save` wrote.
- **`on_mismatch`** - `"error"` or `"fresh"`.

Returns the operator, `Pipeline` or live session the file holds.

## Examples

```{eval-rst}
.. exec_code::

    import pathlib, tempfile
    import screamer

    path = pathlib.Path(tempfile.mkdtemp()) / "mean.json"
    f = screamer.RollingMean(3)
    for x in [1.0, 2.0, 3.0]:
        f(x)
    f.save(path)

    g = screamer.load(path)
    print(g, g(4.0))   # continues: the mean of 2, 3 and 4
```
