---
name: loads
title: loads
kind: function
short: Read a save file from a string or bytes and return the object it holds.
tags:
- save
- load
- restore
- serialize
- json
- cbor
see_also:
- load
- from_settings
- resume
topics:
- save-restore
---

# `loads`

`loads(data)` reads the file that `dumps` returned, a JSON `str` or CBOR
`bytes`, and returns the function, `Pipeline` or live session it holds. It is
`load` without a file, for a file kept in a database, sent over a network, or
held in a test.

Every function has the method `dumps(format="json")`, which returns its save
file as a `str`, or as `bytes` for `format="cbor"` (CBOR, RFC 8949, a binary
encoding of the same content). `loads` reads either encoding and takes the
same `on_mismatch` as `load`. In JavaScript, `loads(data, { onMismatch })`
takes the string or the `Uint8Array` that `dumps` returned.

<!-- HELP_END -->

## Signature

`loads(data, on_mismatch="error")`

- **`data`** - the `str` or `bytes` that `dumps` returned.
- **`on_mismatch`** - `"error"` or `"fresh"`, as for `load`.

## Examples

```{eval-rst}
.. exec_code::

    import screamer

    f = screamer.EwMean(alpha=0.5)
    f(1.0)
    text = f.dumps()
    print(text)

    g = screamer.loads(f.dumps(format="cbor"))
    print(g.dumps() == text)
```
