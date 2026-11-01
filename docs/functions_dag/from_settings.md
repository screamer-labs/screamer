---
name: from_settings
title: from_settings
kind: function
short: Build a fresh function from its type and settings, with no state.
tags:
- save
- settings
- clone
- restore
- configuration
see_also:
- load
- loads
- resume
topics:
- save-restore
---

# `from_settings`

`from_settings(node)` builds a fresh function from `{"type": T, "settings":
{...}}`: the `"object"` of a save file, or a dict made from a function's
`settings()`. A `"state"` in the node is ignored, so the new function starts
from no data, as scikit-learn's `clone` does.

`settings()` returns a function's constructor arguments by name, with
defaults filled in and `None` for an optional argument that was not given.
Use `from_settings` to rebuild a function whose saved state another release
cannot read, then warm it up with `resume` on history. An unknown type, an
unknown setting or a setting the constructor refuses raises `SaveError`. In
JavaScript the call is `fromSettings(node)`.

<!-- HELP_END -->

## Signature

`from_settings(node)`

- **`node`** - a dict with `"type"` (the class name) and `"settings"`.

## Examples

```{eval-rst}
.. exec_code::

    import screamer

    f = screamer.RollingZscore(20)
    print(f.settings())

    g = screamer.from_settings({"type": "RollingZscore", "settings": f.settings()})
    print(g)
```
