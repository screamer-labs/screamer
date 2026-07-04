# `split`

The inverse of `merge`: partition a source-tagged stream back into one stream per
source. `split(*merge(a, b))` reconstructs `a` and `b`.

```{eval-rst}
.. autofunction:: screamer.streams.split
```

## Example

Merge two feeds, then split them apart again and read back the first source.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer.streams import merge, split
   # --- hide: stop ---
   tagged = merge((np.array([1, 3]), np.array([1.0, 3.0])),
                  (np.array([2, 4]), np.array([2.0, 4.0])))

   streams = split(*tagged)
   print(streams[0][1])
```
