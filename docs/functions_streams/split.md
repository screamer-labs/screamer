---
name: split
title: split
kind: function
short: Partition a merged tagged stream back into per-source streams.
topics:
- streams
---

# `split`

The inverse of `merge`: partition a source-tagged stream back into one stream per
source. `split(*merge(a_v, b_v, index=[a_k, b_k]))` reconstructs the original
streams.

<!-- HELP_END -->

```{eval-rst}
.. autofunction:: screamer.streams.split
```

## Example

Merge two streams, then split them apart again and read back the first source.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer.streams import merge, split
   # --- hide: stop ---
   a_v = np.array([1.0, 3.0])
   a_k = np.array([1, 3])
   b_v = np.array([2.0, 4.0])
   b_k = np.array([2, 4])

   merged_v, merged_s, merged_k = merge(a_v, b_v, index=[a_k, b_k])
   streams = split(merged_v, merged_s, index=merged_k)
   print(streams[0][0])   # values of source 0
```
