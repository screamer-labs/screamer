# `Dag`

Define a computation graph once, then run it in batch or live with identical
results. `Input` names a source stream; calling functors and stream operators on
those handles records the graph. `Dag(inputs=[...], outputs=[...])` compiles it
into a callable: `dag(...)` runs it on stored arrays, `dag.stream(...)` runs it
live, event by event, and the two produce byte-identical output.

```{eval-rst}
.. autoclass:: screamer.dag.Input
.. autoclass:: screamer.dag.Dag
   :members: stream
```

## Example

Align two streams and take their difference. Feeds are `(values, index)` pairs
(values-first). The same graph runs in batch and live.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import Input, Dag, Sub
   from screamer.streams import combine_latest
   # --- hide: stop ---
   a, b = Input("a"), Input("b")
   dag = Dag(inputs=[a, b], outputs=[Sub()(combine_latest(a, b))])

   # feeds are (values, index) - values-first
   fa = (np.array([10.0, 20.0, 30.0]), np.array([1, 2, 3]))
   fb = (np.array([1.0, 2.0, 3.0]),   np.array([1, 2, 3]))

   spread, idx = dag(fa, fb)
   print(spread.reshape(-1))
```

Running `dag.stream(fa, fb)` on the same streams returns the identical result,
event by event. See the notebooks for a full walkthrough.
