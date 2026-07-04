# `Dag`

Define a computation graph once, then run it in batch or live with identical
results. `Input` names a source stream; calling functors and combinators on those
handles records the graph. `Dag(inputs=[...], outputs=[...])` compiles it into a
callable: `dag(...)` runs it on stored arrays, `dag.stream(...)` runs it live,
event by event, and the two produce byte-identical output.

```{eval-rst}
.. autoclass:: screamer.dag.Input
.. autoclass:: screamer.dag.Dag
   :members: stream
```

## Example

Align two feeds and take their difference. The same graph runs in batch and live.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import Input, Dag, Sub
   from screamer.streams import combine_latest
   # --- hide: stop ---
   a, b = Input("a"), Input("b")
   dag = Dag(inputs=[a, b], outputs=[Sub()(combine_latest(a, b))])

   fa = (np.array([1, 2, 3]), np.array([10.0, 20.0, 30.0]))
   fb = (np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0]))

   keys, spread = dag(fa, fb)
   print(spread.reshape(-1))
```

Running `dag.stream(fa, fb)` on the same feeds returns the identical result, event
by event. See the notebooks for a full walkthrough.
