Screamer
========

Screamer is a fast C++ library for time series in finance and signal processing:
indicators, filters, order flow, costed backtests and significance tests, with the
same functions on arrays, on live streams and in JavaScript.

.. code-block:: console

  pip install screamer

:doc:`Start the Guide <guide/index>`


One example
-----------

The example fits a quadratic to each sliding window of 50 values, takes its slope
at the latest value, and keeps the sign of the slope: +1 when the trend is up, -1
when it is down.

.. tab-set::

   .. tab-item:: Python

      .. code-block:: python
         :emphasize-lines: 6

         from screamer import RollingPoly2, Sign

         slope = RollingPoly2(window_size=50, derivative_order=1)
         sign = Sign()

         result = sign(slope(data))

   .. tab-item:: JavaScript

      .. code-block:: javascript
         :emphasize-lines: 8

         import { ready, RollingPoly2, Sign } from "@screamer-labs/screamer";

         await ready();

         const slope = RollingPoly2(50, 1);   // windowSize, derivativeOrder
         const sign = Sign();

         const result = sign(slope(data));    // data: Float64Array

The plot applies it to one-minute closes of the BTC perpetual, built from six
hours of Deribit trades: green where the trend is up, red where it is down.

.. plotly::
   :include-source: False

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from screamer import Resample, RollingPoly2, Sign

    # One-minute closes from the committed Deribit BTC-perpetual trade slice
    trades = pd.read_csv("notebooks/data/deribit_btc_perp_6h.csv")
    close, ts = Resample(freq=60_000, agg="last")(
        trades["price"].to_numpy(), trades["timestamp"].to_numpy())
    t = pd.to_datetime(ts, unit="ms")

    trend = Sign()(RollingPoly2(window_size=50, derivative_order=1)(close))

    # Each segment takes the color of the trend at its right end.
    up = trend > 0
    down = trend < 0
    y_up = np.where(up | np.r_[up[1:], False], close, np.nan)
    y_down = np.where(down | np.r_[down[1:], False], close, np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=close, mode="lines", name="BTC close",
                             line=dict(color="lightgray")))
    fig.add_trace(go.Scatter(x=t, y=y_up, mode="lines", name="trend up",
                             line=dict(color="#2ca02c")))
    fig.add_trace(go.Scatter(x=t, y=y_down, mode="lines", name="trend down",
                             line=dict(color="#d62728")))
    fig.update_layout(yaxis=dict(title="Price (USD)"),
                      margin=dict(l=20, r=20, t=20, b=20))
    fig.show()


Guide
-----

.. include:: guide_toc.inc


Why screamer
------------

- Fast: rolling statistics in constant time per sample, see :doc:`Performance <performance>`.
- Causal: every output uses only current and past inputs, see :doc:`Conventions <conventions>`.
- One code path: the same function runs on a stored array and on a live feed, see :doc:`Streams in depth <streaming>`.


.. toctree::
   :maxdepth: 1
   :caption: Guide
   :hidden:

   guide/index


.. toctree::
   :maxdepth: 1
   :caption: Reference
   :hidden:

   reference


.. toctree::
   :maxdepth: 1
   :caption: Release notes
   :hidden:

   Release notes <changelog>
