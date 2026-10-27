Screamer
========

Screamer is a fast C++ library for time series in finance and signal processing:
indicators, filters, order flow, costed backtests and significance tests, with the
same functions on arrays, on live streams and in JavaScript.

.. code-block:: console

  pip install screamer

:doc:`Start the Guide <guide/index>`


Detect a trend in four lines
----------------------------

The example fits a straight line to the last 100 values and keeps the sign of
its slope: +1 while the series is rising, -1 while it is falling.

.. tab-set::

   .. tab-item:: Python

      .. code-block:: python
         :emphasize-lines: 6

         from screamer import RollingPoly1, Sign

         slope = RollingPoly1(window_size=100, derivative_order=1)
         sign = Sign()

         trend = sign(slope(close))

   .. tab-item:: JavaScript

      .. code-block:: javascript
         :emphasize-lines: 8

         import { ready, RollingPoly1, Sign } from "@screamer-labs/screamer";

         await ready();

         const slope = RollingPoly1(100, 1);   // windowSize, derivativeOrder
         const sign = Sign();

         const trend = sign(slope(close));     // close: Float64Array

The plot applies it to daily BTC closes from 2022 to 2026: green where the trend
is up, red where it is down. The trend uses only past prices, so it turns a few
weeks after the market does.

.. plotly::
   :include-source: False

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from screamer import RollingPoly1, Sign

    bars = pd.read_csv("notebooks/data/binance_daily_20.csv", parse_dates=["date"])
    btc = bars[bars["symbol"] == "BTC"]
    t, close = btc["date"].to_numpy(), btc["close"].to_numpy()

    trend = Sign()(RollingPoly1(window_size=100, derivative_order=1)(close))

    fig = go.Figure()
    # One shaded band per run of equal trend, from its first day to the next run.
    runs = np.flatnonzero(np.r_[True, trend[1:] != trend[:-1]])
    for i, j in zip(runs, np.r_[runs[1:], len(trend) - 1]):
        if trend[i] in (1.0, -1.0):
            fig.add_vrect(x0=t[i], x1=t[j], line_width=0, opacity=0.18,
                          fillcolor="#2ca02c" if trend[i] > 0 else "#d62728")
    fig.add_trace(go.Scatter(x=t, y=close, mode="lines", name="BTC close",
                             line=dict(color="#333333", width=1.5)))
    fig.update_layout(yaxis=dict(title="Price (USD)"), showlegend=False,
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
