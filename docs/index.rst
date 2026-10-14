Screamer
========


Screamer is a C++ library for researching and trading time series. It computes
rolling statistics, technical indicators, filters, and order-flow measures, runs
costed backtests of single assets, portfolios, and pairs, and tests whether a
result is more than luck. The same functions run on NumPy arrays, on live
streams, and in JavaScript.

New here? :doc:`Your first strategy <recipes/first-strategy>` goes from price
data to a costed backtest and a p-value in one page.


.. code-block:: console

  pip install screamer

Requires Python 3.12 or newer. See :doc:`Installation <installation>` for
prerequisites and development setup.



.. image:: https://img.shields.io/pypi/l/screamer?color=#28A745
   :target: https://github.com/screamer-labs/screamer/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/pypi/pyversions/screamer
   :alt: Python Versions

.. image:: https://github.com/screamer-labs/screamer/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/screamer-labs/screamer/actions/workflows/tests.yml
   :alt: Tests

.. image:: https://readthedocs.org/projects/screamer/badge/?version=latest
   :target: https://screamer.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation

.. image:: https://img.shields.io/pypi/v/screamer
   :target: https://pypi.org/project/screamer/
   :alt: PyPI

A short example
---------------

The example below fits a quadratic to each sliding window of 50 values and returns
its slope at the latest value, then takes the sign of the slope to give the trend
direction (up or down).


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


The plot applies it to one-minute closing prices of the BTC perpetual, built from
six hours of Deribit trades (top), and shows the slope over the previous 50
minutes (middle) and its sign (bottom).


.. plotly::
   :include-source: False

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import RollingPoly2, Sign

    import pandas as pd
    from screamer import Resample

    # One-minute closes from the committed Deribit BTC-perpetual trade slice
    trades = pd.read_csv("notebooks/data/deribit_btc_perp_6h.csv")
    data, ts = Resample(freq=60_000, agg="last")(
        trades["price"].to_numpy(), trades["timestamp"].to_numpy())
    t = pd.to_datetime(ts, unit="ms")

    # Create subplots with specified row heights and shared x-axis
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[1/3, 1/3, 1/3],
        vertical_spacing=0.02
    )

    # slope over a rolling window
    slope = RollingPoly2(window_size=50, derivative_order=1)

    # sign
    sign = Sign()

    # Add traces for input data and results with different derivative orders
    fig.add_trace(go.Scatter(x=t, y=data, mode='lines', name='BTC close'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=slope(data), mode='lines', name='slope(data)', line=dict(color='orange')), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=sign(slope(data)), mode='lines', name='sign(slope(data))', line=dict(color='red')), row=3, col=1)

    # Update layout with titles and axis labels
    fig.update_layout(
        title=None,
        xaxis3_title=None,
        yaxis=dict(title="Price (USD)"),
        yaxis2=dict(title="Slope"),
        yaxis3=dict(title="Sign"),
         margin=dict(l=20, r=20, t=20, b=20)  # Adjust left, right, top, and bottom margins        
    )

    fig.show()


What you can build
------------------

Each tile opens a group of worked examples on the :doc:`What you can build
<capabilities>` page, and the :doc:`gallery <gallery>` shows a plot for every
function.

.. include:: examples_tiles.inc


Speed
-----

screamer runs batch operations as fast as or faster than NumPy and pandas, and
many times faster for rolling-window statistics, because each function updates in
constant time per sample. See the :doc:`Performance <performance>` page for the
full breakdown.


.. image:: /img/speed_chart.png
   :target: /img/speed_chart.png
   :alt: screamer speedup versus the fastest alternative


Causality
---------

Every function is causal: its output depends only on current and past inputs,
so there is no look-ahead bias. Moving from a stored dataset to a live feed
requires no change to the operator logic, since the same C++ engine handles
both; only the surrounding code changes, reading from a live source instead of
an array.



For how functions are built and called, see :doc:`Using screamer <usage>`.


Functions by group
------------------

.. include:: by_group_cards.inc

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Gallery
      :link: gallery
      :link-type: doc

      A plot for every function, grouped like this list.

   .. grid-item-card:: What you can build
      :link: capabilities
      :link-type: doc

      Every recipe and notebook, grouped by task.


.. toctree::
   :maxdepth: 1
   :caption: Get started
   :hidden:

   installation
   recipes/first-strategy
   usage


.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:

   choosing_an_operator
   backtesting_best_practices
   portfolio_backtesting
   javascript


.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:

   capabilities


.. toctree::
   :maxdepth: 1
   :caption: Concepts
   :hidden:

   references_index


.. toctree::
   :maxdepth: 1
   :caption: API reference
   :hidden:

   by_group_index
   gallery


.. toctree::
   :maxdepth: 1
   :caption: Release notes
   :hidden:

   changelog
