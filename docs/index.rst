Screamer
========


Screamer computes rolling statistics, technical indicators, and signal filters on
time series. The same functions run on NumPy arrays and on live streams.


.. code-block:: console

  pip install screamer



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

The example below fits a line to each sliding window of 50 values and returns its
slope, then takes the sign of the slope to give the trend direction (up or down).


.. code-block:: python
   :emphasize-lines: 6

    from screamer import RollingPoly2, Sign

    slope = RollingPoly2(window_size=50, derivative_order=1)
    sign = Sign()

    result = sign(slope(data))


The plot shows the input data (top, blue), the slope over the previous 50 samples
(middle, orange), and the sign of the slope (bottom, red).


.. plotly::
   :include-source: False

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import RollingPoly2, Sign

    # Generate example data
    np.random.seed(0)
    data = np.cumsum(np.random.normal(size=300))

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
    fig.add_trace(go.Scatter(y=data, mode='lines', name='data'), row=1, col=1)
    fig.add_trace(go.Scatter(y=slope(data), mode='lines', name='slope(data)', line=dict(color='orange')), row=2, col=1)
    fig.add_trace(go.Scatter(y=sign(slope(data)), mode='lines', name='sign(slope(data))', line=dict(color='red')), row=3, col=1)

    # Update layout with titles and axis labels
    fig.update_layout(
        title=None,
        xaxis3_title="Sample index",
        yaxis=dict(title="Data"),
        yaxis2=dict(title="Slope"),
        yaxis3=dict(title="Sign"),
         margin=dict(l=20, r=20, t=20, b=20)  # Adjust left, right, top, and bottom margins        
    )

    fig.show()


Speed
-----

screamer runs batch operations as fast as or faster than NumPy and pandas, and
many times faster for rolling-window statistics, because each function updates in
constant time per sample. See the :doc:`Performance <performance>` page for the
full breakdown.


.. image:: /img/speed_chart.png
   :target: /img/speed_chart.png
   :alt: screamer speedup versus the fastest alternative


Batch and streaming
-------------------

The same code runs on a stored array or a live stream and produces identical
results, so code tested on historical data can be deployed to production unchanged.

Every function is causal: its output depends only on current and past inputs,
which eliminates look-ahead bias.



For a step-by-step walkthrough see the :doc:`User Guide <usage>`.


.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Statistics
      :link: by_group/statistics
      :link-type: doc

      Central tendency, dispersion, quantiles, moments, and correlation.

   .. grid-item-card:: Smoothing & filters
      :link: by_group/smoothing-filters
      :link-type: doc

      Moving averages and designed filters.

   .. grid-item-card:: Technical indicators
      :link: by_group/indicators
      :link-type: doc

      Trend, momentum, bands, volume, and returns.

   .. grid-item-card:: Market microstructure
      :link: by_group/microstructure
      :link-type: doc

      Trade signing, imbalance, price impact, and arrivals.

   .. grid-item-card:: Backtesting & risk
      :link: by_group/backtesting-risk
      :link-type: doc

      Costed equity curves, drawdown, and performance.

   .. grid-item-card:: Math & logic
      :link: by_group/math-logic
      :link-type: doc

      Elementwise numeric, trigonometric, activation, and boolean operations.

   .. grid-item-card:: Data preparation
      :link: by_group/data-prep
      :link-type: doc

      Fill or drop missing values, and despike outliers.

   .. grid-item-card:: Streaming & pipelines
      :link: by_group/streaming
      :link-type: doc

      Align and reshape event streams, and build a runnable pipeline.

   .. grid-item-card:: Examples
      :link: notebooks/index
      :link-type: doc

      Runnable notebooks grouped by task.


.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:

   usage


.. toctree::
   :maxdepth: 1
   :caption: Functions
   :hidden:

   by_group_index
   by_group/statistics
   by_group/smoothing-filters
   by_group/indicators
   by_group/microstructure
   by_group/backtesting-risk
   by_group/math-logic
   by_group/data-prep
   by_group/streaming


.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:

   notebooks/index


.. toctree::
   :maxdepth: 1
   :caption: References
   :hidden:

   polymorphic_api
   nan_and_warmup
   multistream
   pipelines
   microstructure
   performance
   conventions


.. toctree::
   :maxdepth: 1
   :caption: Release notes
   :hidden:

   changelog