Screamer
========


Screamer computes rolling statistics, technical indicators, and signal filters on
time series. The same functions run on NumPy arrays and on live streams.


.. code-block:: console

  pip install screamer



.. image:: https://img.shields.io/pypi/l/screamer?color=#28A745
   :target: https://github.com/simu-ai/screamer/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/pypi/pyversions/screamer
   :alt: Python Versions

.. image:: https://github.com/simu-ai/screamer/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/simu-ai/screamer/actions/workflows/tests.yml
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

Screamer's functions are implemented in C++. For the operations shown below, they
run faster than equivalent NumPy and pandas code.


.. image:: /img/speed.png
   :target: /img/speed.png
   :alt: speed comparison


Batch and streaming
-------------------

The same code runs on a stored array or a live stream and produces identical
results, so code tested on historical data can be deployed to production unchanged.

Every function is causal: its output depends only on current and past inputs, not
on future ones, which eliminates look-ahead bias.



For a step-by-step walkthrough see the :doc:`User Guide <usage>`.

.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:

   usage


.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:

   notebooks/01-quickstart-polymorphic-api
   notebooks/02-rolling-and-ew-statistics
   notebooks/03-financial-indicators
   notebooks/04-signal-processing
   notebooks/05-nan-handling
   notebooks/06-streaming-live-events
   notebooks/07-working-with-streams
   notebooks/08-replay-backtest-live
   notebooks/09-computational-dag


.. toctree::
   :maxdepth: 1
   :caption: Functions
   :hidden:

   by_topic/arithmetic
   by_topic/trig
   by_topic/activations
   by_topic/smoothing
   by_topic/filtering
   by_topic/statistics
   by_topic/volatility
   by_topic/standardization
   by_topic/returns
   by_topic/cumulative
   by_topic/trend
   by_topic/momentum
   by_topic/bands
   by_topic/volume
   by_topic/regression
   by_topic/risk
   by_topic/missing-data
   by_topic/outliers
   by_topic/streams
   by_topic/graphs


.. toctree::
   :maxdepth: 1
   :caption: Concepts
   :hidden:

   polymorphic_api
   nan_and_warmup
   multistream
   dag
   conventions


.. toctree::
   :maxdepth: 1
   :caption: Release notes
   :hidden:

   changelog