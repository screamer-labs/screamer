How screamer works
==================

How screamer behaves. Each page explains one part in depth: input and output
shapes, warmup and missing values, streams, pipelines, the microstructure
model, performance, where definitions differ from other libraries, and the
longer guides to backtesting, labels, agents, and JavaScript.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Polymorphic input and output
      :link: polymorphic_api
      :link-type: doc

      How the container, rank, and time axis of your input decide the shape and
      type of the output.

   .. grid-item-card:: Streams in depth
      :link: streaming
      :link-type: doc

      How live calls, iterators, and async iterators handle operator state,
      several sources, asset events, and errors.

   .. grid-item-card:: NaN and warmup
      :link: nan_and_warmup
      :link-type: doc

      What an operator returns before its window fills, and how each one treats
      missing values.

   .. grid-item-card:: Streams, values, and alignment
      :link: multistream
      :link-type: doc

      Indexed event streams, and how several streams are merged, aligned, and
      resampled.

   .. grid-item-card:: Pipelines
      :link: pipelines
      :link-type: doc

      Define a graph of operators once, then run it on stored data or on a live
      feed.

   .. grid-item-card:: Market microstructure
      :link: microstructure
      :link-type: doc

      The trade-signing, order-flow, and price-impact model behind the
      microstructure operators.

   .. grid-item-card:: Performance
      :link: performance
      :link-type: doc

      Benchmarks against other libraries, and what makes the operators fast.

   .. grid-item-card:: Conventions
      :link: conventions
      :link-type: doc

      Where screamer's definitions differ from TA-Lib, pandas, and other
      libraries, and why.

   .. grid-item-card:: Using screamer
      :link: usage
      :link-type: doc

      How a function is built and called, chained, and run on arrays, frames, and streams.

   .. grid-item-card:: Choosing an operator
      :link: choosing_an_operator
      :link-type: doc

      Which function to use for each job, grouped by task.

   .. grid-item-card:: Using screamer with AI agents
      :link: agents
      :link-type: doc

      How an AI agent searches and reads the documentation from the command line or an MCP server.

   .. grid-item-card:: Backtesting best practices
      :link: backtesting_best_practices
      :link-type: doc

      How to model costs and fills, and how to test a backtest result for luck.

   .. grid-item-card:: Portfolios and pairs
      :link: portfolio_backtesting
      :link-type: doc

      How to backtest many assets, or a pair, from one account.

   .. grid-item-card:: Labels and training sets
      :link: labels
      :link-type: doc

      How to build targets for supervised learning and pair them with features.

   .. grid-item-card:: JavaScript and the browser
      :link: javascript
      :link-type: doc

      How to use the WebAssembly package in Node and in the browser.

.. toctree::
   :hidden:

   polymorphic_api
   streaming
   nan_and_warmup
   multistream
   pipelines
   microstructure
   performance
   conventions
   usage
   choosing_an_operator
   agents
   backtesting_best_practices
   portfolio_backtesting
   labels
   javascript
