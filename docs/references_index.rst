Concepts
========

How screamer behaves. Each page explains one part in depth: input and output
shapes, warmup and missing values, streams, pipelines, the microstructure
model, performance, and where definitions differ from other libraries.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Polymorphic input and output
      :link: polymorphic_api
      :link-type: doc

      How the container, rank, and time axis of your input decide the shape and
      type of the output.

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

.. toctree::
   :hidden:

   polymorphic_api
   nan_and_warmup
   multistream
   pipelines
   microstructure
   performance
   conventions
