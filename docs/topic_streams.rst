Stream Processing
=================

Stream operators for streams that do not tick together: align them, reshape them,
downsample them, and replay them. Unlike the single-stream compute functors
(which preserve length), these change the shape or cardinality of a stream. See
:doc:`Streams, values, and alignment <multistream>` for the underlying model.

.. toctree::
   :maxdepth: 1
   :hidden:
   :titlesonly:

   functions_streams/combine_latest
   functions_streams/merge
   functions_streams/dropna
   functions_streams/select
   functions_streams/filter
   functions_streams/split
   functions_streams/resample
   functions_streams/pace
