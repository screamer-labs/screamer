"""The coroutine behind every operator's async iterator.

An operator called on async iterables returns a ``LazyAsyncIterator`` (C++).
Its ``__anext__`` returns :func:`anext_event`, which awaits one item from each
source and passes them to the iterator's ``_step``, the same per-event code the
synchronous iterator runs. nanobind has no ``py::exec``, so the coroutine lives
in this module.
"""


async def anext_event(sources, iterator):
    """Await one item from each async source, then run the event.

    The first exhausted source raises ``StopAsyncIteration``, which ends the
    iteration.
    """
    items = [await source.__anext__() for source in sources]
    return iterator._step(items)
