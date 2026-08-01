"""Async helper coroutines for screamer's streaming operators.

nanobind removed pybind11's ``py::exec`` / ``py::eval``, so the coroutine that
drives an async-generator source through a functor lives here as a real,
importable Python function. The C++ ``AnextAwaitable::__await__`` imports this
module and calls :func:`process_awaitable`.
"""


async def process_awaitable(awaitable, processor):
    """Await one upstream item and run it through ``processor.process_scalar``.

    ``processor`` is the functor's own Python wrapper; ``awaitable`` yields the
    next raw value from the async-generator source.
    """
    result = await awaitable
    return processor.process_scalar(result)
