"""Python-side helper imported from C++ via nb::module_::import_.

This replaces the pybind11 py::exec(<string>) async bridge, which nanobind
removed. C++ imports this module and calls named functions instead of
compiling a source string at runtime.
"""


def scale(x):
    return x * 10.0


async def stream(source, op):
    """Async-generator rework target: drive a C++ op over an async source.

    Kept here as real importable Python instead of an exec'd string.
    """
    async for item in source:
        yield op(item)
