#ifndef SCREAMER_ASYNC_GENERATOR_H
#define SCREAMER_ASYNC_GENERATOR_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace screamer {

// Forward declaration of ScreamerBase
class ScreamerBase;

// Function to check if an object is an async generator
bool is_async_generator(const py::object& obj);

// Class representing the awaitable returned by __anext__
class AnextAwaitable {
public:
    AnextAwaitable(py::object awaitable, py::object processor);
    py::object __await__();

private:
    py::object awaitable_;
    py::object processor_owner_;   // the functor's Python wrapper (kept alive)
};

// Class representing the async iterator
class LazyAsyncIterator {
public:
    // `processor` is the functor's own Python wrapper; holding it keeps the
    // functor alive across the whole async iteration, even when it is a
    // transient (e.g. `RollingMean(5)(agen())`).
    LazyAsyncIterator(py::object async_iterable, py::object processor);

    // __aiter__ method
    LazyAsyncIterator& __aiter__();

    // __anext__ method
    py::object __anext__();

private:
    py::object async_iterator_;
    py::object processor_owner_;   // the functor's Python wrapper (kept alive)
};

} // namespace screamer

#endif // SCREAMER_ASYNC_GENERATOR_H
