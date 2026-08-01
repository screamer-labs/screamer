#ifndef SCREAMER_ASYNC_GENERATOR_H
#define SCREAMER_ASYNC_GENERATOR_H

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace screamer {

// Forward declaration of ScreamerBase
class ScreamerBase;

// Function to check if an object is an async generator
bool is_async_generator(const nb::object& obj);

// Class representing the awaitable returned by __anext__
class AnextAwaitable {
public:
    AnextAwaitable(nb::object awaitable, nb::object processor);
    nb::object __await__();

private:
    nb::object awaitable_;
    nb::object processor_owner_;   // the functor's Python wrapper (kept alive)
};

// Class representing the async iterator
class LazyAsyncIterator {
public:
    // `processor` is the functor's own Python wrapper; holding it keeps the
    // functor alive across the whole async iteration, even when it is a
    // transient (e.g. `RollingMean(5)(agen())`).
    LazyAsyncIterator(nb::object async_iterable, nb::object processor);

    // __aiter__ method
    LazyAsyncIterator& __aiter__();

    // __anext__ method
    nb::object __anext__();

private:
    nb::object async_iterator_;
    nb::object processor_owner_;   // the functor's Python wrapper (kept alive)
};

} // namespace screamer

#endif // SCREAMER_ASYNC_GENERATOR_H
