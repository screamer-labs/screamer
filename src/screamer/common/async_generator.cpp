#include "screamer/common/async_generator.h"
#include <utility>

namespace screamer {

bool is_async_generator(const nb::object& obj) {
    return nb::hasattr(obj, "__aiter__") && nb::hasattr(obj, "__anext__");
}

// Implementation of AnextAwaitable
AnextAwaitable::AnextAwaitable(nb::object awaitable, nb::object processor)
    : awaitable_(std::move(awaitable)),
      processor_owner_(std::move(processor)) {}

nb::object AnextAwaitable::__await__() {
    // The coroutine that awaits the upstream item and runs it through the
    // functor lives in screamer/_async.py. nanobind has no py::exec, so the
    // coroutine ships as a real importable module instead of a code string.
    nb::object mod = nb::module_::import_("screamer._async");
    nb::object coro = mod.attr("process_awaitable")(awaitable_, processor_owner_);
    return coro.attr("__await__")();
}


// Implementation of LazyAsyncIterator

LazyAsyncIterator::LazyAsyncIterator(nb::object async_iterable, nb::object processor)
    : async_iterator_(async_iterable.attr("__aiter__")()),
      processor_owner_(std::move(processor)) {}

LazyAsyncIterator& LazyAsyncIterator::__aiter__() {
    return *this;
}

nb::object LazyAsyncIterator::__anext__() {
    // Get the next item from the async iterator
    nb::object awaitable = async_iterator_.attr("__anext__")();
    // Return an AnextAwaitable that will process the item. Pass the same
    // functor wrapper so it too keeps the functor alive for the await.
    return nb::cast(AnextAwaitable(std::move(awaitable), processor_owner_));
}

} // namespace screamer
