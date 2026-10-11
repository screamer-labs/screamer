#ifndef SCREAMER_ASYNC_GENERATOR_H
#define SCREAMER_ASYNC_GENERATOR_H

#include <vector>
#include <nanobind/nanobind.h>
#include "screamer/common/lazy_eval_iterator.h"

namespace nb = nanobind;

namespace screamer {

// True for an async iterable (it has __aiter__), such as an async generator.
bool is_async_iterable(nb::handle obj);

// Routes the inputs of an operator call when any of them is an async iterable.
// Returns a LazyAsyncIterator when every input is one, an invalid object when
// none is, and raises TypeError when sync and async inputs are mixed or the
// number of inputs is neither 1 nor n_in.
nb::object route_async(nb::object op_owner, nb::handle args, std::size_t n_in);

// The async counterpart of LazyEvalIterator. `sources` holds n_in() async
// iterables, or one whose events hold all n_in() fields. Each __anext__ returns
// a coroutine (screamer/_async.py) that awaits one item from each source and
// passes them to the same EventStepper the synchronous iterator uses.
class LazyAsyncIterator {
public:
    LazyAsyncIterator(nb::object op_owner, std::vector<nb::object> sources);

    LazyAsyncIterator& __aiter__() { return *this; }
    nb::object __anext__();

    // Runs one event, given one awaited item per source.
    nb::object step(nb::handle items);

private:
    EventStepper stepper_;
    nb::list sources_;                    // the sources' async iterators
    nb::object anext_event_;              // screamer._async.anext_event
    std::vector<nb::object> items_;       // per-event buffer, kept to reuse its storage
};

} // namespace screamer

#endif // SCREAMER_ASYNC_GENERATOR_H
