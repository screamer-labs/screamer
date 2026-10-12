#ifndef SCREAMER_ASYNC_GENERATOR_H
#define SCREAMER_ASYNC_GENERATOR_H

#include <memory>
#include <vector>
#include <nanobind/nanobind.h>
#include "screamer/common/lazy_eval_iterator.h"

namespace nb = nanobind;

namespace screamer {

// True for an async iterable (its type defines __aiter__), such as an async
// generator.
bool is_async_iterable(nb::handle obj);

// True when every input is an async iterable, false when none is. A mix raises
// TypeError.
bool all_async(nb::handle args);

// Routes the inputs of an EvalOp call when they are async iterables. Returns a
// LazyAsyncIterator when every input is one, an invalid object when none is,
// and raises TypeError when sync and async inputs are mixed or the number of
// inputs is neither 1 nor n_in.
nb::object route_async(nb::object op_owner, nb::handle args, std::size_t n_in);

// The async counterpart of LazyEvalIterator. Each __anext__ returns a coroutine
// (screamer/_async.py) that awaits one item from each source and passes them
// to the same EventStep the synchronous iterator uses. Like an async
// generator, it raises RuntimeError when a second __anext__ runs while one is
// awaiting.
class LazyAsyncIterator {
public:
    LazyAsyncIterator(std::unique_ptr<EventStep> stepper, const std::vector<nb::object>& sources);
    // The EvalOp form: n_in() sources, or one whose events hold all n_in() fields.
    LazyAsyncIterator(nb::object op_owner, const std::vector<nb::object>& sources);

    LazyAsyncIterator& __aiter__() { return *this; }
    nb::object __anext__();

    // Called by the coroutine around the awaits: begin() marks an __anext__ as
    // running, end() clears the mark.
    void begin();
    void end() { running_ = false; }

    // Runs one event, given one awaited item per source.
    nb::object step(nb::list items);

private:
    std::unique_ptr<EventStep> stepper_;
    nb::list sources_;                    // the sources' async iterators
    nb::object anext_event_;              // screamer._async.anext_event
    bool running_ = false;
};

} // namespace screamer

#endif // SCREAMER_ASYNC_GENERATOR_H
