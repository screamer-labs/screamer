#ifndef SCREAMER_STREAMS_PY_SOURCE_H
#define SCREAMER_STREAMS_PY_SOURCE_H

#include <cstdint>
#include <optional>
#include <utility>
#include <pybind11/pybind11.h>
#include "screamer/streams/event.h"

namespace py = pybind11;

namespace screamer { namespace streams {

// PySource: a Source<Index> that pulls events from a Python iterator.
//
// positional=true  - item is a bare scalar; index = arrival counter (int64).
// positional=false - item is a (value, index) tuple; index extracted as Index.
//
// GIL note: next() is called from C++ while Python drives the puller, so the
// GIL is held. No acquire/release needed.
template <class Index>
class PySource : public Source<Index> {
public:
    PySource(py::object it, bool positional)
        : it_(std::move(it)), positional_(positional), counter_(0) {}

    std::optional<Event<Index>> next() override {
        py::object item;
        try {
            item = it_.attr("__next__")();
        } catch (py::error_already_set& e) {
            // Normal end-of-iterator: swallow StopIteration as flow control.
            if (e.matches(PyExc_StopIteration)) return std::nullopt;
            throw;
        }
        // source is assigned by the consumer from the child slot, not here.
        Event<Index> ev;
        if (positional_) {
            ev.index = static_cast<Index>(counter_++);
            ev.value = item.cast<double>();
        } else {
            py::tuple tup = item.cast<py::tuple>();
            ev.value = tup[0].cast<double>();
            ev.index = tup[1].cast<Index>();
        }
        return ev;
    }

private:
    py::object it_;
    bool positional_;
    std::int64_t counter_;
};

}} // namespace screamer::streams
#endif // SCREAMER_STREAMS_PY_SOURCE_H
