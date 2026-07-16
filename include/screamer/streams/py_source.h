#ifndef SCREAMER_STREAMS_PY_SOURCE_H
#define SCREAMER_STREAMS_PY_SOURCE_H

#include <cmath>
#include <cstdint>
#include <optional>
#include <type_traits>
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
            if constexpr (std::is_integral_v<Index>) {
                // An integer-indexed consumer (the DAG engine): accept Python
                // ints exactly (no precision loss across the full int64 range)
                // and integer-valued floats (2.0), but reject fractional floats
                // rather than silently flooring them. Mirrors _LazyDag's
                // `int(k) != k` guard. A float-indexed merge (Index=double)
                // takes the branch below and preserves fractional indices.
                py::handle idx = tup[1];
                if (py::isinstance<py::float_>(idx)) {
                    double d = idx.cast<double>();
                    // Reject non-finite (inf/nan) and fractional floats: only a
                    // finite integer-valued float maps to an int64 index. Guarding
                    // finiteness first avoids UB in the static_cast below (casting
                    // inf/nan to int64 is undefined).
                    if (!std::isfinite(d) || std::floor(d) != d) {
                        throw py::type_error(
                            "stream index must be a finite integer-valued number; "
                            "got a fractional or non-finite float. The engine is "
                            "int64-indexed.");
                    }
                    ev.index = static_cast<Index>(d);
                } else {
                    ev.index = idx.cast<Index>();
                }
            } else {
                ev.index = tup[1].cast<Index>();
            }
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
