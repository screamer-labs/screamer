#ifndef SCREAMER_STREAMS_PY_SOURCE_H
#define SCREAMER_STREAMS_PY_SOURCE_H

#include <cmath>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>
#include <nanobind/nanobind.h>
#include "screamer/streams/event.h"

namespace nb = nanobind;

namespace screamer { namespace streams {

// PySource: a Source<Index> that pulls events from a Python iterator.
//
// positional=true  - item is a bare scalar; index = arrival counter (int64).
// positional=false - item is a (value, index) tuple; index extracted as Index.
//
// Wide events: when item is (sequence_of_floats, index), the value is stored
// in wide_values_ and ev.value is set to wide_values_[0] (for merge ordering
// by index only). Call is_wide() and wide_values() to retrieve the full row.
//
// GIL note: next() is called from C++ while Python drives the puller, so the
// GIL is held. No acquire/release needed.
template <class Index>
class PySource : public Source<Index> {
public:
    PySource(nb::object it, bool positional)
        : it_(std::move(it)), positional_(positional), counter_(0) {}

    std::optional<Event<Index>> next() override {
        nb::object item;
        try {
            item = it_.attr("__next__")();
        } catch (nb::python_error& e) {
            // Normal end-of-iterator: swallow StopIteration as flow control.
            if (e.matches(PyExc_StopIteration)) return std::nullopt;
            throw;
        }
        // source is assigned by the consumer from the child slot, not here.
        Event<Index> ev;
        wide_values_.clear();
        if (positional_) {
            ev.index = static_cast<Index>(counter_++);
            ev.value = nb::cast<double>(item);
        } else {
            nb::tuple tup = nb::cast<nb::tuple>(item);
            nb::object val_obj = nb::borrow<nb::object>(tup[0]);

            // Try scalar extraction first (common fast path).
            // For wide events (tuple, list, or numpy array of floats), fall
            // through to the sequence path.
            bool is_scalar = false;
            if (nb::isinstance<nb::float_>(val_obj) ||
                nb::isinstance<nb::int_>(val_obj)) {
                ev.value = nb::cast<double>(val_obj);
                is_scalar = true;
            }
            if (!is_scalar) {
                // Wide event: extract all values from the sequence.
                nb::sequence seq;
                if (nb::try_cast<nb::sequence>(val_obj, seq)) {
                    const std::size_t n = static_cast<std::size_t>(nb::len(seq));
                    wide_values_.reserve(n);
                    for (std::size_t j = 0; j < n; ++j)
                        wide_values_.push_back(nb::cast<double>(seq[j]));
                    ev.value = wide_values_.empty() ? 0.0 : wide_values_[0];
                } else {
                    // Last resort: try direct cast (handles numpy scalars, etc.)
                    ev.value = nb::cast<double>(val_obj);
                }
            }

            if constexpr (std::is_integral_v<Index>) {
                nb::handle idx = tup[1];
                if (nb::isinstance<nb::float_>(idx)) {
                    double d = nb::cast<double>(idx);
                    if (!std::isfinite(d) || std::floor(d) != d) {
                        throw nb::type_error(
                            "stream index must be a finite integer-valued number; "
                            "got a fractional or non-finite float. The engine is "
                            "int64-indexed.");
                    }
                    ev.index = static_cast<Index>(d);
                } else {
                    ev.index = nb::cast<Index>(idx);
                }
            } else {
                ev.index = nb::cast<Index>(tup[1]);
            }
        }
        return ev;
    }

    // Returns true when the last event was wide (a sequence of floats).
    bool is_wide() const { return !wide_values_.empty(); }

    // Returns the wide values from the last event (valid only when is_wide()).
    const std::vector<double>& wide_values() const { return wide_values_; }

private:
    nb::object it_;
    bool positional_;
    std::int64_t counter_;
    std::vector<double> wide_values_;   // populated when last event was wide
};

}} // namespace screamer::streams
#endif // SCREAMER_STREAMS_PY_SOURCE_H
