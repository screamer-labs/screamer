#ifndef SCREAMER_STREAMS_PY_SOURCE_H
#define SCREAMER_STREAMS_PY_SOURCE_H

#include <cmath>
#include <limits>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>
#include <string>
#include <nanobind/nanobind.h>
#include "screamer/common/cast_double.h"
#include "screamer/common/gc.h"
#include "screamer/streams/py_index.h"
#include "screamer/streams/event.h"
#include <cstddef>

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
// A wrong-kind item raises TypeError naming `owner`, the event and the input
// (e.g. "Pipeline: the value of event 3 of input 0 is a str, expected a
// number"); the message is built only on that error path.
//
// GIL note: next() is called from C++ while Python drives the puller, so the
// GIL is held. No acquire/release needed.
template <class Index>
class PySource : public Source<Index> {
public:
    // `unit`, when given, is the index kind shared by every source of the call
    // (py_index.h TimeUnit); it must outlive the source.
    PySource(nb::object it, bool positional, std::size_t input = 0,
             const char* owner = "stream", screamer::TimeUnit* unit = nullptr)
        : it_(std::move(it)), positional_(positional), counter_(0), event_(0),
          input_(input), owner_(owner), unit_(unit) {}

    std::optional<Event<Index>> next() override {
        if (!it_.is_valid()) return std::nullopt;   // cleared by the garbage collector
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
        const std::size_t event = event_++;
        if (positional_) {
            ev.index = static_cast<Index>(counter_++);
            if constexpr (std::is_integral_v<Index>) {
                try {
                    screamer::note_number(unit_);         // an arrival count
                } catch (const screamer::dag::IndexTypeError& e) {
                    throw screamer::dag::IndexTypeError(std::string(owner_) + ": " +
                        where(event, "the index of event") + ": " + e.what());
                }
            }
            if (!nb::try_cast<double>(item, ev.value)) {
                if (!item.is_none()) bad(event, "event", item);
                ev.value = kNaN;                    // a missing value (A9)
            }
        } else {
            nb::tuple tup;
            if (!nb::try_cast<nb::tuple>(item, tup) || tup.size() < 2)
                bad(event, "event", item, "a (value, index) pair");
            nb::object val_obj = nb::borrow<nb::object>(tup[0]);

            // Try scalar extraction first (common fast path).
            // For wide events (tuple, list, or numpy array of floats), fall
            // through to the sequence path.
            PyObject* vp = val_obj.ptr();
            bool is_scalar = false;
            if (PyFloat_Check(vp) || PyLong_Check(vp)) {
                if (!nb::try_cast<double>(val_obj, ev.value))
                    bad(event, "the value of event", val_obj);
                is_scalar = true;
            } else if (vp == Py_None) {
                ev.value = kNaN;                    // a missing value (A9)
                is_scalar = true;
            } else if (PyUnicode_Check(vp) || PyBytes_Check(vp) || PyByteArray_Check(vp)) {
                bad(event, "the value of event", val_obj);   // text, not a sequence
            }
            if (!is_scalar) {
                // Wide event: extract all values from the sequence.
                nb::sequence seq;
                if (nb::try_cast<nb::sequence>(val_obj, seq)) {
                    const std::size_t n = static_cast<std::size_t>(nb::len(seq));
                    if (n == 0)   // as a live push: an event has at least one value
                        throw nb::value_error((std::string(owner_) + ": " +
                            where(event, "the value of event") + " is empty").c_str());
                    wide_values_.reserve(n);
                    for (std::size_t j = 0; j < n; ++j) {
                        nb::object x = seq[j];
                        double d;
                        if (!nb::try_cast<double>(x, d)) {
                            if (x.is_none()) { wide_values_.push_back(kNaN); continue; }
                            const std::string w = where(event, "the value of event");
                            throw_if_wrong_kind(owner_, val_obj, w);
                            throw_bad_shape(owner_, "item [" + std::to_string(j) + "] of " + w, x);
                        }
                        wide_values_.push_back(d);
                    }
                    ev.value = wide_values_[0];
                } else {
                    // Last resort: try direct cast (handles numpy scalars, etc.)
                    if (!nb::try_cast<double>(val_obj, ev.value))
                        bad(event, "the value of event", val_obj);
                }
            }

            if constexpr (std::is_integral_v<Index>) {
                // The engine's index rule, as in a live push (py_index.h), with
                // the event's position in the message.
                try {
                    ev.index = static_cast<Index>(screamer::py_index(tup[1], unit_));
                } catch (const screamer::dag::IndexTypeError& e) {
                    throw screamer::dag::IndexTypeError(
                        std::string(owner_) + ": " + where(event, "the index of event") +
                        ": " + e.what());
                }
            } else {
                nb::handle idx = tup[1];
                if (!nb::try_cast<Index>(idx, ev.index)) bad(event, "the index of event", idx);
            }
        }
        return ev;
    }

    // Garbage-collector support for the class that owns the source (gc.h).
    int traverse(visitproc visit, void* arg) const { return gc_visit(it_, visit, arg); }
    // Drops the iterator; the source then ends.
    void clear() { nb::object it = std::move(it_); }

    // Returns true when the last event was wide (a sequence of floats).
    bool is_wide() const { return !wide_values_.empty(); }

    // Returns the wide values from the last event (valid only when is_wide()).
    const std::vector<double>& wide_values() const { return wide_values_; }

private:
    // The wrong-kind TypeError for `x`, `what` naming it ("the value of event").
    std::string where(std::size_t event, const char* what) const {
        return where_text(what, static_cast<std::ptrdiff_t>(event),
                          static_cast<std::ptrdiff_t>(input_));
    }
    [[noreturn]] void bad(std::size_t event, const char* what, nb::handle x) const {
        throw_not_a_number(owner_, x, where(event, what));
    }
    [[noreturn]] void bad(std::size_t event, const char* what, nb::handle x,
                          const char* expected) const {
        throw_wrong_kind(owner_, where(event, what), x, expected);
    }

    nb::object it_;
    bool positional_;
    std::int64_t counter_;
    std::size_t event_;                 // items read, for error messages
    std::size_t input_;                 // this source's input position
    const char* owner_;                 // "Pipeline", "stream"
    std::vector<double> wide_values_;   // populated when last event was wide
    screamer::TimeUnit* unit_;          // the call's index kind, shared; may be null
    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
};

}} // namespace screamer::streams
#endif // SCREAMER_STREAMS_PY_SOURCE_H
