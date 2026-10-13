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
#include "screamer/common/py_iter.h"
#include "screamer/streams/py_index.h"
#include "screamer/streams/event.h"
#include <cstddef>
#include <cstring>
#include <algorithm>

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
        nb::object item = next_item(it_);
        if (!item.is_valid()) return std::nullopt;   // the end of the iterator
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
            if (rows_) {
                read_value(item, event, ev, "event");   // a number or a row
            } else if (!nb::try_cast<double>(item, ev.value)) {
                if (!item.is_none()) bad(event, "event", item);
                ev.value = kNaN;                    // a missing value (A9)
            }
        } else {
            nb::tuple tup;
            if (!nb::try_cast<nb::tuple>(item, tup) || tup.size() < 2)
                bad(event, "event", item, "a (value, index) pair");
            read_value(nb::borrow<nb::object>(tup[0]), event, ev);

            if constexpr (std::is_integral_v<Index>) {
                // The engine's index rule, as in a live push (py_index.h), with
                // the event's position in the message. The index must not
                // decrease (checked_index.h), checked after the kind rules.
                try {
                    ev.index = static_cast<Index>(screamer::py_index(tup[1], unit_));
                } catch (const screamer::dag::IndexTypeError& e) {
                    throw screamer::dag::IndexTypeError(
                        std::string(owner_) + ": " + where(event, "the index of event") +
                        ": " + e.what());
                }
            } else {
                // A float index (a float merge): numbers of any value, under the
                // call's one-kind rule (py_index.h); a time index is ticks, and
                // after a number it raises the kind error.
                nb::handle idx = tup[1];
                try {
                    ev.index = static_cast<Index>(screamer::py_float_index(idx, unit_));
                } catch (const screamer::dag::IndexTypeError& e) {
                    throw screamer::dag::IndexTypeError(
                        std::string(owner_) + ": " + where(event, "the index of event") +
                        ": " + e.what());
                }
            }
            if (event != 0 && ev.index < last_index_)
                dag::throw_index_decreases(input_, event, last_index_, ev.index);
            last_index_ = ev.index;
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

    // The shape of the last wide event: (n,) for a sequence, the array's
    // shape for a float64 buffer (valid only when is_wide()).
    const std::vector<std::size_t>& event_shape() const { return event_shape_; }

    // Lets a float64 array event of rank 2 or more be read as its values in
    // C order (one wide event), as a batch call reads a (T, *A) array. Off by
    // default: a Pipeline stream holds (T,) or (T, w) values.
    void set_flatten(bool on) { flatten_ = on; }

    // The number of items read so far (the last event is events_read() - 1).
    std::size_t events_read() const { return event_; }

    // The text naming event `event` of this input in an error message.
    std::string event_text(std::size_t event) const { return where(event, "event"); }
    const char* owner() const { return owner_; }

    // Lets a positional source take rows as events (a sequence or an array
    // of numbers), as an indexed source does. Off by default: a positional
    // Pipeline stream holds numbers.
    void set_rows(bool on) { rows_ = on; }

private:
    // Reads the value of an event: a number (None is NaN), or a row of
    // numbers into wide_values_ with its shape in event_shape_.
    // `what` names the value in an error: "the value of event" for an
    // indexed event, "event" for a positional one.
    void read_value(nb::object val_obj, std::size_t event, Event<Index>& ev,
                    const char* what = "the value of event") {
        // Try scalar extraction first (common fast path).
        // For wide events (tuple, list, or numpy array of floats), fall
        // through to the sequence path.
        PyObject* vp = val_obj.ptr();
        bool is_scalar = false;
        if (PyFloat_Check(vp) || PyLong_Check(vp)) {
            if (!nb::try_cast<double>(val_obj, ev.value))
                bad(event, what, val_obj);
            is_scalar = true;
        } else if (vp == Py_None) {
            ev.value = kNaN;                    // a missing value (A9)
            is_scalar = true;
        } else if (PyUnicode_Check(vp) || PyBytes_Check(vp) || PyByteArray_Check(vp)) {
            bad(event, what, val_obj);   // text, not a sequence
        }
        if (!is_scalar && read_float64_buffer(vp, event)) {
            ev.value = wide_values_[0];
        } else if (!is_scalar && flatten_ && read_any_rank(val_obj, event)) {
            ev.value = wide_values_[0];
        } else if (!is_scalar) {
            // Wide event: extract all values from the sequence.
            nb::sequence seq;
            if (nb::try_cast<nb::sequence>(val_obj, seq)) {
                const std::size_t n = static_cast<std::size_t>(nb::len(seq));
                if (n == 0)   // as a live push: an event has at least one value
                    throw nb::value_error((std::string(owner_) + ": " +
                        where(event, what) + " is empty").c_str());
                wide_values_.reserve(n);
                for (std::size_t j = 0; j < n; ++j) {
                    nb::object x = seq[j];
                    double d;
                    if (!nb::try_cast<double>(x, d)) {
                        if (x.is_none()) { wide_values_.push_back(kNaN); continue; }
                        const std::string w = where(event, what);
                        throw_if_wrong_kind(owner_, val_obj, w);
                        throw_bad_shape(owner_, "item [" + std::to_string(j) + "] of " + w, x);
                    }
                    wide_values_.push_back(d);
                }
                ev.value = wide_values_[0];
                event_shape_.assign(1, n);
            } else {
                // Last resort: try direct cast (handles numpy scalars, etc.)
                if (!nb::try_cast<double>(val_obj, ev.value))
                    bad(event, what, val_obj);
            }
        }
    }

    // Reads a C-contiguous native float64 buffer of rank 1 (or more, under
    // set_flatten) into wide_values_ with one copy. Returns false, with no
    // error set, for any other object, which then takes the sequence path.
    bool read_float64_buffer(PyObject* vp, std::size_t event) {
        if (!PyObject_CheckBuffer(vp)) return false;
        Py_buffer view;
        if (PyObject_GetBuffer(vp, &view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) != 0) {
            PyErr_Clear();
            return false;
        }
        const char* f = view.format ? view.format : "B";
        if (*f == '@' || *f == '=' || (*f == '<' && little_endian())) ++f;
        const bool ok = f[0] == 'd' && f[1] == '\0' &&
                        view.itemsize == static_cast<Py_ssize_t>(sizeof(double)) &&
                        (view.ndim == 1 || (view.ndim > 1 && flatten_));
        if (ok) {
            const std::size_t n = static_cast<std::size_t>(view.len) / sizeof(double);
            if (n == 0) {
                PyBuffer_Release(&view);
                throw nb::value_error((std::string(owner_) + ": " +
                    where(event, "the value of event") + " is empty").c_str());
            }
            wide_values_.resize(n);
            std::memcpy(wide_values_.data(), view.buf, n * sizeof(double));
            event_shape_.resize(static_cast<std::size_t>(view.ndim));
            for (int d = 0; d < view.ndim; ++d)
                event_shape_[static_cast<std::size_t>(d)] =
                    static_cast<std::size_t>(view.shape[d]);
        }
        PyBuffer_Release(&view);
        return ok;
    }
    // The slow path of set_flatten: a numeric event of any rank, dtype or
    // layout (a strided or Fortran-order array, float32, int64, bool,
    // float16, another byte order, or nested lists and tuples), read in C
    // order with its shape, as a batch call reads it. Returns false for an
    // object this path does not take (a flat sequence of numbers, which the
    // sequence path reads with its rules).
    bool read_any_rank(nb::handle v, std::size_t event) {
        PyObject* p = v.ptr();
        if (PyObject_CheckBuffer(p)) {
            wide_values_.clear();
            event_shape_.clear();
            shape_fixed_ = false;
            path_.clear();
            if (!read_strided(p, event, v, 0)) return false;
        } else {
            if (!(PyList_Check(p) || PyTuple_Check(p)) || PySequence_Size(p) <= 0) return false;
            nb::object item0 = nb::steal(PySequence_GetItem(p, 0));
            if (!item0.is_valid()) throw nb::python_error();
            PyObject* f = item0.ptr();
            const bool nested = PyObject_CheckBuffer(f) ? !PyBytes_Check(f) && !PyByteArray_Check(f)
                                                        : (PyList_Check(f) || PyTuple_Check(f));
            if (!nested) return false;
            wide_values_.clear();
            event_shape_.clear();
            shape_fixed_ = false;
            path_.clear();
            read_nested(v, 0, event);
        }
        if (wide_values_.empty())
            throw nb::value_error((std::string(owner_) + ": " +
                where(event, "the value of event") + " is empty").c_str());
        return true;
    }

    // Reads a buffer of rank >= 1 at depth `depth` of the event's shape.
    // Returns false (no error) for a rank-0 buffer (a NumPy scalar, read as a
    // number) or an object without a readable buffer.
    bool read_strided(PyObject* p, std::size_t event, nb::handle whole, std::size_t depth) {
        Py_buffer view;
        if (PyObject_GetBuffer(p, &view, PyBUF_RECORDS_RO) != 0) {
            PyErr_Clear();
            return false;
        }
        struct Release { Py_buffer* b; ~Release() { PyBuffer_Release(b); } } release{&view};
        if (view.ndim == 0) return false;   // a NumPy scalar: read as a number
        const char* f = view.format ? view.format : "B";
        bool swap = false;
        if (*f == '@' || *f == '=') ++f;
        else if (*f == '<') { swap = !little_endian(); ++f; }
        else if (*f == '>' || *f == '!') { swap = little_endian(); ++f; }
        const char code = f[0];
        if (code == 'O' && f[1] == '\0' && nb::hasattr(whole, "tolist")) {
            // An array of Python objects: read its items as nested lists.
            nb::object items = whole.attr("tolist")();
            read_nested(items, depth, event);
            return true;
        }
        if (code == '\0' || f[1] != '\0' || !known_code(code, view.itemsize)) {
            std::string kind = std::string("format '") + (view.format ? view.format : "") + "'";
            if (PyObject* dt = PyObject_GetAttrString(whole.ptr(), "dtype")) {
                nb::object dtype = nb::steal(dt);
                if (PyObject* txt = PyObject_Str(dtype.ptr())) {
                    nb::object t = nb::steal(txt);
                    Py_ssize_t len = 0;
                    if (const char* c = PyUnicode_AsUTF8AndSize(t.ptr(), &len))
                        kind.assign(c, static_cast<std::size_t>(len));
                }
            }
            PyErr_Clear();
            throw nb::type_error((std::string(owner_) + ": " +
                where(event, "the value of event") + " is an array of " + kind +
                ", expected numbers").c_str());
        }
        // The shape: set by the first item at this depth, checked after.
        for (int d = 0; d < view.ndim; ++d)
            fix_dim(depth + static_cast<std::size_t>(d),
                    static_cast<std::size_t>(view.shape[d]), event);
        // The buffer's items are leaves: the shape is complete at its depth.
        if (depth + static_cast<std::size_t>(view.ndim) != event_shape_.size()) ragged(event);
        shape_fixed_ = true;
        std::size_t n = 1;
        for (int d = 0; d < view.ndim; ++d) n *= static_cast<std::size_t>(view.shape[d]);
        if (n == 0) return true;
        std::vector<Py_ssize_t> at(static_cast<std::size_t>(view.ndim), 0);
        for (std::size_t k = 0; k < n; ++k) {
            const char* item = static_cast<const char*>(view.buf);
            for (int d = 0; d < view.ndim; ++d)
                item += at[static_cast<std::size_t>(d)] * (view.strides ? view.strides[d]
                                                                         : view.itemsize);
            wide_values_.push_back(read_item(item, code, view.itemsize, swap));
            for (int d = view.ndim - 1; d >= 0; --d) {
                if (++at[static_cast<std::size_t>(d)] < view.shape[d]) break;
                at[static_cast<std::size_t>(d)] = 0;
            }
        }
        return true;
    }

    // The deepest nesting of an event (NumPy's most dimensions): a deeper
    // list raises ValueError before the recursion can exhaust the C stack.
    static constexpr std::size_t kMaxEventDepth = 64;

    // Nested lists and tuples (items may be arrays): rectangular, read in C
    // order. path_ holds the item's position ("[0][2]" in an error), built
    // as text only for an error.
    void read_nested(nb::handle v, std::size_t depth, std::size_t event) {
        PyObject* p = v.ptr();
        const bool text = PyUnicode_Check(p) || PyBytes_Check(p) || PyByteArray_Check(p);
        if (!text && (PyList_Check(p) || PyTuple_Check(p))) {
            if (depth >= kMaxEventDepth)
                throw nb::value_error((std::string(owner_) + ": " +
                    where(event, "the value of event") + " is nested more than " +
                    std::to_string(kMaxEventDepth) + " levels deep").c_str());
            const std::size_t n = static_cast<std::size_t>(PySequence_Size(p));
            fix_dim(depth, n, event);
            path_.push_back(0);
            for (std::size_t j = 0; j < n; ++j) {
                nb::object item = nb::steal(PySequence_GetItem(p, static_cast<Py_ssize_t>(j)));
                if (!item.is_valid()) throw nb::python_error();
                path_.back() = j;
                read_nested(item, depth + 1, event);
            }
            path_.pop_back();
            return;
        }
        if (!text && PyObject_CheckBuffer(p) && read_strided(p, event, v, depth)) return;
        // A leaf: a number, or None (NaN), at the full depth. The first leaf
        // completes the shape.
        if (depth != event_shape_.size()) ragged(event);
        shape_fixed_ = true;
        double d;
        if (nb::try_cast<double>(v, d)) { wide_values_.push_back(d); return; }
        if (v.is_none()) { wide_values_.push_back(kNaN); return; }
        std::string at;
        for (std::size_t j : path_) at += "[" + std::to_string(j) + "]";
        const std::string w = "item " + at + " of " + where(event, "the value of event");
        throw_not_a_number(owner_, v, w);
    }

    // Records dimension `depth` of the event shape (the first time), or
    // checks it against the recorded one.
    void fix_dim(std::size_t depth, std::size_t n, std::size_t event) {
        if (depth < event_shape_.size()) {
            if (event_shape_[depth] != n) ragged(event);
        } else if (depth == event_shape_.size() && !shape_fixed_) {
            event_shape_.push_back(n);
        } else {
            ragged(event);
        }
    }

    [[noreturn]] void ragged(std::size_t event) const {
        throw nb::value_error((std::string(owner_) + ": " +
            where(event, "the value of event") + " has an irregular shape").c_str());
    }

    static bool known_code(char c, Py_ssize_t size) {
        switch (c) {
        case '?': return size == 1;
        case 'b': case 'B': case 'h': case 'H': case 'i': case 'I':
        case 'l': case 'L': case 'q': case 'Q': case 'n': case 'N':
            return size == 1 || size == 2 || size == 4 || size == 8;
        case 'e': return size == 2;
        case 'f': return size == 4;
        case 'd': return size == 8;
        default: return false;
        }
    }

    // One item of a buffer as a double; `size` bytes, byte-swapped if `swap`.
    static double read_item(const char* src, char code, Py_ssize_t size, bool swap) {
        unsigned char b[8];
        std::memcpy(b, src, static_cast<std::size_t>(size));
        if (swap) std::reverse(b, b + size);
        auto get = [&](auto t) { decltype(t) v; std::memcpy(&v, b, sizeof v); return v; };
        const bool is_unsigned = code == 'B' || code == 'H' || code == 'I' || code == 'L' ||
                                 code == 'Q' || code == 'N' || code == '?';
        switch (code) {
        case 'e': return half_to_double(get(std::uint16_t{}));
        case 'f': return static_cast<double>(get(float{}));
        case 'd': return get(double{});
        default: break;
        }
        if (is_unsigned) {
            switch (size) {
            case 1: return static_cast<double>(get(std::uint8_t{}));
            case 2: return static_cast<double>(get(std::uint16_t{}));
            case 4: return static_cast<double>(get(std::uint32_t{}));
            default: return static_cast<double>(get(std::uint64_t{}));
            }
        }
        switch (size) {
        case 1: return static_cast<double>(get(std::int8_t{}));
        case 2: return static_cast<double>(get(std::int16_t{}));
        case 4: return static_cast<double>(get(std::int32_t{}));
        default: return static_cast<double>(get(std::int64_t{}));
        }
    }

    static double half_to_double(std::uint16_t h) {
        const int sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, mant = h & 0x3ff;
        double v;
        if (exp == 0) v = std::ldexp(static_cast<double>(mant), -24);
        else if (exp == 31) v = mant ? std::numeric_limits<double>::quiet_NaN()
                                     : std::numeric_limits<double>::infinity();
        else v = std::ldexp(static_cast<double>(mant | 0x400), exp - 25);
        return sign ? -v : v;
    }

    static bool little_endian() {
        const std::uint16_t one = 1;
        unsigned char b;
        std::memcpy(&b, &one, 1);
        return b == 1;
    }

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
    Index last_index_{};                // the last indexed event's index
    std::size_t input_;                 // this source's input position
    const char* owner_;                 // "Pipeline", "stream"
    std::vector<double> wide_values_;   // populated when last event was wide
    std::vector<std::size_t> event_shape_;  // the last wide event's shape
    bool flatten_ = false;              // read a rank-2+ float64 array event
    bool rows_ = false;                 // a positional event may be a row
    bool shape_fixed_ = false;          // read_any_rank: the event shape is complete
    std::vector<std::size_t> path_;     // read_nested: the position of the item read
    screamer::TimeUnit* unit_;          // the call's index kind, shared; may be null
    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
};

}} // namespace screamer::streams
#endif // SCREAMER_STREAMS_PY_SOURCE_H
