#ifndef SCREAMER_LAZY_EVAL_ITERATOR_H
#define SCREAMER_LAZY_EVAL_ITERATOR_H

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "screamer/common/cast_double.h"
#include "screamer/common/eval_op.h"
#include "screamer/common/gc.h"
#include "screamer/common/py_iter.h"
#include "screamer/common/shape.h"

namespace nb = nanobind;

namespace screamer {

// The per-event work shared by the synchronous and the async iterators. The
// caller reads one event (one item per source) and passes the items to step(),
// which runs the operator on them and returns the output. LazyEvalIterator pulls
// the items from Python iterators; LazyAsyncIterator awaits them from async
// iterators. Both call the same step(), so both give the batch result.
class EventStep {
public:
    virtual ~EventStep() = default;
    // Runs one event. `items` holds one item per source.
    virtual nb::object step(const EventItems& items) = 0;
    // Reports every Python object the stepper holds to the garbage collector
    // (see gc.h). The iterator that owns the stepper calls it.
    virtual int traverse(visitproc visit, void* arg) const = 0;
};

// The EventStep of every EvalOp with a fixed number of fields.
//
// Two input forms, matching the batch dispatch:
//   - n_sources == n_in(): one item per field;
//   - n_sources == 1 and n_in() > 1: the one item holds all n_in() fields, as a
//     tuple or list (one value per field) or an array whose last axis has n_in()
//     fields.
//
// Events are scalars (one float out, or a tuple of n_out() floats), or arrays of
// one fixed shape: each holds one value per asset (rules 1 and 4), the
// first event fixes the shape, and every asset runs its own copy of the op, as
// the batch path gives every column its own state.
//
// A stream is an independent run (decision D8). Opening it resets the
// operator's running series, like a batch call, and every event runs on copies
// the stream owns, made with the wrapper's _fresh(), so two streams of one
// operator never share state and live calls op(x) never touch a stream. An eager
// call (a list of events, consumed before it returns) runs column 0 on the
// operator itself, which the caller resets afterwards: nothing can interleave
// with it, so it saves that copy.
class EventStepper final : public EventStep {
public:
    EventStepper(nb::object op_owner, std::size_t n_sources, bool stream = true)
        : op_owner_(std::move(op_owner)),
          op_(nb::cast<EvalOp&>(op_owner_)),
          n_in_(op_.n_in()), n_out_(op_.n_out()),
          in_(op_.n_in()), out_(op_.n_out()),
          unpack_(n_sources == 1 && n_in_ > 1) {
        if (stream) {
            first_owner_ = fresh_copy(op_owner_);
            first_ = &nb::cast<EvalOp&>(first_owner_);
        } else {
            first_ = &op_;
        }
        op_.reset();
    }

    // A new copy of the operator in its initial state. A stream needs one, so an
    // operator built outside the screamer namespace (no _fresh) cannot stream.
    static nb::object fresh_copy(const nb::object& owner) {
        if (!nb::hasattr(owner, "_fresh"))
            throw nb::type_error("a stream needs an operator built from the screamer namespace, "
                                 "which can copy itself for every stream and asset");
        return owner.attr("_fresh")();
    }

    nb::object step(const EventItems& items) override {
        ++event_;
        if (mode_ == Mode::Unknown) {
            if (scalar_values(items)) {
                mode_ = Mode::Scalar;
                return eval_scalar();
            }
            mode_ = Mode::Assets;
        }
        if (mode_ == Mode::Scalar) {
            if (!scalar_values(items)) throw_scalar_mismatch(items);
            return eval_scalar();
        }
        return step_assets(items);
    }

    int traverse(visitproc visit, void* arg) const override {
        if (int r = gc_visit(op_owner_, visit, arg)) return r;
        if (int r = gc_visit(first_owner_, visit, arg)) return r;
        if (int r = gc_visit(ascontiguous_, visit, arg)) return r;
        if (int r = gc_visit(float64_, visit, arg)) return r;
        if (int r = gc_visit_all(owners_, visit, arg)) return r;
        return gc_visit_all(parts_, visit, arg);
    }

private:
    enum class Mode { Unknown, Scalar, Assets };
    using Arr = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;

    // A plain number. An array with an axis is not one, even with one element:
    // an event of shape (1,) holds one asset.
    static bool as_number(nb::handle h, double& v) {
        if (PyFloat_Check(h.ptr())) { v = PyFloat_AsDouble(h.ptr()); return true; }
        return as_number_slow(h, v);
    }
    // Everything but a float, kept out of line so the float path stays small.
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((noinline))
#endif
    static bool as_number_slow(nb::handle h, double& v) {
        int nd = 0;
        if (!PyLong_Check(h.ptr()) && nb::hasattr(h, "ndim") &&
            nb::try_cast<int>(h.attr("ndim"), nd) && nd > 0)
            return false;
        return read_double(h, v);                     // None is NaN
    }

    // Reads the n_in field values of the event into in_ if every one is a
    // plain number.
    bool scalar_values(const EventItems& items) {
        if (unpack_) {
            nb::sequence seq;
            if (!nb::try_cast<nb::sequence>(items[0], seq)) return false;
            if (nb::len(seq) != static_cast<Py_ssize_t>(n_in_)) return false;
            for (std::size_t i = 0; i < n_in_; ++i)
                if (!as_number(seq[i], in_[i])) return false;
            return true;
        }
        for (std::size_t i = 0; i < n_in_; ++i)
            if (!as_number(items[i], in_[i])) return false;
        return true;
    }

    nb::object eval_scalar() {
        first_->eval(in_.data(), out_.data());
        if (n_out_ == 1) return nb::float_(out_[0]);
        nb::tuple t = nb::steal<nb::tuple>(PyTuple_New(static_cast<Py_ssize_t>(n_out_)));
        for (std::size_t i = 0; i < n_out_; ++i)
            PyTuple_SetItem(t.ptr(), static_cast<Py_ssize_t>(i), nb::float_(out_[i]).release().ptr());
        return t;
    }

    nb::object step_assets(const EventItems& items) {
        parts_.clear();
        if (unpack_ && (nb::isinstance<nb::tuple>(items[0]) || nb::isinstance<nb::list>(items[0]))) {
            for (nb::handle h : items[0]) parts_.push_back(nb::borrow<nb::object>(h));   // one per field
        } else {
            for (std::size_t i = 0; i < items.size(); ++i) parts_.push_back(nb::borrow<nb::object>(items[i]));
        }
        if (!ascontiguous_.is_valid()) {
            nb::module_ np = nb::module_::import_("numpy");
            ascontiguous_ = np.attr("ascontiguousarray");
            float64_ = np.attr("float64");
        }
        arrays_.clear();
        current_.resize(parts_.size());
        for (std::size_t k = 0; k < parts_.size(); ++k) {
            Arr arr;
            // A time or a ctypes object is a wrong kind, never read as numbers
            // (nor cast: a ctypes buffer has no strides).
            if (is_foreign_value(parts_[k].ptr())) {
                throw_if_wrong_kind_event(items);
                refuse_text(parts_[k], op_display_name(op_owner_));
            }
            // A string, or a list or array of them, is a wrong kind; the
            // float64 cast below would read "3" as 3.
            if (!nb::try_cast<Arr>(parts_[k], arr, /*convert=*/false))
                refuse_text(parts_[k], op_display_name(op_owner_));
            try {
                parts_[k] = ascontiguous_(parts_[k], float64_);   // no copy if already contiguous float64
            } catch (nb::python_error& e) {
                if (is_conversion_error(e)) throw_if_wrong_kind_event(items);
                throw;
            }
            if (!nb::try_cast<Arr>(parts_[k], arr)) {
                throw_if_wrong_kind_event(items);
                throw nb::type_error((op_display_name(op_owner_) + ": " +
                    where_text("event", static_cast<std::ptrdiff_t>(event_ - 1)) +
                    " is not numeric").c_str());
            }
            arrays_.push_back(std::move(arr));
            const Arr& a = arrays_.back();
            shape::Shape& s = current_[k];
            s.resize(a.ndim() + 1);
            s[0] = 1;                                         // one time step
            for (std::size_t d = 0; d < a.ndim(); ++d) s[d + 1] = a.shape(d);
        }
        if (copies_.empty()) start_assets();
        else if (current_ != shapes_)
            throw nb::value_error((op_display_name(op_owner_) + ": every event must have the same "
                "shape; the first was " + event_shapes(shapes_) + ", " +
                where_text("event", static_cast<std::ptrdiff_t>(event_ - 1)) + " is " +
                event_shapes(current_)).c_str());

        const std::size_t cols = plan_.columns;
        const std::size_t total = cols * n_out_;
        double* out = new double[total ? total : 1];
        nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        const bool packed = arrays_.size() == 1;
        for (std::size_t c = 0; c < cols; ++c) {
            for (std::size_t i = 0; i < n_in_; ++i)
                in_[i] = arrays_[packed ? 0 : i].data()[plan_.offset[i * cols + c]];
            copies_[c]->eval(in_.data(), out + c * n_out_);
        }
        return nb::cast(nb::ndarray<nb::numpy, double>(out, out_shape_.size(), out_shape_.data(), owner));
    }

    // The first asset event fixes the shape; build one op per asset. Column 0
    // runs on first_, which no event has touched yet, and the others on new
    // copies from the wrapper's _fresh().
    // Everything is built in locals and committed only when every copy exists,
    // so a failed _fresh() leaves the stepper as it was.
    void start_assets() {
        shape::Plan plan = shape::broadcast(shape::input_views(current_, n_in_));
        std::vector<nb::object> owners;
        std::vector<EvalOp*> copies;
        owners.reserve(plan.columns ? plan.columns - 1 : 0);
        copies.reserve(plan.columns);
        copies.push_back(first_);
        for (std::size_t c = 1; c < plan.columns; ++c) {
            owners.push_back(fresh_copy(op_owner_));
            copies.push_back(&nb::cast<EvalOp&>(owners.back()));
        }
        out_shape_.assign(plan.shape.begin() + 1, plan.shape.end());
        if (n_out_ > 1) out_shape_.push_back(n_out_);
        plan_ = std::move(plan);
        shapes_ = current_;
        owners_ = std::move(owners);
        copies_ = std::move(copies);
    }

    // A failed event: names the first element that is not a number, list,
    // tuple or array, if any (error path only).
    void throw_if_wrong_kind_event(const EventItems& items) const {
        screamer::throw_if_wrong_kind_event(op_display_name(op_owner_), items, event_ - 1);
    }

    // A later event of a number stream is not numbers: TypeError for a wrong
    // kind, else ValueError giving the shape of the first field that is not a
    // number.
    [[noreturn]] void throw_scalar_mismatch(const EventItems& items) const {
        throw_if_wrong_kind_event(items);
        const std::string who = op_display_name(op_owner_);
        const auto event = static_cast<std::ptrdiff_t>(event_ - 1);
        double v;
        if (unpack_) {
            const std::string where = where_text("event", event);
            nb::sequence seq;
            if (!nb::try_cast<nb::sequence>(items[0], seq) ||
                nb::len(seq) != static_cast<Py_ssize_t>(n_in_)) {
                const std::string expected = std::to_string(n_in_) + " numbers";
                throw_bad_shape(who, where, items[0], expected.c_str());
            }
            for (std::size_t j = 0; j < n_in_; ++j)
                if (!as_number(seq[j], v))
                    throw_bad_shape(who, "item [" + std::to_string(j) + "] of " + where, seq[j]);
        }
        for (std::size_t i = 0; i < items.size(); ++i)
            if (!as_number(items[i], v))
                throw_bad_shape(who, where_text("event", event,
                    items.size() > 1 ? static_cast<std::ptrdiff_t>(i) : -1), items[i]);
        throw nb::value_error((who + ": every event must have the same shape; the first "
                               "events were numbers, " + where_text("event", event) +
                               " is not").c_str());
    }

    static std::string event_shapes(const std::vector<shape::Shape>& shapes) {
        std::string s;
        for (std::size_t i = 0; i < shapes.size(); ++i) {
            shape::Shape e(shapes[i].begin() + 1, shapes[i].end());
            s += (i ? " and " : "") + shape::to_string(e);
        }
        return s;
    }

    nb::object op_owner_;                 // keeps the functor wrapper alive
    EvalOp& op_;
    nb::object first_owner_;              // a stream's own copy for column 0
    EvalOp* first_;                       // runs scalar events and column 0
    std::size_t n_in_, n_out_;
    std::vector<double> in_, out_;
    bool unpack_;
    Mode mode_ = Mode::Unknown;
    std::size_t event_ = 0;               // events seen, for error messages
    // Per-event buffers, kept to reuse their storage.
    std::vector<nb::object> parts_;
    std::vector<Arr> arrays_;
    std::vector<shape::Shape> current_;
    nb::object ascontiguous_, float64_;   // numpy.ascontiguousarray, numpy.float64
    // Fixed by the first asset event.
    std::vector<shape::Shape> shapes_;    // per argument, with a leading 1 for T
    shape::Plan plan_;
    std::vector<std::size_t> out_shape_;
    std::vector<nb::object> owners_;      // the _fresh() copies, kept alive
    std::vector<EvalOp*> copies_;         // one op per asset; copies_[0] is first_
};

// The lazy iterator of every operator. Per __next__ it reads one item from
// each source and passes them to an EventStep; nothing is read before the first
// __next__. The EvalOp form takes n_in() separate iterators, or one iterator
// whose events hold all n_in() fields (see EventStepper).
class LazyEvalIterator {
public:
    LazyEvalIterator(std::unique_ptr<EventStep> stepper, const std::vector<nb::object>& iterables)
        : stepper_(std::move(stepper)) {
        for (auto& it : iterables) iters_.push_back(it.attr("__iter__")());
    }
    LazyEvalIterator(nb::object op_owner, const std::vector<nb::object>& iterables)
        : LazyEvalIterator(std::make_unique<EventStepper>(std::move(op_owner), iterables.size()),
                           iterables) {}

    LazyEvalIterator& __iter__() { return *this; }

    nb::object __next__() {
        if (!stepper_) throw nb::stop_iteration();   // cleared by the garbage collector
        items_.clear();
        for (auto& it : iters_) items_.push_back(next_or_stop(it));
        return stepper_->step(EventItems(items_));
    }

    // Garbage-collector support (gc.h): the sources, the last event and the
    // stepper's operators.
    int traverse(visitproc visit, void* arg) const {
        if (int r = gc_visit_all(iters_, visit, arg)) return r;
        if (int r = gc_visit_all(items_, visit, arg)) return r;
        return stepper_ ? stepper_->traverse(visit, arg) : 0;
    }

    // Drops every held object; the iterator then ends. The members are emptied
    // before the objects are released, so code a release runs sees an ended
    // iterator.
    void clear() {
        auto stepper = std::move(stepper_);
        auto iters = std::move(iters_);
        auto items = std::move(items_);
        stepper_.reset();
        iters_.clear();
        items_.clear();
    }

private:
    static nb::object next_or_stop(nb::object& it) {
        nb::object item = next_item(it);
        if (!item.is_valid()) throw nb::stop_iteration();
        return item;
    }

    std::unique_ptr<EventStep> stepper_;
    std::vector<nb::object> iters_;
    std::vector<nb::object> items_;       // per-event buffer, kept to reuse its storage
};

}  // namespace screamer
#endif
