#ifndef SCREAMER_LAZY_EVAL_ITERATOR_H
#define SCREAMER_LAZY_EVAL_ITERATOR_H

#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "screamer/common/eval_op.h"
#include "screamer/common/shape.h"

namespace nb = nanobind;

namespace screamer {

// One lazy iterator for any EvalOp. Holds the functor's Python wrapper (keep-alive)
// and the input source(s). Per __next__ it reads one event from each source.
//
// Two input forms, matching the batch dispatch:
//   - `sources` holds n_in() separate iterators;
//   - `sources` holds one iterator whose items hold all n_in() fields: a tuple or
//     list (one value per field) or an array whose last axis has n_in() fields.
//
// Events are scalars (one float out, or a tuple of n_out() floats), or arrays of
// one fixed shape: each holds one value per asset (shape.h rules 2 and 3), the
// first event fixes the shape, and every asset runs its own copy of the op, as
// the batch path gives every column its own state.
class LazyEvalIterator {
public:
    LazyEvalIterator(nb::object op_owner, std::vector<nb::object> iterables)
        : op_owner_(std::move(op_owner)),
          op_(nb::cast<EvalOp&>(op_owner_)),
          n_in_(op_.n_in()), n_out_(op_.n_out()),
          in_(op_.n_in()), out_(op_.n_out()) {
        for (auto& it : iterables) iters_.push_back(it.attr("__iter__")());
        unpack_ = (iters_.size() == 1 && n_in_ > 1);
    }

    LazyEvalIterator& __iter__() { return *this; }

    nb::object __next__() {
        items_.clear();
        for (auto& it : iters_) items_.push_back(next_or_stop(it));
        if (mode_ == Mode::Unknown) {
            if (scalar_values()) {
                mode_ = Mode::Scalar;
                return eval_scalar();
            }
            mode_ = Mode::Assets;
        }
        if (mode_ == Mode::Scalar) {
            if (!scalar_values())
                throw nb::value_error("every event must have the same shape; the first events "
                                      "were numbers, this one is not");
            return eval_scalar();
        }
        return next_assets();
    }

private:
    enum class Mode { Unknown, Scalar, Assets };
    using Arr = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;

    static nb::object next_or_stop(nb::object& it) {
        try {
            return it.attr("__next__")();
        } catch (nb::python_error& e) {
            if (e.matches(PyExc_StopIteration)) throw nb::stop_iteration();
            throw;
        }
    }

    // A plain number. An array with an axis is not one, even with one element:
    // an event of shape (1,) holds one asset.
    static bool as_number(nb::handle h, double& v) {
        if (PyFloat_Check(h.ptr())) { v = PyFloat_AsDouble(h.ptr()); return true; }
        if (!PyLong_Check(h.ptr()) && nb::hasattr(h, "ndim") && nb::cast<int>(h.attr("ndim")) > 0)
            return false;
        return nb::try_cast<double>(h, v);
    }

    // Reads the n_in field values of the current event into in_ if every one is
    // a plain number.
    bool scalar_values() {
        if (unpack_) {
            nb::sequence seq;
            if (!nb::try_cast<nb::sequence>(items_[0], seq)) return false;
            if (nb::len(seq) != static_cast<Py_ssize_t>(n_in_)) return false;
            for (std::size_t i = 0; i < n_in_; ++i)
                if (!as_number(seq[i], in_[i])) return false;
            return true;
        }
        for (std::size_t i = 0; i < n_in_; ++i)
            if (!as_number(items_[i], in_[i])) return false;
        return true;
    }

    nb::object eval_scalar() {
        op_.eval(in_.data(), out_.data());
        if (n_out_ == 1) return nb::float_(out_[0]);
        nb::tuple t = nb::steal<nb::tuple>(PyTuple_New(static_cast<Py_ssize_t>(n_out_)));
        for (std::size_t i = 0; i < n_out_; ++i)
            PyTuple_SetItem(t.ptr(), static_cast<Py_ssize_t>(i), nb::float_(out_[i]).release().ptr());
        return t;
    }

    nb::object next_assets() {
        parts_.clear();
        if (unpack_ && (nb::isinstance<nb::tuple>(items_[0]) || nb::isinstance<nb::list>(items_[0]))) {
            for (nb::handle h : items_[0]) parts_.push_back(nb::borrow<nb::object>(h));   // one per field
        } else {
            parts_ = items_;
        }
        if (!ascontiguous_.is_valid()) {
            nb::module_ np = nb::module_::import_("numpy");
            ascontiguous_ = np.attr("ascontiguousarray");
            float64_ = np.attr("float64");
        }
        arrays_.clear();
        current_.resize(parts_.size());
        for (std::size_t k = 0; k < parts_.size(); ++k) {
            parts_[k] = ascontiguous_(parts_[k], float64_);   // no copy if already contiguous float64
            arrays_.push_back(nb::cast<Arr>(parts_[k]));
            const Arr& a = arrays_.back();
            shape::Shape& s = current_[k];
            s.resize(a.ndim() + 1);
            s[0] = 1;                                         // one time step
            for (std::size_t d = 0; d < a.ndim(); ++d) s[d + 1] = a.shape(d);
        }
        if (copies_.empty()) start_assets();
        else if (current_ != shapes_)
            throw nb::value_error(("every event must have the same shape; the first was " +
                event_shapes(shapes_) + ", this one is " + event_shapes(current_)).c_str());

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

    // The first asset event fixes the shape; build one op per asset. Copies come
    // from the wrapper's _fresh(); the original op is reset so every asset
    // starts clean.
    void start_assets() {
        plan_ = shape::broadcast(shape::input_views(current_, n_in_));
        if (plan_.columns > 1 && !nb::hasattr(op_owner_, "_fresh"))
            throw nb::type_error("events with several assets need an operator built from the "
                                 "screamer namespace, which can copy itself once per asset");
        shapes_ = current_;
        out_shape_.assign(plan_.shape.begin() + 1, plan_.shape.end());
        if (n_out_ > 1) out_shape_.push_back(n_out_);
        op_.reset();
        copies_.reserve(plan_.columns);
        owners_.reserve(plan_.columns ? plan_.columns - 1 : 0);
        copies_.push_back(&op_);
        for (std::size_t c = 1; c < plan_.columns; ++c) {
            owners_.push_back(op_owner_.attr("_fresh")());
            copies_.push_back(&nb::cast<EvalOp&>(owners_.back()));
        }
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
    std::size_t n_in_, n_out_;
    std::vector<nb::object> iters_;
    std::vector<double> in_, out_;
    bool unpack_ = false;
    Mode mode_ = Mode::Unknown;
    // Per-event buffers, kept to reuse their storage.
    std::vector<nb::object> items_, parts_;
    std::vector<Arr> arrays_;
    std::vector<shape::Shape> current_;
    nb::object ascontiguous_, float64_;   // numpy.ascontiguousarray, numpy.float64
    // Fixed by the first asset event.
    std::vector<shape::Shape> shapes_;    // per argument, with a leading 1 for T
    shape::Plan plan_;
    std::vector<std::size_t> out_shape_;
    std::vector<nb::object> owners_;      // the _fresh() copies, kept alive
    std::vector<EvalOp*> copies_;         // one op per asset; copies_[0] is op_
};

}  // namespace screamer
#endif
