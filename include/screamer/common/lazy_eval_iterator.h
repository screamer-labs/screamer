#ifndef SCREAMER_LAZY_EVAL_ITERATOR_H
#define SCREAMER_LAZY_EVAL_ITERATOR_H

#include <vector>
#include <nanobind/nanobind.h>
#include "screamer/common/eval_op.h"

namespace nb = nanobind;

namespace screamer {

// One lazy iterator for any EvalOp. Holds the functor's Python wrapper (keep-alive)
// and the input source(s). Per __next__ it collects n_in() input values, calls
// eval() once, and yields a scalar (n_out()==1) or a tuple of n_out() floats.
//
// Two input shapes are supported, matching the C++ dispatch:
//   - `sources` holds n_in() separate iterators (one value pulled from each);
//   - `sources` holds exactly one iterator whose items are n_in()-tuples (unpacked),
//     used for the "one iterable of tuples" call form.
class LazyEvalIterator {
public:
    LazyEvalIterator(nb::object op_owner, std::vector<nb::object> iterables)
        : op_owner_(std::move(op_owner)),
          op_(nb::cast<EvalOp&>(op_owner_)),
          n_in_(op_.n_in()), n_out_(op_.n_out()),
          in_(op_.n_in()), out_(op_.n_out()) {
        for (auto& it : iterables) iters_.push_back(it.attr("__iter__")());
        unpack_tuples_ = (iters_.size() == 1 && n_in_ > 1);
        in_.resize(n_in_);
        out_.resize(n_out_);
    }

    LazyEvalIterator& __iter__() { return *this; }

    nb::object __next__() {
        if (unpack_tuples_) {
            nb::object item = next_or_stop(iters_[0]);          // an n_in-tuple
            nb::sequence seq = nb::cast<nb::sequence>(item);
            if (nb::len(seq) != static_cast<Py_ssize_t>(n_in_))
                throw nb::value_error("LazyEvalIterator: tuple size does not match n_in");
            for (std::size_t i = 0; i < n_in_; ++i)
                in_[i] = nb::cast<double>(seq[i]);
        } else {
            for (std::size_t i = 0; i < n_in_; ++i)
                in_[i] = nb::cast<double>(next_or_stop(iters_[i]));
        }
        op_.eval(in_.data(), out_.data());
        if (n_out_ == 1) return nb::float_(out_[0]);
        nb::list lst;
        for (std::size_t i = 0; i < n_out_; ++i) lst.append(nb::float_(out_[i]));
        return nb::steal(PySequence_Tuple(lst.ptr()));
    }

private:
    static nb::object next_or_stop(nb::object& it) {
        try {
            return it.attr("__next__")();
        } catch (nb::python_error& e) {
            if (e.matches(PyExc_StopIteration)) throw nb::stop_iteration();
            throw;
        }
    }

    nb::object op_owner_;                 // keeps the functor wrapper alive
    EvalOp& op_;
    std::size_t n_in_, n_out_;
    std::vector<nb::object> iters_;
    std::vector<double> in_, out_;
    bool unpack_tuples_ = false;
};

}  // namespace screamer
#endif
