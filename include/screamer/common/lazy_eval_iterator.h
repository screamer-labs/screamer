#ifndef SCREAMER_LAZY_EVAL_ITERATOR_H
#define SCREAMER_LAZY_EVAL_ITERATOR_H

#include <vector>
#include <pybind11/pybind11.h>
#include "screamer/common/eval_op.h"

namespace py = pybind11;

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
    LazyEvalIterator(py::object op_owner, std::vector<py::object> iterables)
        : op_owner_(std::move(op_owner)),
          op_(op_owner_.cast<EvalOp&>()),
          n_in_(op_.n_in()), n_out_(op_.n_out()),
          in_(op_.n_in()), out_(op_.n_out()) {
        for (auto& it : iterables) iters_.push_back(py::iter(it));
        unpack_tuples_ = (iters_.size() == 1 && n_in_ > 1);
        in_.resize(n_in_);
        out_.resize(n_out_);
    }

    LazyEvalIterator& __iter__() { return *this; }

    py::object __next__() {
        if (unpack_tuples_) {
            py::object item = next_or_stop(iters_[0]);          // an n_in-tuple
            py::sequence seq = py::cast<py::sequence>(item);
            if (py::len(seq) != static_cast<py::ssize_t>(n_in_))
                throw py::value_error("LazyEvalIterator: tuple size does not match n_in");
            for (std::size_t i = 0; i < n_in_; ++i)
                in_[i] = seq[i].cast<double>();
        } else {
            for (std::size_t i = 0; i < n_in_; ++i)
                in_[i] = next_or_stop(iters_[i]).cast<double>();
        }
        op_.eval(in_.data(), out_.data());
        if (n_out_ == 1) return py::float_(out_[0]);
        py::tuple t(n_out_);
        for (std::size_t i = 0; i < n_out_; ++i) t[i] = py::float_(out_[i]);
        return std::move(t);
    }

private:
    static py::object next_or_stop(py::iterator& it) {
        if (it == py::iterator()) throw py::stop_iteration();   // default-constructed sentinel
        py::object v = py::reinterpret_borrow<py::object>(*it);
        ++it;
        return v;
    }

    py::object op_owner_;                 // keeps the functor wrapper alive
    EvalOp& op_;
    std::size_t n_in_, n_out_;
    std::vector<py::iterator> iters_;
    std::vector<double> in_, out_;
    bool unpack_tuples_ = false;
};

}  // namespace screamer
#endif
