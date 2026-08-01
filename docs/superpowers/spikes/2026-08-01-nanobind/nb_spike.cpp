// nanobind feasibility spike for screamer's pybind11 -> nanobind migration.
// Exercises the 7 risky mechanisms. Throwaway proof-of-concept.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "screamer/common/eval_op.h"       // pybind-free abstract EvalOp
#include "screamer/detail/rolling_mean.h"  // pybind-free numeric kernel

namespace nb = nanobind;
using namespace nb::literals;

// ---------------------------------------------------------------------------
// Pure-compute ops implementing EvalOp (no Python in these).
// ---------------------------------------------------------------------------

// 1->1: wraps the real detail::RollingMean kernel.
struct MeanEvalOp : screamer::EvalOp {
    screamer::detail::RollingMean impl;
    explicit MeanEvalOp(size_t n, const std::string& policy) : impl(n, policy) {}
    std::size_t n_in() const override { return 1; }
    std::size_t n_out() const override { return 1; }
    void eval(const double* in, double* out) override { out[0] = impl.append(in[0]); }
    void reset() override { impl.reset(); }
};

// 1->2: mirrors a FunctorBase<_,1,2>. Emits (rolling_mean, running_sum).
struct MeanSumEvalOp : screamer::EvalOp {
    screamer::detail::RollingMean impl;
    double running_sum = 0.0;
    explicit MeanSumEvalOp(size_t n) : impl(n, "strict") {}
    std::size_t n_in() const override { return 1; }
    std::size_t n_out() const override { return 2; }
    void eval(const double* in, double* out) override {
        out[0] = impl.append(in[0]);
        running_sum += in[0];
        out[1] = running_sum;
    }
    void reset() override { impl.reset(); running_sum = 0.0; }
};

// ---------------------------------------------------------------------------
// R1 crux: read any-dtype/strided ndarray element -> double.
// ---------------------------------------------------------------------------
static double load_elem(const void* base, int64_t idx, const nb::dlpack::dtype& dt) {
    using C = nb::dlpack::dtype_code;
    if (dt.code == (uint8_t) C::Float) {
        if (dt.bits == 64) return ((const double*)  base)[idx];
        if (dt.bits == 32) return ((const float*)   base)[idx];
    } else if (dt.code == (uint8_t) C::Int) {
        if (dt.bits == 64) return (double)((const int64_t*) base)[idx];
        if (dt.bits == 32) return (double)((const int32_t*) base)[idx];
        if (dt.bits == 16) return (double)((const int16_t*) base)[idx];
    } else if (dt.code == (uint8_t) C::UInt) {
        if (dt.bits == 64) return (double)((const uint64_t*) base)[idx];
        if (dt.bits == 32) return (double)((const uint32_t*) base)[idx];
    }
    throw nb::type_error("unsupported ndarray dtype in spike");
}

// R1 crux: allocate a NEW numpy array and hand nanobind an owner capsule.
template <typename Shape>
static nb::object make_owned_array(double* data, const Shape& shape) {
    nb::capsule owner(data, [](void* p) noexcept { delete[] (double*) p; });
    return nb::cast(nb::ndarray<nb::numpy, double>(
        data, shape.size(), shape.data(), owner));
}

// ---------------------------------------------------------------------------
// Item 4/5: hand-written lazy iterator (mirrors LazyEvalIterator).
// ---------------------------------------------------------------------------
struct LazyMeanIter {
    nb::object source;      // the upstream Python iterator
    nb::object keepalive;   // the parent op's own Python wrapper (from nb::find)
    std::unique_ptr<MeanEvalOp> op;

    LazyMeanIter(nb::object src, nb::object ka, size_t n, std::string policy)
        : source(std::move(src)), keepalive(std::move(ka)),
          op(std::make_unique<MeanEvalOp>(n, std::move(policy))) {}

    LazyMeanIter* iter() { return this; }

    nb::object next() {
        nb::object item;
        try {
            item = source.attr("__next__")();
        } catch (nb::python_error& e) {
            if (e.matches(PyExc_StopIteration)) {
                throw;  // let nanobind re-raise StopIteration -> ends the loop
            }
            throw;
        }
        double x = nb::cast<double>(item), out;
        op->eval(&x, &out);
        return nb::cast(out);
    }
};

// ---------------------------------------------------------------------------
// Item 1: dispatching Mean op.
// ---------------------------------------------------------------------------
struct Mean {
    size_t n;
    std::string policy;
    std::unique_ptr<MeanEvalOp> streaming;  // stateful scalar stream
    int reset_count = 0;

    Mean(size_t n_, std::string policy_ = "strict")
        : n(n_), policy(std::move(policy_)),
          streaming(std::make_unique<MeanEvalOp>(n_, policy)) {}

    nb::object call(nb::object arg) {
        // (a) scalar float / int -> stateful stream, returns float
        if (nb::isinstance<nb::float_>(arg) || nb::isinstance<nb::int_>(arg)) {
            double x = nb::cast<double>(arg), out;
            streaming->eval(&x, &out);
            return nb::cast(out);
        }
        // (d) generic iterator (has __next__ but is not a list) -> lazy iterator
        if (!nb::isinstance<nb::list>(arg) && nb::hasattr(arg, "__next__")) {
            nb::object self_wrapper = nb::find(*this);  // item 5
            return nb::cast(new LazyMeanIter(arg, self_wrapper, n, policy),
                            nb::rv_policy::take_ownership);
        }
        // (b) list -> list (batch: reset first)
        if (nb::isinstance<nb::list>(arg)) {
            MeanEvalOp batch(n, policy);
            batch.reset();
            reset_count++;
            nb::list in = nb::cast<nb::list>(arg);
            nb::list result;
            for (nb::handle h : in) {
                double x = nb::cast<double>(h), out;
                batch.eval(&x, &out);
                result.append(out);
            }
            return result;
        }
        // (c)(d)(e) ndarray -> NEW ndarray (batch along last axis)
        nb::ndarray<> a = nb::cast<nb::ndarray<>>(arg);
        MeanEvalOp batch(n, policy);
        size_t ndim = a.ndim();
        std::vector<size_t> shape(ndim);
        size_t total = 1;
        for (size_t i = 0; i < ndim; i++) { shape[i] = a.shape(i); total *= shape[i]; }
        double* outbuf = new double[total ? total : 1];
        const void* base = a.data();
        nb::dlpack::dtype dt = a.dtype();

        size_t last = ndim ? shape[ndim - 1] : 0;
        size_t rows = last ? total / last : 0;
        // Iterate rows; reset op at each row boundary (batch-per-series).
        for (size_t r = 0; r < rows; r++) {
            batch.reset();
            reset_count++;
            // Decompose row index r into multi-index over the leading axes to
            // find the correct strided offset of element 0 of this row.
            int64_t off0 = 0;
            size_t rem = r;
            for (size_t ax = 0; ndim && ax + 1 < ndim; ax++) {
                size_t below = 1;
                for (size_t k = ax + 1; k + 1 < ndim; k++) below *= shape[k];
                size_t idx_ax = below ? (rem / below) : rem;
                rem = below ? (rem % below) : 0;
                off0 += (int64_t) idx_ax * a.stride(ax);
            }
            int64_t last_stride = ndim ? a.stride(ndim - 1) : 1;
            for (size_t j = 0; j < last; j++) {
                double x = load_elem(base, off0 + (int64_t) j * last_stride, dt), out;
                batch.eval(&x, &out);
                outbuf[r * last + j] = out;   // output is C-contiguous
            }
        }
        return make_owned_array(outbuf, shape);
    }
};

// ---------------------------------------------------------------------------
// Item 2: multi-output op returning tuple (scalar) / trailing-axis array.
// ---------------------------------------------------------------------------
struct MeanSum {
    size_t n;
    std::unique_ptr<MeanSumEvalOp> streaming;
    explicit MeanSum(size_t n_) : n(n_), streaming(std::make_unique<MeanSumEvalOp>(n_)) {}

    nb::object call(nb::object arg) {
        if (nb::isinstance<nb::float_>(arg) || nb::isinstance<nb::int_>(arg)) {
            double x = nb::cast<double>(arg), out[2];
            streaming->eval(&x, out);
            return nb::cast(std::make_tuple(out[0], out[1]));  // stl/tuple.h
        }
        // 1-D ndarray -> (len, 2) array.
        auto a = nb::cast<nb::ndarray<>>(arg);
        MeanSumEvalOp batch(n);
        size_t len = a.shape(0);
        int64_t s0 = a.stride(0);
        nb::dlpack::dtype dt = a.dtype();
        const void* base = a.data();
        double* outbuf = new double[len * 2 ? len * 2 : 1];
        for (size_t i = 0; i < len; i++) {
            double x = load_elem(base, (int64_t) i * s0, dt), out[2];
            batch.eval(&x, out);
            outbuf[i * 2] = out[0];
            outbuf[i * 2 + 1] = out[1];
        }
        std::vector<size_t> shape{len, 2};
        return make_owned_array(outbuf, shape);
    }
};

// ---------------------------------------------------------------------------
// Item 3: optional<double> mutually-exclusive ctor args + vector<double> ctor.
// ---------------------------------------------------------------------------
struct Smoother {
    double param;
    Smoother(std::optional<double> period, std::optional<double> cutoff) {
        if (period.has_value() == cutoff.has_value())
            throw std::invalid_argument("pass exactly one of period=, cutoff=");
        param = period.has_value() ? *period : (1.0 / *cutoff);
    }
    double get_param() const { return param; }
};

struct Taps {
    std::vector<double> taps;
    explicit Taps(const std::vector<double>& t) : taps(t) {}
    size_t size() const { return taps.size(); }
    double sum() const { double s = 0; for (double v : taps) s += v; return s; }
};

// ---------------------------------------------------------------------------
// Item 7: import + call a Python helper module (replaces removed py::exec).
// ---------------------------------------------------------------------------
static nb::object call_helper(double x) {
    nb::module_ helper = nb::module_::import_("spike_helper");
    return helper.attr("scale")(x);              // sync call, proves import path
}
static bool async_helper_importable() {
    nb::module_ helper = nb::module_::import_("spike_helper");
    return nb::hasattr(helper, "stream");        // grab the async gen fn object
}

// ---------------------------------------------------------------------------
NB_MODULE(nb_spike, m) {
    nb::class_<Mean>(m, "Mean")
        .def(nb::init<size_t, std::string>(), "n"_a, "policy"_a = "strict")
        .def("__call__", &Mean::call, "arg"_a.none())
        .def_prop_ro("reset_count", [](const Mean& s) { return s.reset_count; });

    nb::class_<LazyMeanIter>(m, "LazyMeanIter")
        .def("__iter__", &LazyMeanIter::iter, nb::rv_policy::reference_internal)
        .def("__next__", &LazyMeanIter::next);

    nb::class_<MeanSum>(m, "MeanSum")
        .def(nb::init<size_t>(), "n"_a)
        .def("__call__", &MeanSum::call, "arg"_a);

    nb::class_<Smoother>(m, "Smoother")
        .def(nb::init<std::optional<double>, std::optional<double>>(),
             "period"_a = nb::none(), "cutoff"_a = nb::none())
        .def_prop_ro("param", &Smoother::get_param);

    nb::class_<Taps>(m, "Taps")
        .def(nb::init<const std::vector<double>&>(), "taps"_a)
        .def_prop_ro("size", &Taps::size)
        .def_prop_ro("sum", &Taps::sum);

    m.def("call_helper", &call_helper, "x"_a);
    m.def("async_helper_importable", &async_helper_importable);
}
