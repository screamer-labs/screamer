#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>
#include "screamer/common/base.h"
#include "screamer/common/dispatch.h"
#include "screamer/common/eval_op.h"
#include "screamer/return.h"
#include "screamer/log_return.h"
#include "screamer/roc.h"
#include "screamer/rocp.h"
#include "screamer/rocr.h"
#include "screamer/rolling_corr.h"
#include "screamer/rolling_cov.h"
#include "screamer/rolling_beta.h"
#include "screamer/rolling_tls_beta.h"
#include "screamer/kalman_regression.h"
#include "screamer/rolling_adf.h"
#include "screamer/band_signal.h"
#include "screamer/rolling_spread.h"
#include "screamer/drawdown.h"
#include "screamer/max_drawdown.h"
#include "screamer/backtest_price_target.h"
#include "screamer/backtest_ohlc_orders.h"
#include "screamer/backtest_ohlc_target.h"
#include "screamer/backtest_ohlcv_target.h"
#include "screamer/backtest_trades_orders.h"
#include "screamer/backtest_trades_target.h"
#include "screamer/backtest_l1_orders.h"
#include "screamer/backtest_l1_target.h"
#include "screamer/backtest_l1trades_orders.h"
#include "screamer/backtest_report.h"
#include "screamer/backtest_stats.h"
#include "screamer/backtest_carry.h"
#include "screamer/validation/surrogates.h"
#include "screamer/validation/sharpe.h"
#include "screamer/portfolio_report.h"
#include "screamer/common/row_op.h"
#include "screamer/cross_section.h"
#include "screamer/backtest_portfolio_target.h"
#include "screamer/common/per_asset.h"
#include "screamer/rolling_downside_deviation.h"
#include "screamer/rolling_omega.h"
#include "screamer/rolling_cvar.h"
#include "screamer/rolling_max_drawdown.h"
#include "screamer/rolling_sharpe.h"
#include "screamer/rolling_sortino.h"
#include "screamer/rolling_info_ratio.h"
#include "screamer/rolling_calmar.h"
#include "screamer/rolling_hit_rate.h"
#include "screamer/rolling_alpha.h"
#include "screamer/rolling_residual_std.h"
#include "screamer/rolling_linear_regression.h"
#include "screamer/bayesian_regression.h"
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

// A per-asset parameter accepts a single number (one value for all assets) or
// a sequence of numbers (one per asset).
namespace nanobind { namespace detail {
template <> struct type_caster<screamer::PerAsset> {
    NB_TYPE_CASTER(screamer::PerAsset, const_name("float | Sequence[float]"))

    bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
        if (!PySequence_Check(src.ptr())) {
            const double d = PyFloat_AsDouble(src.ptr());
            if (d == -1.0 && PyErr_Occurred()) {
                PyErr_Clear();
                return false;
            }
            value.values = {d};
            return true;
        }
        make_caster<std::vector<double>> seq;
        if (!seq.from_python(src, flags, cleanup)) return false;
        value.values = seq.value;
        return true;
    }

    // One value reads back as a number, the form a user passes it in.
    static handle from_cpp(const screamer::PerAsset& v, rv_policy policy,
                           cleanup_list* cleanup) noexcept {
        if (v.values.size() == 1) return PyFloat_FromDouble(v.values[0]);
        return make_caster<std::vector<double>>::from_cpp(v.values, policy, cleanup);
    }
};
}} // namespace nanobind::detail

namespace {

// Parse one PortfolioReport event. The event is a contiguous (assets, 4)
// array/list with columns [equity, pnl, position, cost].
nb::object portfolio_report_row(screamer::PortfolioReport& self, nb::object row) {
    if (screamer::detail::is_unsupported_dtype_array(row))
        row = screamer::detail::coerce_to_f64(row);
    if (!screamer::detail::is_ndarray(row)) {
        if (!(nb::isinstance<nb::list>(row) || nb::isinstance<nb::tuple>(row)))
            throw nb::type_error("PortfolioReport rows must be arrays or sequences.");
        nb::module_ np = nb::module_::import_("numpy");
        row = np.attr("asarray")(row, nb::arg("dtype") = np.attr("float64"));
    }
    auto values = screamer::detail::read_contig_double(nb::cast<nb::ndarray<nb::ro>>(row));
    if (values.ndim() == 1) {
        if (values.shape[0] == 0 || values.shape[0] % 4 != 0)
            throw nb::value_error("PortfolioReport rows must have a multiple of 4 values.");
        return nb::cast(self.call_assets(values.data.data(), values.shape[0] / 4));
    }
    if (values.ndim() != 2 || values.shape[1] != 4)
        throw nb::value_error("PortfolioReport rows must have shape (assets, 4).");
    return nb::cast(self.call_assets(values.data.data(), values.shape[0]));
}

nb::object portfolio_report_batch(screamer::PortfolioReport& self,
                                  const screamer::detail::ContigDouble& values) {
    if (values.ndim() != 3 || values.shape[2] != 4)
        throw nb::value_error("PortfolioReport batch input must have shape (time, assets, 4).");
    const std::size_t time = values.shape[0];
    const std::size_t assets = values.shape[1];
    if (assets == 0)
        throw nb::value_error("PortfolioReport requires at least one asset.");
    double* out = new double[time * 6 ? time * 6 : 1];
    self.reset();
    for (std::size_t t = 0; t < time; ++t) {
        auto result = self.call_assets(values.data.data() + t * assets * 4, assets);
        screamer::detail::write_tuple_to_memory(out + t * 6, result);
    }
    self.reset();
    return screamer::detail::make_owned_array(out, {time, 6});
}

// One PortfolioReport event per step: an (assets, 4) row, for the sync and
// async iterators alike. A stream is an independent run (decision D8): it runs
// on its own copy and resets the operator's running series when it opens.
class PortfolioReportStepper final : public screamer::EventStep {
public:
    explicit PortfolioReportStepper(const nb::object& owner)
        : owner_(screamer::EventStepper::fresh_copy(owner)),
          op_(nb::cast<screamer::PortfolioReport&>(owner_)) {
        nb::cast<screamer::PortfolioReport&>(owner).reset();
    }

    nb::object step(const std::vector<nb::object>& items) override {
        return portfolio_report_row(op_, items[0]);
    }

private:
    nb::object owner_;                    // the stream's own copy
    screamer::PortfolioReport& op_;
};

nb::object portfolio_report_call(screamer::PortfolioReport& self, nb::args raw) {
    nb::args args = screamer::normalize_args(raw, self);
    bool has_node = false;
    for (nb::handle arg : args)
        has_node = has_node || screamer::is_dag_node(nb::borrow<nb::object>(arg));
    if (has_node) {
        // DAG composition uses the scalar-compatible four-input path. The
        // multi-asset (time, assets, 4) reduction is a batch/stream contract.
        if (args.size() != 4)
            throw nb::type_error("PortfolioReport DAG composition expects four portfolio columns.");
        return screamer::make_dag_functor_node(nb::find(self), nb::cast<nb::tuple>(args));
    }
    if (args.size() != 1)
        throw nb::type_error("PortfolioReport expects one engine-output array or row stream.");
    nb::object input = nb::borrow<nb::object>(args[0]);
    if (screamer::is_async_iterable(input))
        return nb::cast(screamer::LazyAsyncIterator(
            std::make_unique<PortfolioReportStepper>(nb::find(self)), {input}));
    if (screamer::detail::is_unsupported_dtype_array(input))
        input = screamer::detail::coerce_to_f64(input);
    if (screamer::detail::is_ndarray(input)) {
        auto values = screamer::detail::read_contig_double(nb::cast<nb::ndarray<nb::ro>>(input));
        if (values.ndim() == 3) return portfolio_report_batch(self, values);
        if (values.ndim() == 2 || values.ndim() == 1)
            return portfolio_report_row(self, input);
        throw nb::value_error("PortfolioReport input must have rank 1, 2, or 3.");
    }
    if (screamer::is_lazy_iterable(input)) {
        return nb::cast(screamer::LazyEvalIterator(
            std::make_unique<PortfolioReportStepper>(nb::find(self)), {input}));
    }
    if (nb::isinstance<nb::list>(input) || nb::isinstance<nb::tuple>(input)) {
        // NumPy normalizes nested sequences, so a list can be either one row
        // (assets, 4) or a complete (time, assets, 4) batch.
        nb::module_ np = nb::module_::import_("numpy");
        nb::object arr = np.attr("asarray")(input, nb::arg("dtype") = np.attr("float64"));
        auto values = screamer::detail::read_contig_double(nb::cast<nb::ndarray<nb::ro>>(arr));
        if (values.ndim() == 3) return portfolio_report_batch(self, values);
        return portfolio_report_row(self, arr);
    }
    throw nb::type_error("PortfolioReport input must be a numpy array or iterable of rows.");
}

// ---- Row operators (cross-sectional operators, portfolio engines) ---------
//
// A RowOp event is one row of K assets with fields_in values per asset. With
// (T, K) arrays, one per input field, the call is T events and returns (T, K)
// (one output field) or (T, K, fields_out); with 1-D rows of K values it is one
// event and returns (K,) or (K, fields_out). A single-field operator also
// takes an iterator of rows and returns a lazy iterator of rows.

screamer::detail::ContigDouble row_op_input(nb::object x) {
    if (screamer::detail::is_unsupported_dtype_array(x))
        x = screamer::detail::coerce_to_f64(x);
    if (!screamer::detail::is_ndarray(x)) {
        nb::module_ np = nb::module_::import_("numpy");
        x = np.attr("asarray")(x, nb::arg("dtype") = np.attr("float64"));
    }
    return screamer::detail::read_contig_double(nb::cast<nb::ndarray<nb::ro>>(x));
}

std::vector<std::size_t> row_op_shape(std::size_t time, std::size_t k, std::size_t fo,
                                      bool batch) {
    std::vector<std::size_t> shape;
    if (batch) shape.push_back(time);
    shape.push_back(k);
    if (fo > 1) shape.push_back(fo);
    return shape;
}

nb::object row_op_run(screamer::RowOp& self,
                      const std::vector<screamer::detail::ContigDouble>& fields) {
    const std::size_t fi = self.fields_in(), fo = self.fields_out();
    const std::size_t nd = fields[0].ndim();
    if (nd != 1 && nd != 2)
        throw nb::value_error(
            "a row operator takes (T, K) arrays, rows of K values, or an iterator of rows.");
    for (const auto& f : fields)
        if (f.shape != fields[0].shape)
            throw nb::value_error("all inputs of a row operator must have the same shape.");
    const bool batch = nd == 2;
    const std::size_t time = batch ? fields[0].shape[0] : 1;
    const std::size_t k = fields[0].shape[nd - 1];
    if (k == 0) throw nb::value_error("a row operator needs at least one asset.");
    const std::size_t n_out = time * k * fo;
    double* out = new double[n_out ? n_out : 1];
    std::vector<double> event(fi * k);
    if (batch) self.reset();
    try {
        for (std::size_t t = 0; t < time; ++t) {
            const double* in = fields[0].data.data() + t * k;
            if (fi > 1) {
                for (std::size_t f = 0; f < fi; ++f)
                    std::copy_n(fields[f].data.data() + t * k, k, event.data() + f * k);
                in = event.data();
            }
            self.call_row(in, out + t * k * fo, k);
        }
    } catch (...) {
        delete[] out;
        if (batch) self.reset();
        throw;
    }
    if (batch) self.reset();
    return screamer::detail::make_owned_array(out, row_op_shape(time, k, fo, batch));
}

// One RowOp event per step, for the sync and async iterators alike. The items
// are fields_in rows of K values, one per source, or with one source and
// fields_in > 1 a tuple or list of fields_in rows. A (K, 1) row reads as (K,)
// (rule 1 on an event). Each step equals the same row of the batch call. A
// stream is an independent run (decision D8): it runs on its own copy and
// resets the operator's running series when it opens.
class RowStepper final : public screamer::EventStep {
public:
    RowStepper(const nb::object& owner, std::size_t n_sources)
        : owner_(screamer::EventStepper::fresh_copy(owner)),
          op_(nb::cast<screamer::RowOp&>(owner_)),
          unpack_(n_sources == 1 && op_.fields_in() > 1) {
        nb::cast<screamer::RowOp&>(owner).reset();
    }

    nb::object step(const std::vector<nb::object>& items) override {
        fields_.clear();
        if (unpack_) {
            nb::handle event = items[0];
            if (!(nb::isinstance<nb::tuple>(event) || nb::isinstance<nb::list>(event)) ||
                nb::len(event) != op_.fields_in())
                throw nb::value_error(("each event of one row stream is a tuple of " +
                    std::to_string(op_.fields_in()) + " rows of K values.").c_str());
            for (nb::handle row : event) add(nb::borrow<nb::object>(row));
        } else {
            for (const auto& row : items) add(row);
        }
        return row_op_run(op_, fields_);
    }

private:
    void add(nb::object row) {
        fields_.push_back(row_op_input(std::move(row)));
        auto& f = fields_.back();
        if (f.ndim() == 2 && f.shape[1] == 1) f.shape.pop_back();
        if (f.ndim() != 1)
            throw nb::value_error("each event of a row stream is one row of K values.");
    }

    nb::object owner_;                    // the stream's own copy
    screamer::RowOp& op_;
    bool unpack_;
    std::vector<screamer::detail::ContigDouble> fields_;
};

nb::object row_op_call(screamer::RowOp& self, nb::args raw) {
    nb::args args = screamer::normalize_args(raw, self);
    for (nb::handle a : args)
        if (screamer::is_dag_node(nb::borrow<nb::object>(a)))
            throw nb::type_error("row operators run on arrays and row streams, "
                                 "not inside a Pipeline.");
    const std::size_t fi = self.fields_in(), n = args.size();
    // Streams: fields_in sources of rows, or one source of fields_in-tuples.
    const bool async = screamer::all_async(args);
    std::size_t n_lazy = 0;
    for (nb::handle a : args) n_lazy += screamer::is_lazy_iterable(a) ? 1 : 0;
    if (n_lazy != 0 && n_lazy != n)
        throw nb::type_error("iterator and array inputs cannot be mixed: pass every input "
                             "as an iterator of rows, or none");
    const bool lazy = !async && n_lazy == n;
    if ((async || lazy) && (n == fi || n == 1)) {
        std::vector<nb::object> sources;
        for (nb::handle a : args) sources.push_back(nb::borrow<nb::object>(a));
        auto stepper = std::make_unique<RowStepper>(nb::find(self), n);
        if (async) return nb::cast(screamer::LazyAsyncIterator(std::move(stepper), sources));
        return nb::cast(screamer::LazyEvalIterator(std::move(stepper), sources));
    }
    if (n != fi)
        throw nb::type_error(("this operator takes " + std::to_string(fi) +
                              " inputs, got " + std::to_string(n) + ".").c_str());
    std::vector<screamer::detail::ContigDouble> fields;
    for (nb::handle a : args) {
        fields.push_back(row_op_input(nb::borrow<nb::object>(a)));
        // Rule 1: (T, K, 1) reads as (T, K); the data is already in that order.
        fields.back().shape = screamer::shape::consume_unit_field(fields.back().shape);
    }
    return row_op_run(self, fields);
}

} // namespace

namespace {

// Batch-only surrogate generators for randomization tests (see
// screamer/validation.py). Each returns a new (T, ncols) float64 array holding
// columns col0 .. col0 + ncols - 1 of the surrogate set.
using SurrogateIn = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

nb::object circular_shift_batch(SurrogateIn x, long long min_shift,
                                std::uint64_t seed, size_t col0, size_t ncols,
                                long long start) {
    const size_t T = x.shape(0);
    const size_t m = min_shift < 0
        ? screamer::validation::default_min_shift(x.data(), T)
        : static_cast<size_t>(min_shift);
    double* out = new double[T * ncols];
    try {
        screamer::validation::circular_shift(
            x.data(), T, m, seed, col0, ncols, out,
            start < 0 ? static_cast<size_t>(-1) : static_cast<size_t>(start));
    } catch (...) { delete[] out; throw; }
    return screamer::detail::make_owned_array(out, {T, ncols});
}

nb::object segment_shuffle_batch(SurrogateIn x, std::uint64_t seed,
                                 size_t col0, size_t ncols) {
    const size_t T = x.shape(0);
    double* out = new double[T * ncols];
    screamer::validation::segment_shuffle(x.data(), T, seed, col0, ncols, out);
    return screamer::detail::make_owned_array(out, {T, ncols});
}

nb::object permute_bars_batch(SurrogateIn o, SurrogateIn h, SurrogateIn l, SurrogateIn c,
                              const std::string& mode, std::uint64_t seed,
                              size_t col0, size_t ncols,
                              SurrogateIn volume, bool notional) {
    const size_t T = o.shape(0);
    if (h.shape(0) != T || l.shape(0) != T || c.shape(0) != T)
        throw std::invalid_argument("open, high, low, and close must have the same length.");
    const bool has_vol = volume.shape(0) > 0;
    if (has_vol && volume.shape(0) != T)
        throw std::invalid_argument("volume must have the same length as the prices.");
    const auto m = screamer::validation::parse_permute_mode(mode);
    const size_t N = T * ncols;
    std::vector<double*> bufs;
    for (int k = 0; k < (has_vol ? 6 : 5); ++k) bufs.push_back(new double[N]);
    try {
        screamer::validation::permute_bars(
            o.data(), h.data(), l.data(), c.data(), T, m, seed, col0, ncols,
            bufs[0], bufs[1], bufs[2], bufs[3], bufs[4],
            has_vol ? volume.data() : nullptr, notional, has_vol ? bufs[5] : nullptr);
    } catch (...) {
        for (double* b : bufs) delete[] b;
        throw;
    }
    using screamer::detail::make_owned_array;
    nb::list out;
    for (double* b : bufs) out.append(make_owned_array(b, {T, ncols}));
    return nb::tuple(out);
}

nb::object permute_returns_batch(SurrogateIn x, const std::string& mode, std::uint64_t seed,
                                 size_t col0, size_t ncols,
                                 SurrogateIn volume, bool notional) {
    const size_t T = x.shape(0);
    const bool has_vol = volume.shape(0) > 0;
    if (has_vol && volume.shape(0) != T)
        throw std::invalid_argument("volume must have the same length as the prices.");
    const auto m = screamer::validation::parse_permute_mode(mode);
    std::vector<double*> bufs;
    for (int k = 0; k < (has_vol ? 3 : 2); ++k) bufs.push_back(new double[T * ncols]);
    try {
        screamer::validation::permute_returns(
            x.data(), T, m, seed, col0, ncols, bufs[0], bufs[1],
            has_vol ? volume.data() : nullptr, notional, has_vol ? bufs[2] : nullptr);
    } catch (...) {
        for (double* b : bufs) delete[] b;
        throw;
    }
    using screamer::detail::make_owned_array;
    nb::list out;
    for (double* b : bufs) out.append(make_owned_array(b, {T, ncols}));
    return nb::tuple(out);
}

}  // namespace

void init_bindings_fin(nb::module_& m) {
    m.def("_probabilistic_sharpe", [](SurrogateIn pnl, double benchmark) {
        return screamer::validation::probabilistic_sharpe(pnl.data(), pnl.shape(0), benchmark);
    }, "pnl"_a, "benchmark"_a);
    m.def("_deflated_sharpe", [](SurrogateIn pnl, SurrogateIn trials) {
        double benchmark = 0.0;
        const double p = screamer::validation::deflated_sharpe(
            pnl.data(), pnl.shape(0), trials.data(), trials.shape(0), benchmark);
        return nb::make_tuple(p, benchmark);
    }, "pnl"_a, "trials"_a);
    m.def("_permute_bars", &permute_bars_batch,
          "open"_a, "high"_a, "low"_a, "close"_a, "mode"_a, "seed"_a, "col0"_a, "ncols"_a,
          "volume"_a, "notional"_a);
    m.def("_permute_returns", &permute_returns_batch,
          "x"_a, "mode"_a, "seed"_a, "col0"_a, "ncols"_a, "volume"_a, "notional"_a);
    m.def("_circular_shift", &circular_shift_batch,
          "x"_a, "min_shift"_a, "seed"_a, "col0"_a, "ncols"_a, "start"_a = -1);
    m.def("_segment_shuffle", &segment_shuffle_batch,
          "x"_a, "seed"_a, "col0"_a, "ncols"_a);
    m.def("_default_min_shift", [](SurrogateIn x) {
        return screamer::validation::default_min_shift(x.data(), x.shape(0));
    }, "x"_a);


    nb::class_<screamer::Return, screamer::ScreamerBase>(m, "Return")
        .def(nb::init<int>(), "window_size"_a = 1)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Return::reset, "Reset to the initial state.");

    nb::class_<screamer::LogReturn, screamer::ScreamerBase>(m, "LogReturn")
        .def(nb::init<int>(), "window_size"_a = 1)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::LogReturn::reset, "Reset to the initial state.");

    // ROC family: rate-of-change variants. TA-Lib has all three as
    // separate functions; we provide them under TA-Lib's names so
    // users can port directly. ROCP is mathematically identical to
    // Return.
    nb::class_<screamer::ROC, screamer::ScreamerBase>(m, "ROC")
        // TA-Lib's timeperiod default is 10, which is what the docs page and
        // the sibling Momentum both declare. This was 1.
        .def(nb::init<int>(), "window_size"_a = 10)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::ROC::reset, "Reset to the initial state.");

    nb::class_<screamer::ROCP, screamer::ScreamerBase>(m, "ROCP")
        // TA-Lib's timeperiod default is 10, which is what the docs page and
        // the sibling Momentum both declare. This was 1.
        .def(nb::init<int>(), "window_size"_a = 10)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::ROCP::reset, "Reset to the initial state.");

    nb::class_<screamer::ROCR, screamer::ScreamerBase>(m, "ROCR")
        // TA-Lib's timeperiod default is 10, which is what the docs page and
        // the sibling Momentum both declare. This was 1.
        .def(nb::init<int>(), "window_size"_a = 10)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::ROCR::reset, "Reset to the initial state.");

    // RollingCorr: 2 inputs (x, y), 1 output (Pearson correlation).
    // Inherits from FunctorBase<_, 2, 1>, NOT ScreamerBase -- the
    // multi-input class hierarchy is separate. handle_input dispatches
    // on the variadic args (scalars / N parallel arrays / list of N-tuples
    // / N parallel iterables).
    nb::class_<screamer::RollingCorr, screamer::EvalOp>(m, "RollingCorr")
        .def(nb::init<int, const std::string&>(),
             "window_size"_a = 20,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::functor_call<screamer::RollingCorr>)
        .def("reset", &screamer::RollingCorr::reset, "Reset to the initial state.");

    // Rolling sample covariance of two streams.
    nb::class_<screamer::RollingCov, screamer::EvalOp>(m, "RollingCov")
        .def(nb::init<int, const std::string&>(),
             "window_size"_a = 20,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::functor_call<screamer::RollingCov>)
        .def("reset", &screamer::RollingCov::reset, "Reset to the initial state.");

    // Rolling regression slope of x on y: beta = cov(x, y) / var(y).
    nb::class_<screamer::RollingTLSBeta, screamer::EvalOp>(m, "RollingTLSBeta")
        .def(nb::init<int, const std::string&>(),
             "window_size"_a = 20,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::functor_call<screamer::RollingTLSBeta>)
        .def("reset", &screamer::RollingTLSBeta::reset, "Reset to the initial state.");

    nb::class_<screamer::KalmanRegression, screamer::EvalOp>(m, "KalmanRegression")
        .def(nb::init<double, double, double>(),
             "delta"_a = 1e-6, "obs_var"_a = 1e-3, "prior_var"_a = 1.0)
        .def("__call__", &screamer::functor_call<screamer::KalmanRegression>)
        .def("reset", &screamer::KalmanRegression::reset, "Reset to the diffuse prior.");

    nb::class_<screamer::RollingADF, screamer::ScreamerBase>(m, "RollingADF")
        .def(nb::init<int, int>(), "window_size"_a = 100, "lags"_a = 1)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingADF::reset, "Reset to the initial state.");

    nb::class_<screamer::BandSignal, screamer::ScreamerBase>(m, "BandSignal")
        .def(nb::init<double, double, double>(),
             "entry"_a = 2.0, "exit"_a = 0.0,
             "stop"_a = std::numeric_limits<double>::infinity())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::BandSignal::reset, "Reset to flat.");

    nb::class_<screamer::RollingBeta, screamer::EvalOp>(m, "RollingBeta")
        .def(nb::init<int, const std::string&>(),
             "window_size"_a = 20,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::functor_call<screamer::RollingBeta>)
        .def("reset", &screamer::RollingBeta::reset, "Reset to the initial state.");

    // Hedge-adjusted residual of x against y: spread = x - beta * y, with
    // beta computed exactly as in RollingBeta.
    nb::class_<screamer::RollingSpread, screamer::EvalOp>(m, "RollingSpread")
        .def(nb::init<int, const std::string&>(),
             "window_size"_a = 20,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::functor_call<screamer::RollingSpread>)
        .def("reset", &screamer::RollingSpread::reset, "Reset to the initial state.");

    // ----- Performance / risk metrics -----
    nb::class_<screamer::Drawdown, screamer::ScreamerBase>(m, "Drawdown")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Drawdown::reset, "Reset.");

    nb::class_<screamer::MaxDrawdown, screamer::ScreamerBase>(m, "MaxDrawdown")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::MaxDrawdown::reset, "Reset.");

    nb::class_<screamer::RollingMaxDrawdown, screamer::ScreamerBase>(m, "RollingMaxDrawdown")
        .def(nb::init<int>(), "window_size"_a = 252)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingMaxDrawdown::reset, "Reset.");

    nb::class_<screamer::RollingSharpe, screamer::ScreamerBase>(m, "RollingSharpe")
        .def(nb::init<int, double>(),
             "window_size"_a = 252,
             "periods_per_year"_a = 1.0)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingSharpe::reset, "Reset.");

    nb::class_<screamer::RollingSortino, screamer::ScreamerBase>(m, "RollingSortino")
        .def(nb::init<int, double, double>(),
             "window_size"_a = 252,
             "periods_per_year"_a = 1.0,
             "target"_a = 0.0)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingSortino::reset, "Reset.");

    nb::class_<screamer::RollingInfoRatio, screamer::EvalOp>(m, "RollingInfoRatio")
        .def(nb::init<int, double>(),
             "window_size"_a = 252,
             "periods_per_year"_a = 1.0)
        .def("__call__", &screamer::functor_call<screamer::RollingInfoRatio>)
        .def("reset", &screamer::RollingInfoRatio::reset, "Reset.");

    nb::class_<screamer::RollingCalmar, screamer::ScreamerBase>(m, "RollingCalmar")
        .def(nb::init<int, double>(),
             "window_size"_a = 252,
             "periods_per_year"_a = 1.0)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingCalmar::reset, "Reset.");

    nb::class_<screamer::RollingHitRate, screamer::ScreamerBase>(m, "RollingHitRate")
        .def(nb::init<int>(), "window_size"_a = 252)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingHitRate::reset, "Reset.");

    // ----- Regression-family additions -----
    nb::class_<screamer::RollingAlpha, screamer::EvalOp>(m, "RollingAlpha")
        .def(nb::init<int, const std::string&>(),
             "window_size"_a = 20,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::functor_call<screamer::RollingAlpha>)
        .def("reset", &screamer::RollingAlpha::reset, "Reset.");

    nb::class_<screamer::RollingResidualStd, screamer::EvalOp>(m, "RollingResidualStd")
        .def(nb::init<int, const std::string&>(),
             "window_size"_a = 20,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::functor_call<screamer::RollingResidualStd>)
        .def("reset", &screamer::RollingResidualStd::reset, "Reset.");

    // 2 -> 4 OLS fit returning (slope, intercept, r_squared, stderr).
    // First 2->4 consumer of the N->M dispatcher.
    nb::class_<screamer::RollingLinearRegression, screamer::EvalOp>(m, "RollingLinearRegression")
        .def(nb::init<int, const std::string&>(),
             "window_size"_a = 20,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::functor_call<screamer::RollingLinearRegression>)
        .def("reset", &screamer::RollingLinearRegression::reset, "Reset.");

    // 2 -> 4 online Bayesian regression returning (pred_mean, pred_std, slope, intercept).
    // Uses exponential forgetting with a conjugate Normal-Inverse-Gamma prior.
    nb::class_<screamer::BayesianRegression, screamer::EvalOp>(m, "BayesianRegression")
        .def(nb::init<std::optional<double>, std::optional<double>, std::optional<double>,
                      std::optional<double>, double, double>(),
             "com"_a = nb::none(), "span"_a = nb::none(),
             "halflife"_a = nb::none(), "alpha"_a = nb::none(),
             "prior_precision"_a = 1.0, "prior_sigma"_a = 1.0)
        .def("__call__", &screamer::functor_call<screamer::BayesianRegression>)
        .def("reset", &screamer::BayesianRegression::reset, "Reset to the prior.");

    nb::class_<screamer::BacktestPriceTarget, screamer::EvalOp>(m, "BacktestPriceTarget")
        .def(nb::init<double, double, double, double, double, double, double, const std::string&, double, double, double, double, const std::string&, double, double>(),
             "spread"_a = 0.0, "fee"_a = 0.0,
             "min_position"_a = -std::numeric_limits<double>::infinity(),
             "max_position"_a = std::numeric_limits<double>::infinity(),
             "multiplier"_a = 1.0, "fee_per_contract"_a = 0.0,
             "slippage"_a = 0.0, "slippage_unit"_a = "fraction", "tick_size"_a = 0.0,
             "stop_loss"_a = 0.0, "take_profit"_a = 0.0, "trailing_stop"_a = 0.0,
             "size"_a = "units", "initial_capital"_a = 0.0, "rebalance_band"_a = 0.0)
        .def("__call__", &screamer::functor_call<screamer::BacktestPriceTarget>)
        .def("reset", &screamer::BacktestPriceTarget::reset, "Reset.");

    nb::class_<screamer::BacktestOHLCOrders, screamer::EvalOp>(m, "BacktestOHLCOrders")
        .def(nb::init<double, double, const std::string&, double, double, double, double, double, double, double, double, const std::string&>(),
             "maker_fee"_a = 0.0, "taker_fee"_a = 0.0,
             "fill"_a = "touch", "participation_ratio"_a = 1.0,
             "tick_size"_a = 0.0,
             "min_position"_a = -std::numeric_limits<double>::infinity(),
             "max_position"_a = std::numeric_limits<double>::infinity(),
             "multiplier"_a = 1.0,
             "maker_fee_per_contract"_a = 0.0,
             "taker_fee_per_contract"_a = 0.0,
             "slippage"_a = 0.0, "slippage_unit"_a = "fraction")
        .def("__call__", &screamer::functor_call<screamer::BacktestOHLCOrders>)
        .def("reset", &screamer::BacktestOHLCOrders::reset, "Reset.");

    nb::class_<screamer::BacktestOHLCTarget, screamer::EvalOp>(m, "BacktestOHLCTarget")
        .def(nb::init<double, double, double, double, double, double, double, const std::string&, const std::string&, int, double, double, double, const std::string&, double, double>(),
             "taker_fee"_a = 0.0, "tick_size"_a = 0.0,
             "min_position"_a = -std::numeric_limits<double>::infinity(),
             "max_position"_a = std::numeric_limits<double>::infinity(),
             "multiplier"_a = 1.0, "fee_per_contract"_a = 0.0,
             "slippage"_a = 0.0, "slippage_unit"_a = "fraction",
             "fill_at"_a = "open", "delay"_a = 1,
             "stop_loss"_a = 0.0, "take_profit"_a = 0.0, "trailing_stop"_a = 0.0,
             "size"_a = "units", "initial_capital"_a = 0.0, "rebalance_band"_a = 0.0)
        .def("__call__", &screamer::functor_call<screamer::BacktestOHLCTarget>)
        .def("reset", &screamer::BacktestOHLCTarget::reset, "Reset.");

    nb::class_<screamer::BacktestOHLCVTarget, screamer::EvalOp>(m, "BacktestOHLCVTarget")
        .def(nb::init<double, double, double, double, double, double, double, double, const std::string&, const std::string&, int, double, double, double, const std::string&, double, double>(),
             "max_participation"_a = 1.0, "taker_fee"_a = 0.0, "tick_size"_a = 0.0,
             "min_position"_a = -std::numeric_limits<double>::infinity(),
             "max_position"_a = std::numeric_limits<double>::infinity(),
             "multiplier"_a = 1.0, "fee_per_contract"_a = 0.0,
             "slippage"_a = 0.0, "slippage_unit"_a = "fraction",
             "fill_at"_a = "open", "delay"_a = 1,
             "stop_loss"_a = 0.0, "take_profit"_a = 0.0, "trailing_stop"_a = 0.0,
             "size"_a = "units", "initial_capital"_a = 0.0, "rebalance_band"_a = 0.0)
        .def("__call__", &screamer::functor_call<screamer::BacktestOHLCVTarget>)
        .def("reset", &screamer::BacktestOHLCVTarget::reset, "Reset.");

    nb::class_<screamer::BacktestTradesOrders, screamer::EvalOp>(m, "BacktestTradesOrders")
        .def(nb::init<double, double, const std::string&, double, double, double, double, double, double, double, double, const std::string&>(),
             "maker_fee"_a = 0.0, "taker_fee"_a = 0.0,
             "fill"_a = "touch", "participation_ratio"_a = 1.0,
             "tick_size"_a = 0.0,
             "min_position"_a = -std::numeric_limits<double>::infinity(),
             "max_position"_a = std::numeric_limits<double>::infinity(),
             "multiplier"_a = 1.0,
             "maker_fee_per_contract"_a = 0.0,
             "taker_fee_per_contract"_a = 0.0,
             "slippage"_a = 0.0, "slippage_unit"_a = "fraction")
        .def("__call__", &screamer::functor_call<screamer::BacktestTradesOrders>)
        .def("reset", &screamer::BacktestTradesOrders::reset, "Reset.");

    nb::class_<screamer::BacktestL1Orders, screamer::EvalOp>(m, "BacktestL1Orders")
        .def(nb::init<double, double, const std::string&, double, double, double, double, double, double, double, double, const std::string&>(),
             "maker_fee"_a = 0.0, "taker_fee"_a = 0.0,
             "fill"_a = "breach", "participation_ratio"_a = 1.0,
             "tick_size"_a = 0.0,
             "min_position"_a = -std::numeric_limits<double>::infinity(),
             "max_position"_a = std::numeric_limits<double>::infinity(),
             "multiplier"_a = 1.0,
             "maker_fee_per_contract"_a = 0.0,
             "taker_fee_per_contract"_a = 0.0,
             "slippage"_a = 0.0, "slippage_unit"_a = "fraction")
        .def("__call__", &screamer::functor_call<screamer::BacktestL1Orders>)
        .def("reset", &screamer::BacktestL1Orders::reset, "Reset.");

    nb::class_<screamer::BacktestL1Target, screamer::EvalOp>(m, "BacktestL1Target")
        .def(nb::init<double, double, double, double, double, double, double, const std::string&, double, double, double, const std::string&, double, double>(),
             "taker_fee"_a = 0.0, "tick_size"_a = 0.0,
             "min_position"_a = -std::numeric_limits<double>::infinity(),
             "max_position"_a = std::numeric_limits<double>::infinity(),
             "multiplier"_a = 1.0, "fee_per_contract"_a = 0.0,
             "slippage"_a = 0.0, "slippage_unit"_a = "fraction",
             "stop_loss"_a = 0.0, "take_profit"_a = 0.0, "trailing_stop"_a = 0.0,
             "size"_a = "units", "initial_capital"_a = 0.0, "rebalance_band"_a = 0.0)
        .def("__call__", &screamer::functor_call<screamer::BacktestL1Target>)
        .def("reset", &screamer::BacktestL1Target::reset, "Reset.");

    nb::class_<screamer::BacktestL1TradesOrders, screamer::EvalOp>(m, "BacktestL1TradesOrders")
        .def(nb::init<double, double, const std::string&, double, double, double, double, double, double, double, double, const std::string&>(),
             "maker_fee"_a = 0.0, "taker_fee"_a = 0.0,
             "fill"_a = "touch", "participation_ratio"_a = 1.0,
             "tick_size"_a = 0.0,
             "min_position"_a = -std::numeric_limits<double>::infinity(),
             "max_position"_a = std::numeric_limits<double>::infinity(),
             "multiplier"_a = 1.0,
             "maker_fee_per_contract"_a = 0.0,
             "taker_fee_per_contract"_a = 0.0,
             "slippage"_a = 0.0, "slippage_unit"_a = "fraction")
        .def("__call__", &screamer::functor_call<screamer::BacktestL1TradesOrders>)
        .def("reset", &screamer::BacktestL1TradesOrders::reset, "Reset.");

    nb::class_<screamer::BacktestTradesTarget, screamer::EvalOp>(m, "BacktestTradesTarget")
        .def(nb::init<double, double, double, double, double, double, double, const std::string&, double, double, double, const std::string&, double, double, double>(),
             "taker_fee"_a = 0.0, "tick_size"_a = 0.0,
             "min_position"_a = -std::numeric_limits<double>::infinity(),
             "max_position"_a = std::numeric_limits<double>::infinity(),
             "multiplier"_a = 1.0, "fee_per_contract"_a = 0.0,
             "slippage"_a = 0.0, "slippage_unit"_a = "fraction",
             "stop_loss"_a = 0.0, "take_profit"_a = 0.0, "trailing_stop"_a = 0.0,
             "size"_a = "units", "initial_capital"_a = 0.0, "rebalance_band"_a = 0.0,
             "max_participation"_a = std::numeric_limits<double>::infinity())
        .def("__call__", &screamer::functor_call<screamer::BacktestTradesTarget>)
        .def("reset", &screamer::BacktestTradesTarget::reset, "Reset.");

    nb::class_<screamer::BacktestReport, screamer::EvalOp>(m, "BacktestReport")
        .def(nb::init<double>(), "periods_per_year"_a = 1.0)
        .def("__call__", &screamer::functor_call<screamer::BacktestReport>)
        .def("reset", &screamer::BacktestReport::reset, "Reset.");

    nb::class_<screamer::BacktestCarry, screamer::EvalOp>(m, "BacktestCarry")
        .def(nb::init<double>(), "multiplier"_a = 1.0)
        .def("__call__", &screamer::functor_call<screamer::BacktestCarry>)
        .def("reset", &screamer::BacktestCarry::reset, "Reset.");

    nb::class_<screamer::BacktestStats, screamer::EvalOp>(m, "BacktestStats")
        .def(nb::init<double>(), "periods_per_year"_a = 1.0)
        .def("__call__", &screamer::functor_call<screamer::BacktestStats>)
        .def("reset", &screamer::BacktestStats::reset, "Reset.");

    nb::class_<screamer::PortfolioReport, screamer::EvalOp>(m, "PortfolioReport")
        .def(nb::init<>())
        .def("__call__", &portfolio_report_call)
        .def("reset", &screamer::PortfolioReport::reset, "Reset.");


    nb::class_<screamer::RowOp, screamer::EvalOp>(m, "RowOp")
        .def("__call__", &row_op_call)
        .def("reset", &screamer::RowOp::reset, "Reset.");


    nb::class_<screamer::CrossRank, screamer::RowOp>(m, "CrossRank")
        .def(nb::init<const std::string&>(), "scale"_a = "signed");

    nb::class_<screamer::CrossZscore, screamer::RowOp>(m, "CrossZscore")
        .def(nb::init<>());

    nb::class_<screamer::CrossDemean, screamer::RowOp>(m, "CrossDemean")
        .def(nb::init<>());

    nb::class_<screamer::CrossWinsorize, screamer::RowOp>(m, "CrossWinsorize")
        .def(nb::init<double, double>(), "lower"_a = 0.05, "upper"_a = 0.95);

    nb::class_<screamer::CrossNormalize, screamer::RowOp>(m, "CrossNormalize")
        .def(nb::init<double, std::optional<double>>(), "gross"_a = 1.0,
             "net"_a = nb::none());

    nb::class_<screamer::CrossTopBottom, screamer::RowOp>(m, "CrossTopBottom")
        .def(nb::init<int>(), "count"_a = 1);

    nb::class_<screamer::BacktestPortfolioTarget, screamer::RowOp>(m, "BacktestPortfolioTarget")
        .def(nb::init<screamer::PerAsset, screamer::PerAsset, const std::string&,
                      screamer::PerAsset, screamer::PerAsset, screamer::PerAsset,
                      const std::string&, double, double, const std::string&,
                      double, double>(),
             "fee"_a = screamer::PerAsset{0.0}, "slippage"_a = screamer::PerAsset{0.0},
             "slippage_unit"_a = "fraction", "tick_size"_a = screamer::PerAsset{0.0},
             "multiplier"_a = screamer::PerAsset{1.0},
             "fee_per_contract"_a = screamer::PerAsset{0.0},
             "size"_a = "weight", "initial_capital"_a = 1.0, "rebalance_band"_a = 0.0,
             "rebalance"_a = "drifted",
             "max_gross"_a = std::numeric_limits<double>::infinity(),
             "max_net"_a = std::numeric_limits<double>::infinity());

    nb::class_<screamer::BacktestPortfolioOHLCTarget, screamer::RowOp>(m, "BacktestPortfolioOHLCTarget")
        .def(nb::init<screamer::PerAsset, screamer::PerAsset, const std::string&,
                      screamer::PerAsset, screamer::PerAsset, screamer::PerAsset,
                      const std::string&, double, double, const std::string&,
                      double, double, const std::string&, int>(),
             "fee"_a = screamer::PerAsset{0.0}, "slippage"_a = screamer::PerAsset{0.0},
             "slippage_unit"_a = "fraction", "tick_size"_a = screamer::PerAsset{0.0},
             "multiplier"_a = screamer::PerAsset{1.0},
             "fee_per_contract"_a = screamer::PerAsset{0.0},
             "size"_a = "weight", "initial_capital"_a = 1.0, "rebalance_band"_a = 0.0,
             "rebalance"_a = "drifted",
             "max_gross"_a = std::numeric_limits<double>::infinity(),
             "max_net"_a = std::numeric_limits<double>::infinity(),
             "fill_at"_a = "open", "delay"_a = 1);

    nb::class_<screamer::RollingDownsideDeviation, screamer::ScreamerBase>(m, "RollingDownsideDeviation")
        .def(nb::init<int, double, const std::string&>(),
             "window_size"_a = 20, "mar"_a = 0.0,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingDownsideDeviation::reset, "Reset.");

    nb::class_<screamer::RollingOmega, screamer::ScreamerBase>(m, "RollingOmega")
        .def(nb::init<int, double>(),
             "window_size"_a = 20, "threshold"_a = 0.0)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingOmega::reset, "Reset.");

    nb::class_<screamer::RollingCVaR, screamer::ScreamerBase>(m, "RollingCVaR")
        .def(nb::init<int, double>(),
             "window_size"_a = 20, "alpha"_a = 0.05)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingCVaR::reset, "Reset.");
}