#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include <nanobind/nanobind.h>
#include "screamer/common/int_arg.h"
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>
#include "screamer/common/base.h"
#include "screamer/common/dispatch.h"
#include "screamer/common/eval_op.h"
#include "screamer/common/gc.h"
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
#include "screamer/backtest_summary.h"
#include "screamer/backtest_carry.h"
#include "screamer/validation/surrogates.h"
#include "screamer/validation/sharpe.h"
#include "screamer/validation/position_test.h"
#include "screamer/validation/strategy_test.h"
#include "screamer/portfolio_report.h"
#include "screamer/common/row_op.h"
#include "screamer/cross_section.h"
#include "screamer/backtest_portfolio_target.h"
#include "screamer/common/per_asset.h"
#include "screamer/rolling_downside_deviation.h"
#include "screamer/rolling_upside_deviation.h"
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
#include "screamer/triple_barrier.h"
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

// A per-asset parameter accepts a single number (one value for all assets) or
// a sequence of numbers (one per asset).
namespace nanobind { namespace detail {
template <> struct type_caster<screamer::PerAsset> {
    NB_TYPE_CASTER(screamer::PerAsset, const_name("float | Sequence[float]"))

    bool from_python(handle src, uint32_t flags, cleanup_list* cleanup) noexcept {
        if (!PySequence_Check(src.ptr())) {
            if (screamer::bool_kind(src.ptr())) return false;   // a bool is not a number setting
            const double d = PyFloat_AsDouble(src.ptr());
            if (d == -1.0 && PyErr_Occurred()) {
                PyErr_Clear();
                return false;
            }
            value.values = {d};
            return true;
        }
        if (screamer::bool_element_kind(src.ptr())) return false;   // as a single bool
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

// Reads a PortfolioReport argument (array, list, or tuple) as contiguous
// float64 values. A value that is not numeric raises the wrong-kind TypeError.
screamer::detail::ContigDouble portfolio_values(nb::object x, const char* what) {
    if (screamer::detail::is_unsupported_dtype_array(x))
        x = screamer::detail::coerce_to_f64(x);
    if (!screamer::detail::is_ndarray(x)) {
        if (!(nb::isinstance<nb::list>(x) || nb::isinstance<nb::tuple>(x)))
            throw nb::type_error((std::string("PortfolioReport ") + what +
                                  " must be arrays or sequences.").c_str());
        screamer::refuse_text(x, "PortfolioReport");   // a string is never a number
        nb::module_ np = nb::module_::import_("numpy");
        x = np.attr("asarray")(x, nb::arg("dtype") = np.attr("float64"));
    }
    return screamer::detail::read_contig_double(nb::cast<nb::ndarray<nb::ro>>(x));
}

std::string shape_text(const std::vector<std::size_t>& shape) {
    std::string s = "(";
    for (std::size_t i = 0; i < shape.size(); ++i)
        s += (i ? ", " : "") + std::to_string(shape[i]);
    return s + (shape.size() == 1 ? ",)" : ")");
}

// Checks that a price argument holds `rows` rows of `assets` prices: shape
// (assets,) for one event, (rows, assets) for a batch. A trailing size-1
// field axis is read away (rule 1).
void check_prices(screamer::detail::ContigDouble& prices, bool batch, std::size_t rows,
                  std::size_t assets) {
    const std::size_t nd = batch ? 2 : 1;
    if (prices.ndim() == nd + 1 && prices.shape.back() == 1) prices.shape.pop_back();
    std::vector<std::size_t> want;
    if (batch) want.push_back(rows);
    want.push_back(assets);
    if (prices.shape != want)
        throw nb::value_error(("PortfolioReport prices must have shape " + shape_text(want) +
                               ", one price per asset" + (batch ? " and time step" : "") +
                               ", got " + shape_text(prices.shape) + ".").c_str());
}

// One PortfolioReport event: a contiguous (assets, 4) row with columns
// [equity, pnl, position, cost], or a flat row of 4 * assets values, and
// optionally (assets,) prices.
nb::object portfolio_report_event(screamer::PortfolioReport& self,
                                  const screamer::detail::ContigDouble& values,
                                  screamer::detail::ContigDouble* prices) {
    std::size_t assets = 0;
    if (values.ndim() == 1) {
        if (values.shape[0] == 0 || values.shape[0] % 4 != 0)
            throw nb::value_error("PortfolioReport rows must have a multiple of 4 values.");
        assets = values.shape[0] / 4;
    } else if (values.ndim() == 2 && values.shape[1] == 4) {
        assets = values.shape[0];
    } else {
        throw nb::value_error("PortfolioReport rows must have shape (assets, 4).");
    }
    if (!prices) return nb::cast(self.call_assets(values.data.data(), assets));
    check_prices(*prices, false, 1, assets);
    return nb::cast(self.call_assets(values.data.data(), prices->data.data(), assets));
}

nb::object portfolio_report_row(screamer::PortfolioReport& self, nb::object row) {
    return portfolio_report_event(self, portfolio_values(std::move(row), "rows"), nullptr);
}

nb::object portfolio_report_batch(screamer::PortfolioReport& self,
                                  const screamer::detail::ContigDouble& values,
                                  screamer::detail::ContigDouble* prices = nullptr) {
    if (values.ndim() != 3 || values.shape[2] != 4)
        throw nb::value_error("PortfolioReport batch input must have shape (time, assets, 4).");
    const std::size_t time = values.shape[0];
    const std::size_t assets = values.shape[1];
    if (assets == 0)
        throw nb::value_error("PortfolioReport requires at least one asset.");
    if (prices) check_prices(*prices, true, time, assets);
    double* out = new double[time * 6 ? time * 6 : 1];
    self.reset();
    for (std::size_t t = 0; t < time; ++t) {
        const double* row = values.data.data() + t * assets * 4;
        auto result = prices
            ? self.call_assets(row, prices->data.data() + t * assets, assets)
            : self.call_assets(row, assets);
        screamer::detail::write_tuple_to_memory(out + t * 6, result);
    }
    self.reset();
    return screamer::detail::make_owned_array(out, {time, 6});
}

// The number of values in a list, tuple, or array, counted without NumPy;
// -1 for anything else. A nested sequence counts its leaves.
long long value_count(nb::handle h) {
    PyObject* p = h.ptr();
    if (PyFloat_Check(p) || PyLong_Check(p)) return -1;
    const bool list = PyList_Check(p), tuple = PyTuple_Check(p);
    if (!list && !tuple)
        return screamer::detail::is_ndarray(h) ? nb::cast<long long>(h.attr("size")) : -1;
    const Py_ssize_t n = list ? PyList_Size(h.ptr()) : PyTuple_Size(h.ptr());
    long long total = 0;
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* x = list ? PyList_GetItem(h.ptr(), i) : PyTuple_GetItem(h.ptr(), i);
        const long long c = value_count(x);
        total += c < 0 ? 1 : c;
    }
    return total;
}

// What a 2-item event of a one-source stream (or the first item of a list)
// holds. A (row, prices) pair has parts of 4 * K and K values; a row given as
// two asset records has two parts of 4 values. Anything else is malformed.
enum class PairKind { NotPair, Pair, Malformed };

PairKind pair_kind(nb::handle event) {
    if (!(PyList_Check(event.ptr()) || PyTuple_Check(event.ptr())) || nb::len(event) != 2)
        return PairKind::NotPair;
    nb::object e = nb::borrow<nb::object>(event);
    const long long rows = value_count(e[0]), prices = value_count(e[1]);
    if (rows < 0 || prices < 0) return PairKind::NotPair;   // a flat row; read as one
    if (prices >= 1 && rows == 4 * prices) return PairKind::Pair;
    if (rows == 4 && prices == 4) return PairKind::NotPair;  // two asset records
    return PairKind::Malformed;
}

// One PortfolioReport event per step, for the sync and async iterators alike:
// an (assets, 4) row from one source, a row and (assets,) prices from two
// sources, or (row, prices) pairs from one source. A stream is an independent
// run (decision D8): it runs on its own copy and resets the operator's running
// series when it opens.
class PortfolioReportStepper final : public screamer::EventStep {
public:
    PortfolioReportStepper(const nb::object& owner, std::size_t n_sources)
        : owner_(screamer::EventStepper::fresh_copy(owner)),
          op_(nb::cast<screamer::PortfolioReport&>(owner_)),
          n_sources_(n_sources) {
        nb::cast<screamer::PortfolioReport&>(owner).reset();
    }

    nb::object step(const screamer::EventItems& items) override {
        const std::size_t event = event_++;
        try {
            return step_event(items);
        } catch (nb::python_error& e) {
            if (screamer::is_conversion_error(e)) wrong_kind(items, event);
            throw;
        } catch (nb::builtin_exception& e) {
            if (screamer::is_conversion_error(e)) wrong_kind(items, event);
            throw;
        }
    }

    int traverse(visitproc visit, void* arg) const override {
        return screamer::gc_visit(owner_, visit, arg);
    }

private:
    nb::object step_event(const screamer::EventItems& items) {
        if (n_sources_ == 2) return priced(items[0], items[1]);
        nb::handle item = items[0];
        // Whether the stream carries prices is fixed in C++ by its first row.
        const PairKind kind = pair_kind(item);
        if (kind == PairKind::Malformed)
            throw nb::value_error(("PortfolioReport stream events are engine rows of shape (assets, 4) "
                "or (row, prices) pairs with 4 * K and K values; got two parts of " +
                std::to_string(value_count(nb::borrow<nb::object>(item)[0])) + " and " +
                std::to_string(value_count(nb::borrow<nb::object>(item)[1])) + " values.").c_str());
        if (kind == PairKind::Pair) {
            nb::object parts = nb::borrow<nb::object>(item);
            return priced(parts[0], parts[1]);
        }
        return portfolio_report_row(op_, nb::borrow<nb::object>(item));
    }

    nb::object priced(nb::handle row, nb::handle prices) {
        auto v = portfolio_values(nb::borrow<nb::object>(row), "rows");
        auto p = portfolio_values(nb::borrow<nb::object>(prices), "prices");
        return portfolio_report_event(op_, v, &p);
    }

    void wrong_kind(const screamer::EventItems& items, std::size_t event) const {
        screamer::throw_if_wrong_kind_event(screamer::op_display_name(owner_), items, event);
    }

    nb::object owner_;                    // the stream's own copy
    screamer::PortfolioReport& op_;
    std::size_t n_sources_;
    std::size_t event_ = 0;               // events seen, for error messages
};

// PortfolioReport()(out, prices): turnover is traded notional. Both inputs
// are arrays or lists, or both are iterators (sync or async) of rows.
nb::object portfolio_report_priced(screamer::PortfolioReport& self, nb::args args) {
    if (screamer::all_async(args))
        return nb::cast(screamer::LazyAsyncIterator(
            std::make_unique<PortfolioReportStepper>(nb::find(self), 2),
            {nb::borrow<nb::object>(args[0]), nb::borrow<nb::object>(args[1])}));
    const bool lazy0 = screamer::is_lazy_iterable(args[0]);
    const bool lazy1 = screamer::is_lazy_iterable(args[1]);
    if (lazy0 != lazy1)
        throw nb::type_error("PortfolioReport: iterator and array inputs cannot be mixed: pass "
                             "the engine rows and the prices both as iterators, or neither");
    if (lazy0)
        return nb::cast(screamer::LazyEvalIterator(
            std::make_unique<PortfolioReportStepper>(nb::find(self), 2),
            {nb::borrow<nb::object>(args[0]), nb::borrow<nb::object>(args[1])}));
    std::vector<screamer::detail::ContigDouble> in;
    const char* what[2] = {"rows", "prices"};
    for (std::size_t i = 0; i < 2; ++i) {
        nb::object a = nb::borrow<nb::object>(args[i]);
        try {
            in.push_back(portfolio_values(a, what[i]));
        } catch (nb::python_error& e) {
            if (screamer::is_conversion_error(e))
                screamer::throw_if_wrong_kind(screamer::op_display_name(nb::find(self)), a,
                                              screamer::where_text("input", i));
            throw;
        } catch (nb::builtin_exception& e) {
            if (screamer::is_conversion_error(e))
                screamer::throw_if_wrong_kind(screamer::op_display_name(nb::find(self)), a,
                                              screamer::where_text("input", i));
            throw;
        }
    }
    if (in[0].ndim() != 3) return portfolio_report_event(self, in[0], &in[1]);
    nb::object out = portfolio_report_batch(self, in[0], &in[1]);
    // Rule 3: lists of rows give a list of result rows; any array among the
    // inputs gives an array (decision A8).
    if (nb::isinstance<nb::list>(args[0]) && nb::isinstance<nb::list>(args[1]))
        return nb::list(out);
    return out;
}

// Runs one call of a row operator or PortfolioReport and puts the operator's
// name in front of a TypeError or ValueError that does not name it yet
// (a value-kind error from a shared reader, a complex dtype).
template <class S, class F>
nb::object named_call(S& self, F&& f) {
    try {
        return f();
    } catch (const nb::builtin_exception& e) {
        const std::string who = screamer::op_display_name(nb::find(self));
        const std::string m = e.what();
        if (m.find(who) != std::string::npos) throw;   // already names it
        if (e.type() == nb::exception_type::type_error) throw nb::type_error((who + ": " + m).c_str());
        if (e.type() == nb::exception_type::value_error) throw nb::value_error((who + ": " + m).c_str());
        throw;
    }
}

nb::object portfolio_report_call(screamer::PortfolioReport& self, nb::args raw) {
    nb::args args = screamer::normalize_args(raw, self);
    bool has_node = false;
    for (nb::handle arg : args)
        has_node = has_node || screamer::is_dag_node(nb::borrow<nb::object>(arg));
    if (has_node) {
        // DAG composition uses the scalar-compatible four-input path. The
        // multi-asset (time, assets, 4) reduction is a batch/stream contract.
        if (args.size() != 4)
            screamer::throw_input_count(screamer::op_display_name(nb::find(self)), 4, args.size());
        return screamer::make_dag_functor_node(nb::find(self), nb::cast<nb::tuple>(args));
    }
    if (args.size() == 2) return portfolio_report_priced(self, args);
    if (args.size() == 4) {
        // One event of the four portfolio columns, as JavaScript reads four
        // numbers (and a Pipeline node takes four inputs).
        screamer::PortfolioReport::InputArray in;
        for (std::size_t i = 0; i < 4; ++i)
            if (!screamer::read_double(args[i], in[i]))
                screamer::throw_not_a_number(screamer::op_display_name(nb::find(self)), args[i],
                                             screamer::where_text("input", static_cast<std::ptrdiff_t>(i)));
        return nb::cast(self.call(in));
    }
    if (args.size() != 1)
        screamer::throw_input_count(screamer::op_display_name(nb::find(self)), 4, args.size(),
                                    ", or the engine output and optional prices");
    nb::object input = nb::borrow<nb::object>(args[0]);
    if (screamer::is_async_iterable(input))
        return nb::cast(screamer::LazyAsyncIterator(
            std::make_unique<PortfolioReportStepper>(nb::find(self), 1), {input}));
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
            std::make_unique<PortfolioReportStepper>(nb::find(self), 1), {input}));
    }
    if (nb::isinstance<nb::list>(input) || nb::isinstance<nb::tuple>(input)) {
        // NumPy normalizes nested sequences, so a list can be either one row
        // (assets, 4) or a complete (time, assets, 4) batch.
        screamer::refuse_text(input, screamer::op_display_name(nb::find(self)));
        nb::module_ np = nb::module_::import_("numpy");
        nb::object arr;
        try {
            arr = np.attr("asarray")(input, nb::arg("dtype") = np.attr("float64"));
        } catch (nb::python_error& e) {
            if (screamer::is_conversion_error(e))
                screamer::throw_if_wrong_kind(screamer::op_display_name(nb::find(self)), input, "");
            if (nb::len(input) > 0 && pair_kind(nb::borrow<nb::object>(input)[0]) == PairKind::Pair)
                throw nb::type_error(
                    "PortfolioReport does not read a list of (row, prices) pairs; pass the rows "
                    "and the prices as two arguments, PortfolioReport()(out, prices), or stream "
                    "the pairs, PortfolioReport()(zip(out, prices)).");
            throw;
        }
        auto values = screamer::detail::read_contig_double(nb::cast<nb::ndarray<nb::ro>>(arr));
        if (values.ndim() != 3) return portfolio_report_row(self, arr);
        // Rule 3: a list of rows gives a list of result rows. A tuple keeps
        // the 2.8.0 array (decision D9).
        nb::object out = portfolio_report_batch(self, values);
        if (nb::isinstance<nb::list>(input)) return nb::list(out);
        return out;
    }
    throw nb::type_error(("PortfolioReport input must be a numpy array or iterable of rows, not a " +
                          screamer::type_short_name(input.type())).c_str());
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
        screamer::refuse_text(x);                      // a string is never a number
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

// A packed (T, K, F) array, F = fields_in: T events, one call_rows over its
// rows. Returns (T, K) or (T, K, fields_out), as the separate inputs do.
nb::object row_op_packed(screamer::RowOp& self, nb::object x) {
    const screamer::detail::ContigDouble p =
        screamer::detail::read_double_view(nb::cast<nb::ndarray<nb::ro>>(x));
    const std::size_t fi = self.fields_in(), fo = self.fields_out();
    if (p.shape[2] != fi)
        throw nb::value_error(("a packed input of " + screamer::op_display_name(nb::find(self)) +
                               " has shape (T, K, " +
                               std::to_string(fi) + "), got (" + std::to_string(p.shape[0]) + ", " +
                               std::to_string(p.shape[1]) + ", " + std::to_string(p.shape[2]) +
                               ").").c_str());
    const std::size_t time = p.shape[0], k = p.shape[1];
    if (k == 0) throw nb::value_error("a row operator needs at least one asset.");
    const std::size_t n_out = time * k * fo;
    double* out = new double[n_out ? n_out : 1];
    self.reset();
    try {
        self.call_rows(p.values(), out, time, k, /*packed=*/true);
    } catch (...) {
        delete[] out;
        self.reset();
        throw;
    }
    self.reset();
    return screamer::detail::make_owned_array(out, row_op_shape(time, k, fo, true));
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

    nb::object step(const screamer::EventItems& items) override {
        const std::size_t event = event_++;
        try {
            return step_event(items);
        } catch (nb::python_error& e) {
            if (screamer::is_conversion_error(e)) wrong_kind(items, event);
            throw;
        } catch (nb::builtin_exception& e) {
            if (screamer::is_conversion_error(e)) wrong_kind(items, event);
            throw;
        }
    }

    int traverse(visitproc visit, void* arg) const override {
        return screamer::gc_visit(owner_, visit, arg);
    }

private:
    void wrong_kind(const screamer::EventItems& items, std::size_t event) const {
        screamer::throw_if_wrong_kind_event(screamer::op_display_name(owner_), items, event);
    }

    nb::object step_event(const screamer::EventItems& items) {
        fields_.clear();
        if (unpack_) {
            nb::handle event = items[0];
            if (!(nb::isinstance<nb::tuple>(event) || nb::isinstance<nb::list>(event)) ||
                nb::len(event) != op_.fields_in())
                throw nb::value_error(("each event of one row stream is a tuple of " +
                    std::to_string(op_.fields_in()) + " rows of K values.").c_str());
            for (nb::handle row : event) add(nb::borrow<nb::object>(row));
        } else {
            for (std::size_t i = 0; i < items.size(); ++i) add(nb::borrow<nb::object>(items[i]));
        }
        return row_op_run(op_, fields_);
    }

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
    std::size_t event_ = 0;               // events seen, for error messages
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
    // One packed (T, K, F) array holds the fi inputs, F last (rule 4). Its
    // rows go to the engine as they are (RowOp::call_rows reads each asset's
    // fields), read in place when the array is float64 and C-contiguous.
    if (n == 1 && fi > 1 && !screamer::is_lazy_iterable(args[0])) {
        nb::object a = nb::borrow<nb::object>(args[0]);
        if (screamer::detail::is_unsupported_dtype_array(a)) a = screamer::detail::coerce_to_f64(a);
        if (screamer::detail::is_ndarray(a) && nb::cast<int>(a.attr("ndim")) == 3)
            return row_op_packed(self, a);
    }
    if (n != fi)
        screamer::throw_input_count(screamer::op_display_name(nb::find(self)), fi, n,
            fi > 1 ? ", or one packed (T, K, " + std::to_string(fi) + ") array" : "");
    std::vector<screamer::detail::ContigDouble> fields;
    for (std::size_t i = 0; i < n; ++i) {
        nb::handle a = args[i];
        try {
            fields.push_back(row_op_input(nb::borrow<nb::object>(a)));
        } catch (nb::python_error& e) {
            if (screamer::is_conversion_error(e))
                screamer::throw_if_wrong_kind(screamer::op_display_name(nb::find(self)), a,
                                              n > 1 ? screamer::where_text("input", i) : "");
            throw;
        }
        // Rule 1: (T, K, 1) reads as (T, K); the data is already in that order.
        fields.back().shape = screamer::shape::consume_unit_field(fields.back().shape);
    }
    nb::object result = row_op_run(self, fields);
    // Rule 3: a list of rows gives a list of result rows. A tuple keeps the
    // 2.8.0 meaning (rows over time, an ndarray, decision D9), and one row
    // (a rank-1 input) stays one array. A mix of lists and arrays also stays an
    // array; position_test and walk_forward rely on that.
    bool listed = true;  // every input a list; a mix with arrays stays an array
    for (nb::handle a : args) listed = listed && nb::isinstance<nb::list>(a);
    if (listed && fields[0].ndim() == 2) return nb::list(result);
    return result;
}

} // namespace

namespace {

// Batch-only surrogate generators for randomization tests (see
// screamer/validation.py). Each returns a new (T, ncols) float64 array holding
// columns col0 .. col0 + ncols - 1 of the surrogate set.
using SurrogateIn = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

nb::object circular_shift_batch(SurrogateIn x, long long min_shift,
                                std::int64_t seed_arg, size_t col0, size_t ncols,
                                long long start) {
    const std::uint64_t seed = screamer::validation::check_seed(seed_arg);
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

nb::object segment_shuffle_batch(SurrogateIn x, std::int64_t seed_arg,
                                 size_t col0, size_t ncols) {
    const std::uint64_t seed = screamer::validation::check_seed(seed_arg);
    const size_t T = x.shape(0);
    double* out = new double[T * ncols];
    screamer::validation::segment_shuffle(x.data(), T, seed, col0, ncols, out);
    return screamer::detail::make_owned_array(out, {T, ncols});
}

// The permutation kernels also take (T, K) blocks of K assets: each copy
// applies one time permutation to every asset. A 1-D input gives (T, ncols)
// outputs as before; a (T, K) input gives (T, ncols, K) outputs. The source
// rows are (T, ncols) either way.
using BlockIn = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;

struct BlockShape { size_t T, K; bool panel; };

BlockShape block_shape(const BlockIn& x, const char* what) {
    if (x.ndim() == 1) return {x.shape(0), 1, false};
    if (x.ndim() == 2) return {x.shape(0), x.shape(1), true};
    throw std::invalid_argument(std::string(what) + " must be 1-D (T,) or 2-D (T, K).");
}

std::vector<size_t> block_out_shape(const BlockShape& b, size_t ncols) {
    if (b.panel) return {b.T, ncols, b.K};
    return {b.T, ncols};
}

// The volume block must match the prices: the same T, and the same K.
bool check_volume(const BlockIn& volume, const BlockShape& b) {
    if (volume.ndim() == 1 && volume.shape(0) == 0) return false;
    const BlockShape v = block_shape(volume, "volume");
    if (v.T != b.T)
        throw std::invalid_argument("volume must have the same length as the prices.");
    if (v.panel != b.panel || v.K != b.K)
        throw std::invalid_argument("volume must have the same shape as the prices.");
    return true;
}

nb::object permute_bars_batch(BlockIn o, BlockIn h, BlockIn l, BlockIn c,
                              const std::string& mode, std::int64_t seed_arg,
                              size_t col0, size_t ncols,
                              BlockIn volume, bool notional) {
    const std::uint64_t seed = screamer::validation::check_seed(seed_arg);
    const BlockShape b = block_shape(o, "open");
    for (const BlockIn* a : {&h, &l, &c}) {
        const BlockShape q = block_shape(*a, "high, low, and close");
        if (q.T != b.T)
            throw std::invalid_argument("open, high, low, and close must have the same length.");
        if (q.panel != b.panel || q.K != b.K)
            throw std::invalid_argument("open, high, low, and close must have the same shape.");
    }
    const bool has_vol = check_volume(volume, b);
    const auto m = screamer::validation::parse_permute_mode(mode);
    const size_t N = b.T * ncols * b.K;
    std::vector<double*> bufs;
    for (int k = 0; k < (has_vol ? 6 : 5); ++k)
        bufs.push_back(new double[k == 4 ? b.T * ncols : N]);
    try {
        screamer::validation::permute_bars(
            o.data(), h.data(), l.data(), c.data(), b.T, b.K, m, seed, col0, ncols,
            bufs[0], bufs[1], bufs[2], bufs[3], bufs[4],
            has_vol ? volume.data() : nullptr, notional, has_vol ? bufs[5] : nullptr);
    } catch (...) {
        for (double* p : bufs) delete[] p;
        throw;
    }
    using screamer::detail::make_owned_array;
    const auto shape = block_out_shape(b, ncols);
    nb::list out;
    for (size_t k = 0; k < bufs.size(); ++k)
        out.append(make_owned_array(bufs[k], k == 4 ? std::vector<size_t>{b.T, ncols} : shape));
    return nb::tuple(out);
}

nb::object permute_returns_batch(BlockIn x, const std::string& mode, std::int64_t seed_arg,
                                 size_t col0, size_t ncols,
                                 BlockIn volume, bool notional) {
    const std::uint64_t seed = screamer::validation::check_seed(seed_arg);
    const BlockShape b = block_shape(x, "x");
    const bool has_vol = check_volume(volume, b);
    const auto m = screamer::validation::parse_permute_mode(mode);
    const size_t N = b.T * ncols * b.K;
    std::vector<double*> bufs;
    for (int k = 0; k < (has_vol ? 3 : 2); ++k)
        bufs.push_back(new double[k == 1 ? b.T * ncols : N]);
    try {
        screamer::validation::permute_returns(
            x.data(), b.T, b.K, m, seed, col0, ncols, bufs[0], bufs[1],
            has_vol ? volume.data() : nullptr, notional, has_vol ? bufs[2] : nullptr);
    } catch (...) {
        for (double* p : bufs) delete[] p;
        throw;
    }
    using screamer::detail::make_owned_array;
    const auto shape = block_out_shape(b, ncols);
    nb::list out;
    for (size_t k = 0; k < bufs.size(); ++k)
        out.append(make_owned_array(bufs[k], k == 1 ? std::vector<size_t>{b.T, ncols} : shape));
    return nb::tuple(out);
}

// backtest_report (screamer/backtest.py): an engine's (T, 4) output, any
// strides, as (report (T, 6), stats (T, 9), summary tuple). The shape and
// periods_per_year are checked first, so a bad call allocates nothing.
using EngineRows = nb::ndarray<const double, nb::device::cpu>;

nb::object backtest_report_py(EngineRows v, nb::handle ppy) {
    screamer::check_engine_rows(v.ndim(), v.ndim() == 2 ? v.shape(1) : 0);
    double periods_per_year;
    if (screamer::bool_kind(ppy.ptr()))
        throw nb::type_error("backtest_report: periods_per_year must be a number, got bool");
    if (!nb::try_cast<double>(ppy, periods_per_year))
        screamer::throw_wrong_kind("backtest_report", "periods_per_year", ppy);
    screamer::BacktestReport rep(periods_per_year);
    screamer::BacktestStats st(periods_per_year);
    const std::size_t T = v.shape(0);
    constexpr std::size_t nr = screamer::BacktestReport::column_names.size();
    constexpr std::size_t ns = screamer::BacktestStats::column_names.size();
    auto report = screamer::detail::new_output(T * nr);
    auto stats = screamer::detail::new_output(T * ns);
    double summary[screamer::kNumStats];
    screamer::backtest_report(rep, st, v.data(), T, static_cast<std::ptrdiff_t>(v.stride(0)),
                              static_cast<std::ptrdiff_t>(v.stride(1)), report.get(), stats.get(),
                              summary);
    nb::list sum;
    for (double x : summary) sum.append(x);
    using screamer::detail::make_owned_array;
    return nb::make_tuple(make_owned_array(std::move(report), {T, nr}),
                          make_owned_array(std::move(stats), {T, ns}), nb::tuple(sum));
}

// The names of backtest_report: (engine columns, BacktestReport columns,
// BacktestStats columns, summary).
nb::object backtest_report_names() {
    const auto names = [](const auto& a) {
        nb::list out;
        for (const char* q : a) out.append(q);
        return nb::tuple(out);
    };
    nb::list summary;
    for (const auto& st : screamer::kStats) summary.append(st.name);
    return nb::make_tuple(names(screamer::kEngineColumns),
                          names(screamer::BacktestReport::column_names),
                          names(screamer::BacktestStats::column_names), nb::tuple(summary));
}

namespace sv = screamer::validation;

// The result arrays of the randomization tests (sv::ResultSink): n * K values
// per statistic and for the null, filled by the driver's batches and handed
// to Python as they are, so the result is held once.
class PyResults {
public:
    void begin(std::size_t n, std::size_t K) {
        K_ = K;
        const std::size_t size = n * K;
        null_.reset(new double[size ? size : 1]);
        for (auto& q : stats_) q.reset(new double[size ? size : 1]);
    }

    void emit(std::size_t j0, std::size_t runs, const double* stats, std::size_t stride,
              const double* metrics, std::size_t count) {
        for (std::size_t q = 0; q < sv::kNumStats; ++q)
            std::copy(stats + q * stride, stats + q * stride + runs * K_, stats_[q].get() + j0 * K_);
        std::copy(metrics, metrics + count, null_.get() + null_count_);
        null_count_ += count;
    }

    // The null distribution: its first values, or (n, K).
    nb::object take_null(const std::vector<size_t>& shape) {
        return screamer::detail::make_owned_array(std::move(null_), shape);
    }

    nb::object take_stats(std::size_t q, const std::vector<size_t>& shape) {
        return screamer::detail::make_owned_array(std::move(stats_[q]), shape);
    }

private:
    std::size_t K_ = 1, null_count_ = 0;
    std::unique_ptr<double[]> null_;
    std::array<std::unique_ptr<double[]>, sv::kNumStats> stats_;
};

// A custom metric's result, as the Python thunk classifies it: (kind,
// float64 values); see _metric_result in screamer/validation.py. The kind
// and the count are checked by the drivers' shared rule, then `count`
// values are copied.
std::size_t metric_values(const nb::object& r, double* values, std::size_t count) {
    using Values = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
    const nb::tuple t = nb::cast<nb::tuple>(r);
    const auto kind = static_cast<sv::MetricKind>(nb::cast<int>(t[0]));
    const auto v = nb::cast<Values>(t[1]);
    sv::check_metric_result(kind, count, v.shape(0));
    std::copy(v.data(), v.data() + count, values);
    return count;
}

// strategy_test (screamer/validation.py): the driver is
// screamer::validation::strategy_test; this host gives it the Python strategy.
// `run` is called with the inputs as positional arguments and returns a
// float64 C-contiguous engine output; `metric` (or None) maps that output to
// a 1-D float64 array. Both are only used during the call, under the GIL; a
// Python exception they raise propagates unchanged.

class PyStrategyHost final : public sv::StrategyHost {
public:
    PyStrategyHost(nb::handle run, nb::handle metric, const nb::list& inputs,
                   const std::vector<sv::StrategyInput>& data)
        : run_(run), metric_(metric), inputs_(inputs), data_(data), pending_(data.size()) {}

    double* moved_input(std::size_t i, std::size_t size) override {
        pending_[i].reset(new double[size ? size : 1]);   // a new array for every run
        return pending_[i].get();
    }

    sv::EngineOutput run(const std::vector<unsigned char>& moved) override {
        const std::size_t k = data_.size();
        nb::object args = nb::steal(PyTuple_New(static_cast<Py_ssize_t>(k)));
        if (!args.is_valid()) throw nb::python_error();
        for (std::size_t i = 0; i < k; ++i) {
            nb::object a;
            if (moved[i]) {
                const sv::StrategyInput& x = data_[i];
                a = screamer::detail::make_owned_array(
                    std::move(pending_[i]),
                    x.panel ? std::vector<size_t>{x.T, x.K} : std::vector<size_t>{x.T});
            } else {
                a = inputs_[i];
            }
            if (PyTuple_SetItem(args.ptr(), static_cast<Py_ssize_t>(i), a.release().ptr()) != 0)
                throw nb::python_error();             // the item reference is stolen either way
        }
        nb::object r = nb::steal(PyObject_Call(run_.ptr(), args.ptr(), nullptr));
        if (!r.is_valid()) throw nb::python_error();
        last_ = nb::cast<OutArray>(r);
        last_obj_ = std::move(r);
        sv::EngineOutput out;
        out.data = last_.data();
        out.ndim = last_.ndim();
        for (std::size_t d = 0; d < out.ndim && d < 3; ++d) out.shape[d] = last_.shape(d);
        if (out.ndim > 3) out.ndim = 4;   // rejected by the driver
        return out;
    }

    std::size_t metric(double* values, std::size_t count) override {
        return metric_values(metric_(last_obj_), values, count);
    }

    void begin(std::size_t n, std::size_t K) override { results.begin(n, K); }

    void emit(std::size_t j0, std::size_t runs, const double* stats, std::size_t stride,
              const double* metrics, std::size_t count) override {
        results.emit(j0, runs, stats, stride, metrics, count);
    }

    PyResults results;

private:
    using OutArray = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
    nb::handle run_, metric_;
    const nb::list& inputs_;
    const std::vector<sv::StrategyInput>& data_;
    std::vector<std::unique_ptr<double[]>> pending_;
    nb::object last_obj_;
    OutArray last_;
};

// position_test and walk_forward (screamer/validation.py): the drivers are
// screamer::validation::position_test and walk_forward; this host gives them
// the Python engine. `run(target)` resets the engine, runs it on the target
// and the market inputs, and returns a float64 C-contiguous output; `reset()`
// resets it; `metric` (or None) maps an output of shape (T, k, 4) to a
// float64 array of values. They are only used during the call, under the GIL;
// a Python exception they raise propagates unchanged. Targets and windows the
// driver fills become new arrays, so an engine or a metric may keep them.
class PyEngineHost final : public sv::PositionHost {
public:
    PyEngineHost(nb::handle run, nb::handle reset, nb::handle metric, nb::handle original)
        : run_(run), reset_(reset), metric_(metric), original_(original) {}

    double* target_buffer(std::size_t size) override {
        target_buf_.reset(new double[size ? size : 1]);
        return target_buf_.get();
    }

    sv::EngineOutput run(bool original, std::size_t T, std::size_t cols) override {
        nb::object arg = original ? nb::borrow<nb::object>(original_)
            : screamer::detail::make_owned_array(
                  std::move(target_buf_), cols ? std::vector<size_t>{T, cols} : std::vector<size_t>{T});
        nb::object r = run_(arg);
        last_ = nb::cast<OutArray>(r);
        last_obj_ = std::move(r);
        last_target_ = std::move(arg);
        sv::EngineOutput out;
        out.data = last_.data();
        out.ndim = last_.ndim();
        for (std::size_t d = 0; d < out.ndim && d < 3; ++d) out.shape[d] = last_.shape(d);
        if (out.ndim > 3) out.ndim = 4;   // rejected by the driver
        return out;
    }

    void reset() override { reset_(); }

    double* window_buffer(std::size_t size) override {
        window_buf_.reset(new double[size ? size : 1]);
        return window_buf_.get();
    }

    std::size_t metric(sv::MetricOf what, double* values, std::size_t count,
                       std::size_t rows) override {
        nb::object arg = what == sv::MetricOf::Window
            ? screamer::detail::make_owned_array(std::move(window_buf_), {rows, count, 4})
            : last_obj_;
        return metric_values(metric_(arg), values, count);
    }

    void begin(std::size_t n, std::size_t K) override { results.begin(n, K); }

    void emit(std::size_t j0, std::size_t runs, const double* stats, std::size_t stride,
              const double* metrics, std::size_t count) override {
        results.emit(j0, runs, stats, stride, metrics, count);
    }

    PyResults results;

    nb::object last_output() const { return last_obj_; }
    nb::object last_target() const { return last_target_; }

private:
    using OutArray = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
    nb::handle run_, reset_, metric_, original_;
    std::unique_ptr<double[]> target_buf_, window_buf_;
    nb::object last_obj_, last_target_;
    OutArray last_;
};

nb::tuple strategy_test_py(nb::callable run, nb::object metric, nb::list inputs,
                           std::int64_t n, const std::string& method,
                           std::vector<std::int64_t> ohlc, std::int64_t price,
                           const std::string& mode, std::optional<std::int64_t> volume,
                           const std::string& volume_mode, std::vector<std::int64_t> columns,
                           std::vector<std::int64_t> fixed, const std::string& metric_name,
                           bool custom, std::optional<bool> larger_is_better,
                           std::optional<std::int64_t> min_shift, std::int64_t seed,
                           double periods_per_year) {
    std::vector<BlockIn> blocks;
    std::vector<sv::StrategyInput> data;
    blocks.reserve(inputs.size());
    for (nb::handle x : inputs) {
        blocks.push_back(nb::cast<BlockIn>(x));
        const BlockIn& b = blocks.back();
        if (b.ndim() == 1) data.push_back({b.data(), b.shape(0), 1, false});
        else if (b.ndim() == 2) data.push_back({b.data(), b.shape(0), b.shape(1), true});
        else throw std::invalid_argument("input must be 1-D (T,) or 2-D (T, K)");
    }
    sv::StrategyTestOptions opt;
    opt.n = n; opt.method = method; opt.ohlc = std::move(ohlc); opt.price = price;
    opt.mode = mode; opt.volume = volume; opt.volume_mode = volume_mode;
    opt.columns = std::move(columns); opt.fixed = std::move(fixed);
    opt.metric = metric_name; opt.custom = custom; opt.larger_is_better = larger_is_better;
    opt.min_shift = min_shift; opt.seed = seed; opt.periods_per_year = periods_per_year;

    PyStrategyHost host(run, metric, inputs, data);
    sv::StrategyTestResult r = sv::strategy_test(host, data, opt);

    const std::size_t nn = static_cast<std::size_t>(n);
    nb::dict observed_stats, null_stats;
    for (std::size_t q = 0; q < sv::kNumStats; ++q) {
        observed_stats[sv::kStats[q].name] = r.observed_stats[q];
        null_stats[sv::kStats[q].name] = host.results.take_stats(q, {nn});
    }
    return nb::make_tuple(r.observed, host.results.take_null({r.null_count}), r.p_value,
                          observed_stats, null_stats);
}

// The shape of an array for the drivers: (T,) or (T, K), or another ndim
// that they reject.
sv::ArrayShape array_shape(const BlockIn& x) {
    sv::ArrayShape a;
    a.data = x.data();
    a.ndim = x.ndim();
    a.T = x.ndim() >= 1 ? x.shape(0) : 1;
    a.K = x.ndim() >= 2 ? x.shape(1) : 1;
    return a;
}

std::vector<sv::ArrayShape> array_shapes(const nb::list& xs) {
    std::vector<sv::ArrayShape> out;
    out.reserve(xs.size());
    for (nb::handle x : xs) out.push_back(array_shape(nb::cast<BlockIn>(x)));
    return out;
}

nb::dict stats_dict(const std::vector<double>& stats, std::size_t K, bool family) {
    nb::dict d;
    for (std::size_t q = 0; q < sv::kNumStats; ++q) {
        if (!family) {
            d[sv::kStats[q].name] = stats[q];
            continue;
        }
        double* v = new double[K ? K : 1];
        std::copy(stats.begin() + q * K, stats.begin() + (q + 1) * K, v);
        d[sv::kStats[q].name] = screamer::detail::make_owned_array(v, {K});
    }
    return d;
}

nb::object double_array(const std::vector<double>& x) {
    double* v = new double[x.size() ? x.size() : 1];
    std::copy(x.begin(), x.end(), v);
    return screamer::detail::make_owned_array(v, {x.size()});
}

// (observed, null, p_value, family_p_value, observed_stats, null_stats,
// min_shift): floats for one strategy or one book, arrays for K strategies.
nb::tuple position_test_py(nb::callable run, nb::callable reset, nb::object metric,
                           nb::object original, BlockIn target, nb::list market, std::int64_t n,
                           const std::string& method, const std::string& metric_name, bool custom,
                           std::optional<bool> larger_is_better,
                           std::optional<std::int64_t> min_shift, std::int64_t seed,
                           double periods_per_year, bool portfolio) {
    sv::PositionTestOptions opt;
    opt.n = n; opt.method = method; opt.metric = metric_name; opt.custom = custom;
    opt.larger_is_better = larger_is_better; opt.min_shift = min_shift; opt.seed = seed;
    opt.periods_per_year = periods_per_year; opt.portfolio = portfolio;
    PyEngineHost host(run, reset, metric, original);
    const sv::PositionTestResult r = sv::position_test(host, array_shape(target), array_shapes(market), opt);

    const std::size_t nn = static_cast<std::size_t>(n), K = r.K;
    nb::dict null_stats;
    for (std::size_t q = 0; q < sv::kNumStats; ++q)
        null_stats[sv::kStats[q].name] = host.results.take_stats(
            q, r.family ? std::vector<size_t>{nn, K} : std::vector<size_t>{nn});
    nb::object min = r.min_shift ? nb::cast(*r.min_shift) : nb::none();
    if (!r.family)
        return nb::make_tuple(r.observed[0], host.results.take_null({r.null_count}), r.p_value[0],
                              nb::none(), stats_dict(r.observed_stats, 1, false), null_stats, min);
    return nb::make_tuple(double_array(r.observed), host.results.take_null({nn, K}),
                          double_array(r.p_value), double_array(r.family_p_value),
                          stats_dict(r.observed_stats, K, true), null_stats, min);
}

nb::object int64_array(const std::vector<std::int64_t>& x) {
    std::int64_t* v = new std::int64_t[x.size() ? x.size() : 1];
    std::copy(x.begin(), x.end(), v);
    nb::capsule owner(v, [](void* p) noexcept { delete[] static_cast<std::int64_t*>(p); });
    const size_t shape[1] = {x.size()};
    return nb::cast(nb::ndarray<nb::numpy, std::int64_t>(v, 1, shape, owner));
}

// (target, choice, fold_start, train_scores, output, stats).
nb::tuple walk_forward_py(nb::callable run, nb::callable reset, nb::object metric,
                          nb::object original, BlockIn targets, nb::list prices, double train, double test,
                          const std::string& metric_name, bool custom,
                          std::optional<bool> larger_is_better, bool anchored,
                          double periods_per_year, bool portfolio) {
    sv::WalkForwardOptions opt;
    opt.train = train; opt.test = test; opt.metric = metric_name; opt.custom = custom;
    opt.larger_is_better = larger_is_better; opt.anchored = anchored;
    opt.periods_per_year = periods_per_year; opt.portfolio = portfolio;
    PyEngineHost host(run, reset, metric, original);
    sv::WalkForwardResult r = sv::walk_forward(host, array_shape(targets), array_shapes(prices), opt);
    nb::dict stats;
    for (std::size_t q = 0; q < sv::kNumStats; ++q) stats[sv::kStats[q].name] = r.stats[q];
    return nb::make_tuple(host.last_target(), int64_array(r.choice), int64_array(r.fold_start),
                          screamer::detail::make_owned_array(std::move(r.scores), {r.folds, r.K}),
                          host.last_output(), stats);
}

}  // namespace

void init_bindings_fin(nb::module_& m) {
    // A custom metric's result of the wrong kind is a TypeError, as in
    // JavaScript.
    nb::register_exception_translator([](const std::exception_ptr& p, void*) {
        try {
            std::rethrow_exception(p);
        } catch (const screamer::validation::MetricTypeError& e) {
            PyErr_SetString(PyExc_TypeError, e.what());
        }
    });
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
    // The statistic table of the driver: (name, source, column, larger),
    // larger None for a statistic with no natural direction.
    m.def("_strategy_stats", []() {
        nb::list out;
        for (const auto& st : sv::kStats)
            out.append(nb::make_tuple(st.name, sv::stat_source_name(st.source), st.column,
                                      st.larger < 0 ? nb::object(nb::none())
                                                     : nb::object(nb::bool_(st.larger == 1))));
        return out;
    });
    m.def("_backtest_report", &backtest_report_py, "values"_a, "periods_per_year"_a.none());
    m.def("_backtest_report_names", &backtest_report_names);
    m.def("_strategy_test", &strategy_test_py,
          "run"_a, "metric"_a.none(), "inputs"_a, "n"_a, "method"_a, "ohlc"_a, "price"_a,
          "mode"_a, "volume"_a.none(), "volume_mode"_a, "columns"_a, "fixed"_a,
          "metric_name"_a, "custom"_a, "larger_is_better"_a.none(), "min_shift"_a.none(),
          "seed"_a, "periods_per_year"_a);
    m.def("_position_test", &position_test_py,
          "run"_a, "reset"_a, "metric"_a.none(), "original"_a, "target"_a, "market"_a, "n"_a,
          "method"_a, "metric_name"_a, "custom"_a, "larger_is_better"_a.none(),
          "min_shift"_a.none(), "seed"_a, "periods_per_year"_a, "portfolio"_a);
    m.def("_walk_forward", &walk_forward_py,
          "run"_a, "reset"_a, "metric"_a.none(), "original"_a, "targets"_a, "prices"_a, "train"_a,
          "test"_a, "metric_name"_a, "custom"_a, "larger_is_better"_a.none(), "anchored"_a,
          "periods_per_year"_a, "portfolio"_a);
    m.def("_default_min_shift", [](SurrogateIn x) {
        return screamer::validation::default_min_shift(x.data(), x.shape(0));
    }, "x"_a);


    nb::class_<screamer::Return, screamer::ScreamerBase>(m, "Return")
        .def(nb::init<screamer::IntArg<int>>(), "window_size"_a = 1)
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::Return::reset, "Reset to the initial state.");

    nb::class_<screamer::LogReturn, screamer::ScreamerBase>(m, "LogReturn")
        .def(nb::init<screamer::IntArg<int>>(), "window_size"_a = 1)
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::LogReturn::reset, "Reset to the initial state.");

    // ROC family: rate-of-change variants. TA-Lib has all three as
    // separate functions; we provide them under TA-Lib's names so
    // users can port directly. ROCP is mathematically identical to
    // Return.
    nb::class_<screamer::ROC, screamer::ScreamerBase>(m, "ROC")
        // TA-Lib's timeperiod default is 10, which is what the docs page and
        // the sibling Momentum both declare. This was 1.
        .def(nb::init<screamer::IntArg<int>>(), "window_size"_a = 10)
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::ROC::reset, "Reset to the initial state.");

    nb::class_<screamer::ROCP, screamer::ScreamerBase>(m, "ROCP")
        // TA-Lib's timeperiod default is 10, which is what the docs page and
        // the sibling Momentum both declare. This was 1.
        .def(nb::init<screamer::IntArg<int>>(), "window_size"_a = 10)
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::ROCP::reset, "Reset to the initial state.");

    nb::class_<screamer::ROCR, screamer::ScreamerBase>(m, "ROCR")
        // TA-Lib's timeperiod default is 10, which is what the docs page and
        // the sibling Momentum both declare. This was 1.
        .def(nb::init<screamer::IntArg<int>>(), "window_size"_a = 10)
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::ROCR::reset, "Reset to the initial state.");

    // RollingCorr: 2 inputs (x, y), 1 output (Pearson correlation).
    // Inherits from FunctorBase<_, 2, 1>, NOT ScreamerBase -- the
    // multi-input class hierarchy is separate. handle_input dispatches
    // on the variadic args (scalars / N parallel arrays / list of N-tuples
    // / N parallel iterables).
    nb::class_<screamer::RollingCorr, screamer::EvalOp>(m, "RollingCorr")
        .def(nb::init<screamer::IntArg<int>, const std::string&>(),
             "window_size"_a = 20,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::functor_call<screamer::RollingCorr>)
        .def("reset", &screamer::RollingCorr::reset, "Reset to the initial state.");

    // Rolling sample covariance of two streams.
    nb::class_<screamer::RollingCov, screamer::EvalOp>(m, "RollingCov")
        .def(nb::init<screamer::IntArg<int>, const std::string&>(),
             "window_size"_a = 20,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::functor_call<screamer::RollingCov>)
        .def("reset", &screamer::RollingCov::reset, "Reset to the initial state.");

    // Rolling regression slope of x on y: beta = cov(x, y) / var(y).
    nb::class_<screamer::RollingTLSBeta, screamer::EvalOp>(m, "RollingTLSBeta")
        .def(nb::init<screamer::IntArg<int>, const std::string&>(),
             "window_size"_a = 20,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::functor_call<screamer::RollingTLSBeta>)
        .def("reset", &screamer::RollingTLSBeta::reset, "Reset to the initial state.");

    nb::class_<screamer::KalmanRegression, screamer::EvalOp>(m, "KalmanRegression")
        .def(nb::init<screamer::FloatArg, screamer::FloatArg, screamer::FloatArg>(),
             "delta"_a = 1e-6, "obs_var"_a = 1e-3, "prior_var"_a = 1.0)
        .def("__call__", &screamer::functor_call<screamer::KalmanRegression>)
        .def("reset", &screamer::KalmanRegression::reset, "Reset to the diffuse prior.");

    nb::class_<screamer::RollingADF, screamer::ScreamerBase>(m, "RollingADF")
        .def(nb::init<screamer::IntArg<int>, screamer::IntArg<int>>(), "window_size"_a = 100, "lags"_a = 1)
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::RollingADF::reset, "Reset to the initial state.");

    nb::class_<screamer::BandSignal, screamer::ScreamerBase>(m, "BandSignal")
        .def(nb::init<screamer::FloatArg, screamer::FloatArg, screamer::FloatArg>(),
             "entry"_a = 2.0, "exit"_a = 0.0,
             "stop"_a = std::numeric_limits<double>::infinity())
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::BandSignal::reset, "Reset to flat.");

    nb::class_<screamer::RollingBeta, screamer::EvalOp>(m, "RollingBeta")
        .def(nb::init<screamer::IntArg<int>, const std::string&>(),
             "window_size"_a = 20,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::functor_call<screamer::RollingBeta>)
        .def("reset", &screamer::RollingBeta::reset, "Reset to the initial state.");

    // Hedge-adjusted residual of x against y: spread = x - beta * y, with
    // beta computed exactly as in RollingBeta.
    nb::class_<screamer::RollingSpread, screamer::EvalOp>(m, "RollingSpread")
        .def(nb::init<screamer::IntArg<int>, const std::string&>(),
             "window_size"_a = 20,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::functor_call<screamer::RollingSpread>)
        .def("reset", &screamer::RollingSpread::reset, "Reset to the initial state.");

    // ----- Performance / risk metrics -----
    nb::class_<screamer::Drawdown, screamer::ScreamerBase>(m, "Drawdown")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::Drawdown::reset, "Reset.");

    nb::class_<screamer::MaxDrawdown, screamer::ScreamerBase>(m, "MaxDrawdown")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::MaxDrawdown::reset, "Reset.");

    nb::class_<screamer::RollingMaxDrawdown, screamer::ScreamerBase>(m, "RollingMaxDrawdown")
        .def(nb::init<screamer::IntArg<int>>(), "window_size"_a = 252)
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::RollingMaxDrawdown::reset, "Reset.");

    nb::class_<screamer::RollingSharpe, screamer::ScreamerBase>(m, "RollingSharpe")
        .def(nb::init<screamer::IntArg<int>, screamer::FloatArg>(),
             "window_size"_a = 252,
             "periods_per_year"_a = 1.0)
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::RollingSharpe::reset, "Reset.");

    nb::class_<screamer::RollingSortino, screamer::ScreamerBase>(m, "RollingSortino")
        .def(nb::init<screamer::IntArg<int>, screamer::FloatArg, screamer::FloatArg>(),
             "window_size"_a = 252,
             "periods_per_year"_a = 1.0,
             "target"_a = 0.0)
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::RollingSortino::reset, "Reset.");

    nb::class_<screamer::RollingInfoRatio, screamer::EvalOp>(m, "RollingInfoRatio")
        .def(nb::init<screamer::IntArg<int>, screamer::FloatArg>(),
             "window_size"_a = 252,
             "periods_per_year"_a = 1.0)
        .def("__call__", &screamer::functor_call<screamer::RollingInfoRatio>)
        .def("reset", &screamer::RollingInfoRatio::reset, "Reset.");

    nb::class_<screamer::RollingCalmar, screamer::ScreamerBase>(m, "RollingCalmar")
        .def(nb::init<screamer::IntArg<int>, screamer::FloatArg>(),
             "window_size"_a = 252,
             "periods_per_year"_a = 1.0)
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::RollingCalmar::reset, "Reset.");

    nb::class_<screamer::RollingHitRate, screamer::ScreamerBase>(m, "RollingHitRate")
        .def(nb::init<screamer::IntArg<int>>(), "window_size"_a = 252)
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::RollingHitRate::reset, "Reset.");

    // ----- Regression-family additions -----
    nb::class_<screamer::RollingAlpha, screamer::EvalOp>(m, "RollingAlpha")
        .def(nb::init<screamer::IntArg<int>, const std::string&>(),
             "window_size"_a = 20,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::functor_call<screamer::RollingAlpha>)
        .def("reset", &screamer::RollingAlpha::reset, "Reset.");

    nb::class_<screamer::RollingResidualStd, screamer::EvalOp>(m, "RollingResidualStd")
        .def(nb::init<screamer::IntArg<int>, const std::string&>(),
             "window_size"_a = 20,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::functor_call<screamer::RollingResidualStd>)
        .def("reset", &screamer::RollingResidualStd::reset, "Reset.");

    // 2 -> 4 OLS fit returning (slope, intercept, r_squared, stderr).
    // First 2->4 consumer of the N->M dispatcher.
    nb::class_<screamer::RollingLinearRegression, screamer::EvalOp>(m, "RollingLinearRegression")
        .def(nb::init<screamer::IntArg<int>, const std::string&>(),
             "window_size"_a = 20,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::functor_call<screamer::RollingLinearRegression>)
        .def("reset", &screamer::RollingLinearRegression::reset, "Reset.");

    // 2 -> 4 online Bayesian regression returning (pred_mean, pred_std, slope, intercept).
    // Uses exponential forgetting with a conjugate Normal-Inverse-Gamma prior.
    nb::class_<screamer::BayesianRegression, screamer::EvalOp>(m, "BayesianRegression")
        .def(nb::init<std::optional<screamer::FloatArg>, std::optional<screamer::FloatArg>, std::optional<screamer::FloatArg>,
                      std::optional<screamer::FloatArg>, screamer::FloatArg, screamer::FloatArg>(),
             "com"_a = nb::none(), "span"_a = nb::none(),
             "halflife"_a = nb::none(), "alpha"_a = nb::none(),
             "prior_precision"_a = 1.0, "prior_sigma"_a = 1.0)
        .def("__call__", &screamer::functor_call<screamer::BayesianRegression>)
        .def("reset", &screamer::BayesianRegression::reset, "Reset to the prior.");

    // Triple-barrier labels: 1 -> 3 (price) and 4 -> 3 (OHLC), outputs
    // (label, return, holding) for the entry `horizon` steps back.
    nb::class_<screamer::TripleBarrier, screamer::EvalOp>(m, "TripleBarrier")
        .def(nb::init<screamer::IntArg<int>, screamer::FloatArg, screamer::FloatArg, const std::string&, screamer::FloatArg>(),
             "horizon"_a = 20, "upper"_a = 0.0, "lower"_a = 0.0,
             "barrier_unit"_a = "fraction", "vol_span"_a = 100.0)
        .def("__call__", &screamer::functor_call<screamer::TripleBarrier>)
        .def("reset", &screamer::TripleBarrier::reset, "Reset.");

    nb::class_<screamer::TripleBarrierOHLC, screamer::EvalOp>(m, "TripleBarrierOHLC")
        .def(nb::init<screamer::IntArg<int>, screamer::FloatArg, screamer::FloatArg, const std::string&, screamer::FloatArg>(),
             "horizon"_a = 20, "upper"_a = 0.0, "lower"_a = 0.0,
             "barrier_unit"_a = "fraction", "vol_span"_a = 100.0)
        .def("__call__", &screamer::functor_call<screamer::TripleBarrierOHLC>)
        .def("reset", &screamer::TripleBarrierOHLC::reset, "Reset.");

    nb::class_<screamer::BacktestPriceTarget, screamer::EvalOp>(m, "BacktestPriceTarget")
        .def(nb::init<screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, const std::string&, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, const std::string&, screamer::FloatArg, screamer::FloatArg>(),
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
        .def(nb::init<screamer::FloatArg, screamer::FloatArg, const std::string&, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, const std::string&>(),
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
        .def(nb::init<screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, const std::string&, const std::string&, screamer::IntArg<int>, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, const std::string&, screamer::FloatArg, screamer::FloatArg>(),
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
        .def(nb::init<screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, const std::string&, const std::string&, screamer::IntArg<int>, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, const std::string&, screamer::FloatArg, screamer::FloatArg>(),
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
        .def(nb::init<screamer::FloatArg, screamer::FloatArg, const std::string&, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, const std::string&>(),
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
        .def(nb::init<screamer::FloatArg, screamer::FloatArg, const std::string&, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, const std::string&>(),
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
        .def(nb::init<screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, const std::string&, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, const std::string&, screamer::FloatArg, screamer::FloatArg>(),
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
        .def(nb::init<screamer::FloatArg, screamer::FloatArg, const std::string&, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, const std::string&>(),
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
        .def(nb::init<screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, const std::string&, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg, const std::string&, screamer::FloatArg, screamer::FloatArg, screamer::FloatArg>(),
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
        .def(nb::init<screamer::FloatArg>(), "periods_per_year"_a = 1.0)
        .def("__call__", &screamer::functor_call<screamer::BacktestReport>)
        .def("reset", &screamer::BacktestReport::reset, "Reset.");

    nb::class_<screamer::BacktestCarry, screamer::EvalOp>(m, "BacktestCarry")
        .def(nb::init<screamer::FloatArg>(), "multiplier"_a = 1.0)
        .def("__call__", &screamer::functor_call<screamer::BacktestCarry>)
        .def("reset", &screamer::BacktestCarry::reset, "Reset.");

    nb::class_<screamer::BacktestStats, screamer::EvalOp>(m, "BacktestStats")
        .def(nb::init<screamer::FloatArg>(), "periods_per_year"_a = 1.0)
        .def("__call__", &screamer::functor_call<screamer::BacktestStats>)
        .def("reset", &screamer::BacktestStats::reset, "Reset.");

    nb::class_<screamer::PortfolioReport, screamer::EvalOp>(m, "PortfolioReport")
        .def(nb::init<>())
        .def("__call__", [](screamer::PortfolioReport& self, nb::args a) {
            return named_call(self, [&] { return portfolio_report_call(self, a); });
        })
        .def("reset", &screamer::PortfolioReport::reset, "Reset.");


    nb::class_<screamer::RowOp, screamer::EvalOp>(m, "RowOp")
        .def("__call__", [](screamer::RowOp& self, nb::args a) {
            return named_call(self, [&] { return row_op_call(self, a); });
        })
        .def("reset", &screamer::RowOp::reset, "Reset.");


    nb::class_<screamer::CrossRank, screamer::RowOp>(m, "CrossRank")
        .def(nb::init<const std::string&>(), "scale"_a = "signed");

    nb::class_<screamer::CrossZscore, screamer::RowOp>(m, "CrossZscore")
        .def(nb::init<>());

    nb::class_<screamer::CrossDemean, screamer::RowOp>(m, "CrossDemean")
        .def(nb::init<>());

    nb::class_<screamer::CrossWinsorize, screamer::RowOp>(m, "CrossWinsorize")
        .def(nb::init<screamer::FloatArg, screamer::FloatArg>(), "lower"_a = 0.05, "upper"_a = 0.95);

    nb::class_<screamer::CrossNormalize, screamer::RowOp>(m, "CrossNormalize")
        .def(nb::init<screamer::FloatArg, std::optional<screamer::FloatArg>>(), "gross"_a = 1.0,
             "net"_a = nb::none());

    nb::class_<screamer::CrossTopBottom, screamer::RowOp>(m, "CrossTopBottom")
        .def(nb::init<screamer::IntArg<int>>(), "count"_a = 1);

    nb::class_<screamer::BacktestPortfolioTarget, screamer::RowOp>(m, "BacktestPortfolioTarget")
        .def(nb::init<screamer::PerAsset, screamer::PerAsset, const std::string&,
                      screamer::PerAsset, screamer::PerAsset, screamer::PerAsset,
                      const std::string&, screamer::FloatArg, screamer::FloatArg, const std::string&,
                      screamer::FloatArg, screamer::FloatArg>(),
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
                      const std::string&, screamer::FloatArg, screamer::FloatArg, const std::string&,
                      screamer::FloatArg, screamer::FloatArg, const std::string&, screamer::IntArg<int>>(),
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
        .def(nb::init<screamer::IntArg<int>, screamer::FloatArg, const std::string&>(),
             "window_size"_a = 20, "mar"_a = 0.0,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::RollingDownsideDeviation::reset, "Reset.");

    nb::class_<screamer::RollingUpsideDeviation, screamer::ScreamerBase>(m, "RollingUpsideDeviation")
        .def(nb::init<screamer::IntArg<int>, screamer::FloatArg, const std::string&>(),
             "window_size"_a = 20, "mar"_a = 0.0,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::RollingUpsideDeviation::reset, "Reset.");

    nb::class_<screamer::RollingOmega, screamer::ScreamerBase>(m, "RollingOmega")
        .def(nb::init<screamer::IntArg<int>, screamer::FloatArg>(),
             "window_size"_a = 20, "threshold"_a = 0.0)
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::RollingOmega::reset, "Reset.");

    nb::class_<screamer::RollingCVaR, screamer::ScreamerBase>(m, "RollingCVaR")
        .def(nb::init<screamer::IntArg<int>, screamer::FloatArg>(),
             "window_size"_a = 20, "alpha"_a = 0.05)
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::RollingCVaR::reset, "Reset.");
}