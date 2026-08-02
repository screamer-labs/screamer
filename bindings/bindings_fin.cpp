#include <limits>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
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
#include "screamer/rolling_spread.h"
#include "screamer/drawdown.h"
#include "screamer/max_drawdown.h"
#include "screamer/backtest_price_target.h"
#include "screamer/backtest_ohlc_orders.h"
#include "screamer/backtest_ohlc_target.h"
#include "screamer/backtest_trades_orders.h"
#include "screamer/backtest_trades_target.h"
#include "screamer/backtest_l1_orders.h"
#include "screamer/backtest_l1_target.h"
#include "screamer/backtest_l1trades_orders.h"
#include "screamer/backtest_report.h"
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

void init_bindings_fin(nb::module_& m) {

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
        .def("__call__", &screamer::RollingCorr::handle_input)
        .def("reset", &screamer::RollingCorr::reset, "Reset to the initial state.");

    // Rolling sample covariance of two streams.
    nb::class_<screamer::RollingCov, screamer::EvalOp>(m, "RollingCov")
        .def(nb::init<int, const std::string&>(),
             "window_size"_a = 20,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::RollingCov::handle_input)
        .def("reset", &screamer::RollingCov::reset, "Reset to the initial state.");

    // Rolling regression slope of x on y: beta = cov(x, y) / var(y).
    nb::class_<screamer::RollingBeta, screamer::EvalOp>(m, "RollingBeta")
        .def(nb::init<int, const std::string&>(),
             "window_size"_a = 20,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::RollingBeta::handle_input)
        .def("reset", &screamer::RollingBeta::reset, "Reset to the initial state.");

    // Hedge-adjusted residual of x against y: spread = x - beta * y, with
    // beta computed exactly as in RollingBeta.
    nb::class_<screamer::RollingSpread, screamer::EvalOp>(m, "RollingSpread")
        .def(nb::init<int, const std::string&>(),
             "window_size"_a = 20,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::RollingSpread::handle_input)
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
        .def("__call__", &screamer::RollingInfoRatio::handle_input)
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
        .def("__call__", &screamer::RollingAlpha::handle_input)
        .def("reset", &screamer::RollingAlpha::reset, "Reset.");

    nb::class_<screamer::RollingResidualStd, screamer::EvalOp>(m, "RollingResidualStd")
        .def(nb::init<int, const std::string&>(),
             "window_size"_a = 20,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::RollingResidualStd::handle_input)
        .def("reset", &screamer::RollingResidualStd::reset, "Reset.");

    // 2 -> 4 OLS fit returning (slope, intercept, r_squared, stderr).
    // First 2->4 consumer of the N->M dispatcher.
    nb::class_<screamer::RollingLinearRegression, screamer::EvalOp>(m, "RollingLinearRegression")
        .def(nb::init<int, const std::string&>(),
             "window_size"_a = 20,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::RollingLinearRegression::handle_input)
        .def("reset", &screamer::RollingLinearRegression::reset, "Reset.");

    // 2 -> 4 online Bayesian regression returning (pred_mean, pred_std, slope, intercept).
    // Uses exponential forgetting with a conjugate Normal-Inverse-Gamma prior.
    nb::class_<screamer::BayesianRegression, screamer::EvalOp>(m, "BayesianRegression")
        .def(nb::init<std::optional<double>, std::optional<double>, std::optional<double>,
                      std::optional<double>, double, double>(),
             "com"_a = nb::none(), "span"_a = nb::none(),
             "halflife"_a = nb::none(), "alpha"_a = nb::none(),
             "prior_precision"_a = 1.0, "prior_sigma"_a = 1.0)
        .def("__call__", &screamer::BayesianRegression::handle_input)
        .def("reset", &screamer::BayesianRegression::reset, "Reset to the prior.");

    nb::class_<screamer::BacktestPriceTarget, screamer::EvalOp>(m, "BacktestPriceTarget")
        .def(nb::init<double, double, double, double>(),
             "spread"_a = 0.0, "fee"_a = 0.0,
             "min_position"_a = -std::numeric_limits<double>::infinity(),
             "max_position"_a = std::numeric_limits<double>::infinity())
        .def("__call__", &screamer::BacktestPriceTarget::handle_input)
        .def("reset", &screamer::BacktestPriceTarget::reset, "Reset.");

    nb::class_<screamer::BacktestOHLCOrders, screamer::EvalOp>(m, "BacktestOHLCOrders")
        .def(nb::init<double, double, const std::string&, double, double, double, double>(),
             "maker_fee"_a = 0.0, "taker_fee"_a = 0.0,
             "fill"_a = "touch", "participation_ratio"_a = 1.0,
             "tick_size"_a = 0.0,
             "min_position"_a = -std::numeric_limits<double>::infinity(),
             "max_position"_a = std::numeric_limits<double>::infinity())
        .def("__call__", &screamer::BacktestOHLCOrders::handle_input)
        .def("reset", &screamer::BacktestOHLCOrders::reset, "Reset.");

    nb::class_<screamer::BacktestOHLCTarget, screamer::EvalOp>(m, "BacktestOHLCTarget")
        .def(nb::init<double, double, double, double>(),
             "taker_fee"_a = 0.0, "tick_size"_a = 0.0,
             "min_position"_a = -std::numeric_limits<double>::infinity(),
             "max_position"_a = std::numeric_limits<double>::infinity())
        .def("__call__", &screamer::BacktestOHLCTarget::handle_input)
        .def("reset", &screamer::BacktestOHLCTarget::reset, "Reset.");

    nb::class_<screamer::BacktestTradesOrders, screamer::EvalOp>(m, "BacktestTradesOrders")
        .def(nb::init<double, double, const std::string&, double, double, double, double>(),
             "maker_fee"_a = 0.0, "taker_fee"_a = 0.0,
             "fill"_a = "touch", "participation_ratio"_a = 1.0,
             "tick_size"_a = 0.0,
             "min_position"_a = -std::numeric_limits<double>::infinity(),
             "max_position"_a = std::numeric_limits<double>::infinity())
        .def("__call__", &screamer::BacktestTradesOrders::handle_input)
        .def("reset", &screamer::BacktestTradesOrders::reset, "Reset.");

    nb::class_<screamer::BacktestL1Orders, screamer::EvalOp>(m, "BacktestL1Orders")
        .def(nb::init<double, double, const std::string&, double, double, double, double>(),
             "maker_fee"_a = 0.0, "taker_fee"_a = 0.0,
             "fill"_a = "breach", "participation_ratio"_a = 1.0,
             "tick_size"_a = 0.0,
             "min_position"_a = -std::numeric_limits<double>::infinity(),
             "max_position"_a = std::numeric_limits<double>::infinity())
        .def("__call__", &screamer::BacktestL1Orders::handle_input)
        .def("reset", &screamer::BacktestL1Orders::reset, "Reset.");

    nb::class_<screamer::BacktestL1Target, screamer::EvalOp>(m, "BacktestL1Target")
        .def(nb::init<double, double, double, double>(),
             "taker_fee"_a = 0.0, "tick_size"_a = 0.0,
             "min_position"_a = -std::numeric_limits<double>::infinity(),
             "max_position"_a = std::numeric_limits<double>::infinity())
        .def("__call__", &screamer::BacktestL1Target::handle_input)
        .def("reset", &screamer::BacktestL1Target::reset, "Reset.");

    nb::class_<screamer::BacktestL1TradesOrders, screamer::EvalOp>(m, "BacktestL1TradesOrders")
        .def(nb::init<double, double, const std::string&, double, double, double, double>(),
             "maker_fee"_a = 0.0, "taker_fee"_a = 0.0,
             "fill"_a = "touch", "participation_ratio"_a = 1.0,
             "tick_size"_a = 0.0,
             "min_position"_a = -std::numeric_limits<double>::infinity(),
             "max_position"_a = std::numeric_limits<double>::infinity())
        .def("__call__", &screamer::BacktestL1TradesOrders::handle_input)
        .def("reset", &screamer::BacktestL1TradesOrders::reset, "Reset.");

    nb::class_<screamer::BacktestTradesTarget, screamer::EvalOp>(m, "BacktestTradesTarget")
        .def(nb::init<double, double, double, double>(),
             "taker_fee"_a = 0.0, "tick_size"_a = 0.0,
             "min_position"_a = -std::numeric_limits<double>::infinity(),
             "max_position"_a = std::numeric_limits<double>::infinity())
        .def("__call__", &screamer::BacktestTradesTarget::handle_input)
        .def("reset", &screamer::BacktestTradesTarget::reset, "Reset.");

    nb::class_<screamer::BacktestReport, screamer::EvalOp>(m, "BacktestReport")
        .def(nb::init<>())
        .def("__call__", &screamer::BacktestReport::handle_input)
        .def("reset", &screamer::BacktestReport::reset, "Reset.");

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