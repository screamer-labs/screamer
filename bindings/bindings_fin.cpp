#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Required for std::optional support
#include "screamer/common/base.h"
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
#include "screamer/backtest_signal.h"
#include "screamer/backtest_ohlc.h"
#include "screamer/backtest_trades.h"
#include "screamer/backtest_l1.h"
#include "screamer/backtest_l1_trades.h"
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

namespace py = pybind11;

void init_bindings_fin(py::module& m) {

    py::class_<screamer::Return, screamer::ScreamerBase>(m, "Return")
        .def(py::init<int>(), py::arg("window_size") = 1)
        .def("__call__", &screamer::Return::operator(), py::arg("value"))
        .def("reset", &screamer::Return::reset, "Reset to the initial state.");

    py::class_<screamer::LogReturn, screamer::ScreamerBase>(m, "LogReturn")
        .def(py::init<int>(), py::arg("window_size") = 1)
        .def("__call__", &screamer::LogReturn::operator(), py::arg("value"))
        .def("reset", &screamer::LogReturn::reset, "Reset to the initial state.");

    // ROC family: rate-of-change variants. TA-Lib has all three as
    // separate functions; we provide them under TA-Lib's names so
    // users can port directly. ROCP is mathematically identical to
    // Return.
    py::class_<screamer::ROC, screamer::ScreamerBase>(m, "ROC")
        .def(py::init<int>(), py::arg("window_size") = 1)
        .def("__call__", &screamer::ROC::operator(), py::arg("value"))
        .def("reset", &screamer::ROC::reset, "Reset to the initial state.");

    py::class_<screamer::ROCP, screamer::ScreamerBase>(m, "ROCP")
        .def(py::init<int>(), py::arg("window_size") = 1)
        .def("__call__", &screamer::ROCP::operator(), py::arg("value"))
        .def("reset", &screamer::ROCP::reset, "Reset to the initial state.");

    py::class_<screamer::ROCR, screamer::ScreamerBase>(m, "ROCR")
        .def(py::init<int>(), py::arg("window_size") = 1)
        .def("__call__", &screamer::ROCR::operator(), py::arg("value"))
        .def("reset", &screamer::ROCR::reset, "Reset to the initial state.");

    // RollingCorr: 2 inputs (x, y), 1 output (Pearson correlation).
    // Inherits from FunctorBase<_, 2, 1>, NOT ScreamerBase -- the
    // multi-input class hierarchy is separate. handle_input dispatches
    // on the variadic args (scalars / N parallel arrays / list of N-tuples
    // / N parallel iterables).
    py::class_<screamer::RollingCorr, screamer::EvalOp>(m, "RollingCorr")
        .def(py::init<int, const std::string&>(),
             py::arg("window_size") = 20,
             py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingCorr::handle_input)
        .def("reset", &screamer::RollingCorr::reset, "Reset to the initial state.");

    // Rolling sample covariance of two streams.
    py::class_<screamer::RollingCov, screamer::EvalOp>(m, "RollingCov")
        .def(py::init<int, const std::string&>(),
             py::arg("window_size") = 20,
             py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingCov::handle_input)
        .def("reset", &screamer::RollingCov::reset, "Reset to the initial state.");

    // Rolling regression slope of x on y: beta = cov(x, y) / var(y).
    py::class_<screamer::RollingBeta, screamer::EvalOp>(m, "RollingBeta")
        .def(py::init<int, const std::string&>(),
             py::arg("window_size") = 20,
             py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingBeta::handle_input)
        .def("reset", &screamer::RollingBeta::reset, "Reset to the initial state.");

    // Hedge-adjusted residual of x against y: spread = x - beta * y, with
    // beta computed exactly as in RollingBeta.
    py::class_<screamer::RollingSpread, screamer::EvalOp>(m, "RollingSpread")
        .def(py::init<int, const std::string&>(),
             py::arg("window_size") = 20,
             py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingSpread::handle_input)
        .def("reset", &screamer::RollingSpread::reset, "Reset to the initial state.");

    // ----- Performance / risk metrics -----
    py::class_<screamer::Drawdown, screamer::ScreamerBase>(m, "Drawdown")
        .def(py::init<>())
        .def("__call__", &screamer::Drawdown::operator(), py::arg("value"))
        .def("reset", &screamer::Drawdown::reset, "Reset.");

    py::class_<screamer::MaxDrawdown, screamer::ScreamerBase>(m, "MaxDrawdown")
        .def(py::init<>())
        .def("__call__", &screamer::MaxDrawdown::operator(), py::arg("value"))
        .def("reset", &screamer::MaxDrawdown::reset, "Reset.");

    py::class_<screamer::RollingMaxDrawdown, screamer::ScreamerBase>(m, "RollingMaxDrawdown")
        .def(py::init<int>(), py::arg("window_size") = 252)
        .def("__call__", &screamer::RollingMaxDrawdown::operator(), py::arg("value"))
        .def("reset", &screamer::RollingMaxDrawdown::reset, "Reset.");

    py::class_<screamer::RollingSharpe, screamer::ScreamerBase>(m, "RollingSharpe")
        .def(py::init<int, double>(),
             py::arg("window_size") = 252,
             py::arg("periods_per_year") = 1.0)
        .def("__call__", &screamer::RollingSharpe::operator(), py::arg("value"))
        .def("reset", &screamer::RollingSharpe::reset, "Reset.");

    py::class_<screamer::RollingSortino, screamer::ScreamerBase>(m, "RollingSortino")
        .def(py::init<int, double, double>(),
             py::arg("window_size") = 252,
             py::arg("periods_per_year") = 1.0,
             py::arg("target") = 0.0)
        .def("__call__", &screamer::RollingSortino::operator(), py::arg("value"))
        .def("reset", &screamer::RollingSortino::reset, "Reset.");

    py::class_<screamer::RollingInfoRatio, screamer::EvalOp>(m, "RollingInfoRatio")
        .def(py::init<int, double>(),
             py::arg("window_size") = 252,
             py::arg("periods_per_year") = 1.0)
        .def("__call__", &screamer::RollingInfoRatio::handle_input)
        .def("reset", &screamer::RollingInfoRatio::reset, "Reset.");

    py::class_<screamer::RollingCalmar, screamer::ScreamerBase>(m, "RollingCalmar")
        .def(py::init<int, double>(),
             py::arg("window_size") = 252,
             py::arg("periods_per_year") = 1.0)
        .def("__call__", &screamer::RollingCalmar::operator(), py::arg("value"))
        .def("reset", &screamer::RollingCalmar::reset, "Reset.");

    py::class_<screamer::RollingHitRate, screamer::ScreamerBase>(m, "RollingHitRate")
        .def(py::init<int>(), py::arg("window_size") = 252)
        .def("__call__", &screamer::RollingHitRate::operator(), py::arg("value"))
        .def("reset", &screamer::RollingHitRate::reset, "Reset.");

    // ----- Regression-family additions -----
    py::class_<screamer::RollingAlpha, screamer::EvalOp>(m, "RollingAlpha")
        .def(py::init<int, const std::string&>(),
             py::arg("window_size") = 20,
             py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingAlpha::handle_input)
        .def("reset", &screamer::RollingAlpha::reset, "Reset.");

    py::class_<screamer::RollingResidualStd, screamer::EvalOp>(m, "RollingResidualStd")
        .def(py::init<int, const std::string&>(),
             py::arg("window_size") = 20,
             py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingResidualStd::handle_input)
        .def("reset", &screamer::RollingResidualStd::reset, "Reset.");

    // 2 -> 4 OLS fit returning (slope, intercept, r_squared, stderr).
    // First 2->4 consumer of the N->M dispatcher.
    py::class_<screamer::RollingLinearRegression, screamer::EvalOp>(m, "RollingLinearRegression")
        .def(py::init<int, const std::string&>(),
             py::arg("window_size") = 20,
             py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingLinearRegression::handle_input)
        .def("reset", &screamer::RollingLinearRegression::reset, "Reset.");

    py::class_<screamer::BacktestSignal, screamer::EvalOp>(m, "BacktestSignal")
        .def(py::init<double, double>(),
             py::arg("spread") = 0.0, py::arg("fee") = 0.0)
        .def("__call__", &screamer::BacktestSignal::handle_input)
        .def("reset", &screamer::BacktestSignal::reset, "Reset.");

    py::class_<screamer::BacktestOHLC, screamer::EvalOp>(m, "BacktestOHLC")
        .def(py::init<double, double, double, const std::string&>(),
             py::arg("spread") = 0.0, py::arg("taker_fee") = 0.0,
             py::arg("maker_fee") = 0.0, py::arg("fill") = "touch")
        .def("__call__", &screamer::BacktestOHLC::handle_input)
        .def("reset", &screamer::BacktestOHLC::reset, "Reset.");

    py::class_<screamer::BacktestTrades, screamer::EvalOp>(m, "BacktestTrades")
        .def(py::init<double, const std::string&, double>(),
             py::arg("maker_fee") = 0.0, py::arg("fill") = "touch",
             py::arg("participation_ratio") = 1.0)
        .def("__call__", &screamer::BacktestTrades::handle_input)
        .def("reset", &screamer::BacktestTrades::reset, "Reset.");

    py::class_<screamer::BacktestL1, screamer::EvalOp>(m, "BacktestL1")
        .def(py::init<double, double, const std::string&, double, double, double, double>(),
             py::arg("maker_fee") = 0.0, py::arg("taker_fee") = 0.0,
             py::arg("fill") = "breach", py::arg("participation_ratio") = 1.0,
             py::arg("tick_size") = 0.0,
             py::arg("max_position") = std::numeric_limits<double>::infinity(),
             py::arg("min_position") = -std::numeric_limits<double>::infinity())
        .def("__call__", &screamer::BacktestL1::handle_input)
        .def("reset", &screamer::BacktestL1::reset, "Reset.");

    py::class_<screamer::BacktestL1Trades, screamer::EvalOp>(m, "BacktestL1Trades")
        .def(py::init<double, double, const std::string&, double, double, double, double>(),
             py::arg("maker_fee") = 0.0, py::arg("taker_fee") = 0.0,
             py::arg("fill") = "touch", py::arg("participation_ratio") = 1.0,
             py::arg("tick_size") = 0.0,
             py::arg("max_position") = std::numeric_limits<double>::infinity(),
             py::arg("min_position") = -std::numeric_limits<double>::infinity())
        .def("__call__", &screamer::BacktestL1Trades::handle_input)
        .def("reset", &screamer::BacktestL1Trades::reset, "Reset.");

    py::class_<screamer::BacktestReport, screamer::EvalOp>(m, "BacktestReport")
        .def(py::init<>())
        .def("__call__", &screamer::BacktestReport::handle_input)
        .def("reset", &screamer::BacktestReport::reset, "Reset.");

    py::class_<screamer::RollingDownsideDeviation, screamer::ScreamerBase>(m, "RollingDownsideDeviation")
        .def(py::init<int, double, const std::string&>(),
             py::arg("window_size") = 20, py::arg("mar") = 0.0,
             py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingDownsideDeviation::operator(), py::arg("value"))
        .def("reset", &screamer::RollingDownsideDeviation::reset, "Reset.");

    py::class_<screamer::RollingOmega, screamer::ScreamerBase>(m, "RollingOmega")
        .def(py::init<int, double>(),
             py::arg("window_size") = 20, py::arg("threshold") = 0.0)
        .def("__call__", &screamer::RollingOmega::operator(), py::arg("value"))
        .def("reset", &screamer::RollingOmega::reset, "Reset.");

    py::class_<screamer::RollingCVaR, screamer::ScreamerBase>(m, "RollingCVaR")
        .def(py::init<int, double>(),
             py::arg("window_size") = 20, py::arg("alpha") = 0.05)
        .def("__call__", &screamer::RollingCVaR::operator(), py::arg("value"))
        .def("reset", &screamer::RollingCVaR::reset, "Reset.");
}