#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Required for std::optional support
#include "screamer/common/base.h"
#include "screamer/return.h"
#include "screamer/log_return.h"
#include "screamer/rolling_fracdiff.h"
#include "screamer/rolling_corr.h"
#include "screamer/rolling_cov.h"
#include "screamer/rolling_beta.h"
#include "screamer/rolling_spread.h"

namespace py = pybind11;

void init_bindings_fin(py::module& m) {

    py::class_<screamer::Return, screamer::ScreamerBase>(m, "Return")
        .def(py::init<int>(), py::arg("window_size"))
        .def("__call__", &screamer::Return::operator(), py::arg("value"))
        .def("reset", &screamer::Return::reset, "Reset to the initial state.");

    py::class_<screamer::LogReturn, screamer::ScreamerBase>(m, "LogReturn")
        .def(py::init<int>(), py::arg("window_size"))
        .def("__call__", &screamer::LogReturn::operator(), py::arg("value"))
        .def("reset", &screamer::LogReturn::reset, "Reset to the initial state.");

    py::class_<screamer::RollingFracDiff, screamer::ScreamerBase>(m, "RollingFracDiff")
        .def(py::init<double, int, double>(), py::arg("frac_order"), py::arg("window_size"), py::arg("threshold")=1e-5)
        .def("__call__", &screamer::RollingFracDiff::operator(), py::arg("value"))
        .def("reset", &screamer::RollingFracDiff::reset, "Reset to the initial state.");

    // RollingCorr: 2 inputs (x, y), 1 output (Pearson correlation).
    // Inherits from FunctorBase<_, 2, 1>, NOT ScreamerBase -- the
    // multi-input class hierarchy is separate. handle_input dispatches
    // on the variadic args (scalars / N parallel arrays / list of N-tuples
    // / N parallel iterables).
    py::class_<screamer::RollingCorr>(m, "RollingCorr")
        .def(py::init<int, const std::string&>(),
             py::arg("window_size"),
             py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingCorr::handle_input)
        .def("reset", &screamer::RollingCorr::reset, "Reset to the initial state.");

    // Rolling sample covariance of two streams.
    py::class_<screamer::RollingCov>(m, "RollingCov")
        .def(py::init<int, const std::string&>(),
             py::arg("window_size"),
             py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingCov::handle_input)
        .def("reset", &screamer::RollingCov::reset, "Reset to the initial state.");

    // Rolling regression slope of x on y: beta = cov(x, y) / var(y).
    py::class_<screamer::RollingBeta>(m, "RollingBeta")
        .def(py::init<int, const std::string&>(),
             py::arg("window_size"),
             py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingBeta::handle_input)
        .def("reset", &screamer::RollingBeta::reset, "Reset to the initial state.");

    // Hedge-adjusted residual of x against y: spread = x - beta * y, with
    // beta computed exactly as in RollingBeta.
    py::class_<screamer::RollingSpread>(m, "RollingSpread")
        .def(py::init<int, const std::string&>(),
             py::arg("window_size"),
             py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingSpread::handle_input)
        .def("reset", &screamer::RollingSpread::reset, "Reset to the initial state.");
}