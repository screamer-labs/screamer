#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Required for std::optional support
#include "screamer/common/base.h"
#include "screamer/lag.h"
#include "screamer/diff.h"
#include "screamer/diff2.h"
#include "screamer/cum_sum.h"
#include "screamer/cum_prod.h"
#include "screamer/cum_max.h"
#include "screamer/cum_min.h"
#include "screamer/detrend.h"

namespace py = pybind11;

void init_bindings_misc(py::module& m) {

    py::class_<screamer::Diff, screamer::ScreamerBase>(m, "Diff")
        .def(py::init<int, const std::string&>(), py::arg("window_size"), py::arg("start_policy") = "strict")
        .def("__call__", &screamer::Diff::operator(), py::arg("value"))
        .def("reset", &screamer::Diff::reset, "Reset to the initial state.");

    // Diff2: second-order finite difference (discrete second derivative).
    // Two NaN warmup samples under "strict". Distinct from Diff(2),
    // which is the lag-2 first difference.
    py::class_<screamer::Diff2, screamer::ScreamerBase>(m, "Diff2")
        .def(py::init<const std::string&>(), py::arg("start_policy") = "strict")
        .def("__call__", &screamer::Diff2::operator(), py::arg("value"))
        .def("reset", &screamer::Diff2::reset, "Reset to the initial state.");

    py::class_<screamer::Lag, screamer::ScreamerBase>(m, "Lag")
        .def(py::init<int, const std::string&>(), py::arg("window_size"), py::arg("start_policy") = "strict")
        .def("__call__", &screamer::Lag::operator(), py::arg("value"))
        .def("reset", &screamer::Lag::reset, "Reset to the initial state.");

    // Cumulative reductions from t=0. O(1) memory each. NaN propagates
    // (matches numpy semantics, not pandas skipna=True).
    py::class_<screamer::CumSum, screamer::ScreamerBase>(m, "CumSum")
        .def(py::init<>())
        .def("__call__", &screamer::CumSum::operator(), py::arg("value"))
        .def("reset", &screamer::CumSum::reset, "Reset to the initial state.");

    py::class_<screamer::CumProd, screamer::ScreamerBase>(m, "CumProd")
        .def(py::init<>())
        .def("__call__", &screamer::CumProd::operator(), py::arg("value"))
        .def("reset", &screamer::CumProd::reset, "Reset to the initial state.");

    py::class_<screamer::CumMax, screamer::ScreamerBase>(m, "CumMax")
        .def(py::init<>())
        .def("__call__", &screamer::CumMax::operator(), py::arg("value"))
        .def("reset", &screamer::CumMax::reset, "Reset to the initial state.");

    py::class_<screamer::CumMin, screamer::ScreamerBase>(m, "CumMin")
        .def(py::init<>())
        .def("__call__", &screamer::CumMin::operator(), py::arg("value"))
        .def("reset", &screamer::CumMin::reset, "Reset to the initial state.");

    // Detrend: y[t] = x[t] - RollingMean(window)(x)[t].
    py::class_<screamer::Detrend, screamer::ScreamerBase>(m, "Detrend")
        .def(py::init<int, const std::string&>(),
             py::arg("window_size"),
             py::arg("start_policy") = "strict")
        .def("__call__", &screamer::Detrend::operator(), py::arg("value"))
        .def("reset", &screamer::Detrend::reset, "Reset to the initial state.");

}