#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "screamer/common/base.h"
#include "screamer/expanding_mean.h"
#include "screamer/expanding_var.h"
#include "screamer/expanding_std.h"
#include "screamer/expanding_skew.h"
#include "screamer/expanding_kurt.h"
#include "screamer/expanding_slope.h"
#include "screamer/expanding_sum.h"
#include "screamer/expanding_max.h"
#include "screamer/expanding_min.h"
#include "screamer/expanding_prod.h"

namespace py = pybind11;

void init_bindings_expanding(py::module& m) {

    // Whole-history, resettable moment statistics. No window, no start_policy.
    // ddof / bias conventions match the Rolling* family (and pandas
    // .expanding() defaults): var/std ddof=1, skew = adjusted G1, kurt =
    // Fisher excess with bias correction.
    py::class_<screamer::ExpandingMean, screamer::ScreamerBase>(m, "ExpandingMean")
        .def(py::init<>())
        .def("__call__", &screamer::ExpandingMean::operator(), py::arg("value"))
        .def("reset", &screamer::ExpandingMean::reset, "Reset to the initial state.");

    py::class_<screamer::ExpandingVar, screamer::ScreamerBase>(m, "ExpandingVar")
        .def(py::init<>())
        .def("__call__", &screamer::ExpandingVar::operator(), py::arg("value"))
        .def("reset", &screamer::ExpandingVar::reset, "Reset to the initial state.");

    py::class_<screamer::ExpandingStd, screamer::ScreamerBase>(m, "ExpandingStd")
        .def(py::init<>())
        .def("__call__", &screamer::ExpandingStd::operator(), py::arg("value"))
        .def("reset", &screamer::ExpandingStd::reset, "Reset to the initial state.");

    py::class_<screamer::ExpandingSkew, screamer::ScreamerBase>(m, "ExpandingSkew")
        .def(py::init<>())
        .def("__call__", &screamer::ExpandingSkew::operator(), py::arg("value"))
        .def("reset", &screamer::ExpandingSkew::reset, "Reset to the initial state.");

    py::class_<screamer::ExpandingKurt, screamer::ScreamerBase>(m, "ExpandingKurt")
        .def(py::init<>())
        .def("__call__", &screamer::ExpandingKurt::operator(), py::arg("value"))
        .def("reset", &screamer::ExpandingKurt::reset, "Reset to the initial state.");

    // OLS slope of y against an implicit time axis x = 0..n-1.
    py::class_<screamer::ExpandingSlope, screamer::ScreamerBase>(m, "ExpandingSlope")
        .def(py::init<>())
        .def("__call__", &screamer::ExpandingSlope::operator(), py::arg("value"))
        .def("reset", &screamer::ExpandingSlope::reset, "Reset to the initial state.");

    // Reduction aliases -- thin subclasses of Cum* exposed under Expanding*.
    py::class_<screamer::ExpandingSum, screamer::ScreamerBase>(m, "ExpandingSum")
        .def(py::init<>())
        .def("__call__", &screamer::ExpandingSum::operator(), py::arg("value"))
        .def("reset", &screamer::ExpandingSum::reset, "Reset to the initial state.");

    py::class_<screamer::ExpandingMax, screamer::ScreamerBase>(m, "ExpandingMax")
        .def(py::init<>())
        .def("__call__", &screamer::ExpandingMax::operator(), py::arg("value"))
        .def("reset", &screamer::ExpandingMax::reset, "Reset to the initial state.");

    py::class_<screamer::ExpandingMin, screamer::ScreamerBase>(m, "ExpandingMin")
        .def(py::init<>())
        .def("__call__", &screamer::ExpandingMin::operator(), py::arg("value"))
        .def("reset", &screamer::ExpandingMin::reset, "Reset to the initial state.");

    py::class_<screamer::ExpandingProd, screamer::ScreamerBase>(m, "ExpandingProd")
        .def(py::init<>())
        .def("__call__", &screamer::ExpandingProd::operator(), py::arg("value"))
        .def("reset", &screamer::ExpandingProd::reset, "Reset to the initial state.");
}
