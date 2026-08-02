#include <nanobind/nanobind.h>
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

namespace nb = nanobind;
using namespace nb::literals;

void init_bindings_expanding(nb::module_& m) {

    // Whole-history, resettable moment statistics. No window, no start_policy.
    // ddof / bias conventions match the Rolling* family (and pandas
    // .expanding() defaults): var/std ddof=1, skew = adjusted G1, kurt =
    // Fisher excess with bias correction.
    nb::class_<screamer::ExpandingMean, screamer::ScreamerBase>(m, "ExpandingMean")
        .def(nb::init<>())
        .def("__call__", &screamer::ExpandingMean::operator(), "value"_a)
        .def("reset", &screamer::ExpandingMean::reset, "Reset to the initial state.");

    nb::class_<screamer::ExpandingVar, screamer::ScreamerBase>(m, "ExpandingVar")
        .def(nb::init<>())
        .def("__call__", &screamer::ExpandingVar::operator(), "value"_a)
        .def("reset", &screamer::ExpandingVar::reset, "Reset to the initial state.");

    nb::class_<screamer::ExpandingStd, screamer::ScreamerBase>(m, "ExpandingStd")
        .def(nb::init<>())
        .def("__call__", &screamer::ExpandingStd::operator(), "value"_a)
        .def("reset", &screamer::ExpandingStd::reset, "Reset to the initial state.");

    nb::class_<screamer::ExpandingSkew, screamer::ScreamerBase>(m, "ExpandingSkew")
        .def(nb::init<>())
        .def("__call__", &screamer::ExpandingSkew::operator(), "value"_a)
        .def("reset", &screamer::ExpandingSkew::reset, "Reset to the initial state.");

    nb::class_<screamer::ExpandingKurt, screamer::ScreamerBase>(m, "ExpandingKurt")
        .def(nb::init<>())
        .def("__call__", &screamer::ExpandingKurt::operator(), "value"_a)
        .def("reset", &screamer::ExpandingKurt::reset, "Reset to the initial state.");

    // OLS slope of y against an implicit time axis x = 0..n-1.
    nb::class_<screamer::ExpandingSlope, screamer::ScreamerBase>(m, "ExpandingSlope")
        .def(nb::init<>())
        .def("__call__", &screamer::ExpandingSlope::operator(), "value"_a)
        .def("reset", &screamer::ExpandingSlope::reset, "Reset to the initial state.");

    // Reduction aliases -- thin subclasses of Cum* exposed under Expanding*.
    nb::class_<screamer::ExpandingSum, screamer::ScreamerBase>(m, "ExpandingSum")
        .def(nb::init<>())
        .def("__call__", &screamer::ExpandingSum::operator(), "value"_a)
        .def("reset", &screamer::ExpandingSum::reset, "Reset to the initial state.");

    nb::class_<screamer::ExpandingMax, screamer::ScreamerBase>(m, "ExpandingMax")
        .def(nb::init<>())
        .def("__call__", &screamer::ExpandingMax::operator(), "value"_a)
        .def("reset", &screamer::ExpandingMax::reset, "Reset to the initial state.");

    nb::class_<screamer::ExpandingMin, screamer::ScreamerBase>(m, "ExpandingMin")
        .def(nb::init<>())
        .def("__call__", &screamer::ExpandingMin::operator(), "value"_a)
        .def("reset", &screamer::ExpandingMin::reset, "Reset to the initial state.");

    nb::class_<screamer::ExpandingProd, screamer::ScreamerBase>(m, "ExpandingProd")
        .def(nb::init<>())
        .def("__call__", &screamer::ExpandingProd::operator(), "value"_a)
        .def("reset", &screamer::ExpandingProd::reset, "Reset to the initial state.");
}
