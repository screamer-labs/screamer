#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "screamer/common/base.h"
#include "screamer/common/dispatch.h"
#include "screamer/lag.h"
#include "screamer/diff.h"
#include "screamer/diff2.h"
#include "screamer/momentum.h"
#include "screamer/cum_sum.h"
#include "screamer/cum_prod.h"
#include "screamer/cum_max.h"
#include "screamer/cum_min.h"
#include "screamer/first.h"
#include "screamer/last.h"
#include "screamer/detrend.h"
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

void init_bindings_misc(nb::module_& m) {

    nb::class_<screamer::Diff, screamer::ScreamerBase>(m, "Diff")
        .def(nb::init<int, const std::string&>(), "window_size"_a = 1, "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Diff::reset, "Reset to the initial state.");

    // Momentum(k): mathematically identical to Diff(k). Exposed under
    // its TA-Lib name (MOM) for portability and discoverability.
    nb::class_<screamer::Momentum, screamer::ScreamerBase>(m, "Momentum")
        .def(nb::init<int, const std::string&>(), "window_size"_a = 10, "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Momentum::reset, "Reset to the initial state.");

    // Diff2: second-order finite difference (discrete second derivative).
    // Two NaN warmup samples under "strict". Distinct from Diff(2),
    // which is the lag-2 first difference.
    nb::class_<screamer::Diff2, screamer::ScreamerBase>(m, "Diff2")
        .def(nb::init<const std::string&>(), "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Diff2::reset, "Reset to the initial state.");

    nb::class_<screamer::Lag, screamer::ScreamerBase>(m, "Lag")
        .def(nb::init<int, const std::string&>(), "window_size"_a = 1, "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Lag::reset, "Reset to the initial state.");

    // Cumulative reductions from t=0. O(1) memory each. NaN policy: ignore
    // (a NaN input leaves state unchanged and emits NaN at that step only).
    nb::class_<screamer::CumSum, screamer::ScreamerBase>(m, "CumSum")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::CumSum::reset, "Reset to the initial state.");

    nb::class_<screamer::CumProd, screamer::ScreamerBase>(m, "CumProd")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::CumProd::reset, "Reset to the initial state.");

    nb::class_<screamer::CumMax, screamer::ScreamerBase>(m, "CumMax")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::CumMax::reset, "Reset to the initial state.");

    nb::class_<screamer::CumMin, screamer::ScreamerBase>(m, "CumMin")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::CumMin::reset, "Reset to the initial state.");

    nb::class_<screamer::First, screamer::ScreamerBase>(m, "First")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::First::reset, "Reset to the initial state.");

    nb::class_<screamer::Last, screamer::ScreamerBase>(m, "Last")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Last::reset, "Reset to the initial state.");

    // Detrend: y[t] = x[t] - RollingMean(window)(x)[t].
    nb::class_<screamer::Detrend, screamer::ScreamerBase>(m, "Detrend")
        .def(nb::init<int, const std::string&>(),
             "window_size"_a = 20,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Detrend::reset, "Reset to the initial state.");

}