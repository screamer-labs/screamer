#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include "screamer/common/base.h"
#include "screamer/common/dispatch.h"
#include "screamer/common/eval_op.h"
#include "screamer/ew_mean.h"
#include "screamer/ew_var.h"
#include "screamer/ew_std.h"
#include "screamer/ew_zscore.h"
#include "screamer/ew_skew.h"
#include "screamer/ew_kurt.h"
#include "screamer/ew_rms.h"
#include "screamer/ew_cov.h"
#include "screamer/ew_corr.h"
#include "screamer/ew_beta.h"
#include "screamer/ew_spread.h"
#include "screamer/ew_alpha.h"
#include "screamer/ew_residual_std.h"

namespace nb = nanobind;
using namespace nb::literals;

void init_bindings_ew(nb::module_& m) {

     nb::class_<screamer::EwMean, screamer::ScreamerBase>(m, "EwMean")
        .def(
          nb::init<
               std::optional<double>,
               std::optional<double>,
               std::optional<double>,
               std::optional<double>
          >(),
          "com"_a = nb::none(),
          "span"_a = nb::none(),
          "halflife"_a = nb::none(),
          "alpha"_a = nb::none()
        )
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::EwMean::reset, "Reset to the initial state.");

     nb::class_<screamer::EwVar, screamer::ScreamerBase>(m, "EwVar")
        .def(
          nb::init<
               std::optional<double>,
               std::optional<double>,
               std::optional<double>,
               std::optional<double>
          >(),
          "com"_a = nb::none(),
          "span"_a = nb::none(),
          "halflife"_a = nb::none(),
          "alpha"_a = nb::none()
        )
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::EwVar::reset, "Reset to the initial state.");

     
     nb::class_<screamer::EwStd, screamer::ScreamerBase>(m, "EwStd")
        .def(
          nb::init<
               std::optional<double>,
               std::optional<double>,
               std::optional<double>,
               std::optional<double>
          >(),
          "com"_a = nb::none(),
          "span"_a = nb::none(),
          "halflife"_a = nb::none(),
          "alpha"_a = nb::none()
        )
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::EwStd::reset, "Reset to the initial state.");

     
     nb::class_<screamer::EwZscore, screamer::ScreamerBase>(m, "EwZscore")
        .def(
          nb::init<
               std::optional<double>,
               std::optional<double>,
               std::optional<double>,
               std::optional<double>
          >(),
          "com"_a = nb::none(),
          "span"_a = nb::none(),
          "halflife"_a = nb::none(),
          "alpha"_a = nb::none()
        )
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::EwZscore::reset, "Reset to the initial state.");


     nb::class_<screamer::EwSkew, screamer::ScreamerBase>(m, "EwSkew")
        .def(
          nb::init<
               std::optional<double>,
               std::optional<double>,
               std::optional<double>,
               std::optional<double>
          >(),
          "com"_a = nb::none(),
          "span"_a = nb::none(),
          "halflife"_a = nb::none(),
          "alpha"_a = nb::none()
        )
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::EwSkew::reset, "Reset to the initial state.");


     nb::class_<screamer::EwKurt, screamer::ScreamerBase>(m, "EwKurt")
        .def(
          nb::init<
               std::optional<double>,
               std::optional<double>,
               std::optional<double>,
               std::optional<double>
          >(),
          "com"_a = nb::none(),
          "span"_a = nb::none(),
          "halflife"_a = nb::none(),
          "alpha"_a = nb::none()
        )
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::EwKurt::reset, "Reset to the initial state.");


     nb::class_<screamer::EwRms, screamer::ScreamerBase>(m, "EwRms")
        .def(
          nb::init<
               std::optional<double>,
               std::optional<double>,
               std::optional<double>,
               std::optional<double>
          >(),
          "com"_a = nb::none(),
          "span"_a = nb::none(),
          "halflife"_a = nb::none(),
          "alpha"_a = nb::none()
        )
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::EwRms::reset, "Reset to the initial state.");

     // 2-input EW pair statistics. Same com/span/halflife/alpha mutex as
     // the 1-input variants. Bias-corrected like EwVar (matches pandas
     // ewm(adjust=True, bias=False).cov / .corr).
     nb::class_<screamer::EwCov, screamer::EvalOp>(m, "EwCov")
        .def(
          nb::init<
               std::optional<double>,
               std::optional<double>,
               std::optional<double>,
               std::optional<double>
          >(),
          "com"_a = nb::none(),
          "span"_a = nb::none(),
          "halflife"_a = nb::none(),
          "alpha"_a = nb::none()
        )
        .def("__call__", &screamer::functor_call<screamer::EwCov>)
        .def("reset", &screamer::EwCov::reset, "Reset to the initial state.");

     nb::class_<screamer::EwCorr, screamer::EvalOp>(m, "EwCorr")
        .def(
          nb::init<
               std::optional<double>,
               std::optional<double>,
               std::optional<double>,
               std::optional<double>
          >(),
          "com"_a = nb::none(),
          "span"_a = nb::none(),
          "halflife"_a = nb::none(),
          "alpha"_a = nb::none()
        )
        .def("__call__", &screamer::functor_call<screamer::EwCorr>)
        .def("reset", &screamer::EwCorr::reset, "Reset to the initial state.");

     nb::class_<screamer::EwBeta, screamer::EvalOp>(m, "EwBeta")
        .def(
          nb::init<
               std::optional<double>,
               std::optional<double>,
               std::optional<double>,
               std::optional<double>
          >(),
          "com"_a = nb::none(),
          "span"_a = nb::none(),
          "halflife"_a = nb::none(),
          "alpha"_a = nb::none()
        )
        .def("__call__", &screamer::functor_call<screamer::EwBeta>)
        .def("reset", &screamer::EwBeta::reset, "Reset to the initial state.");

     // Regression-family EW additions: the EW analogs of RollingSpread /
     // RollingAlpha / RollingResidualStd (target first, regressor second).
     nb::class_<screamer::EwSpread, screamer::EvalOp>(m, "EwSpread")
        .def(
          nb::init<
               std::optional<double>,
               std::optional<double>,
               std::optional<double>,
               std::optional<double>
          >(),
          "com"_a = nb::none(),
          "span"_a = nb::none(),
          "halflife"_a = nb::none(),
          "alpha"_a = nb::none()
        )
        .def("__call__", &screamer::functor_call<screamer::EwSpread>)
        .def("reset", &screamer::EwSpread::reset, "Reset to the initial state.");

     nb::class_<screamer::EwAlpha, screamer::EvalOp>(m, "EwAlpha")
        .def(
          nb::init<
               std::optional<double>,
               std::optional<double>,
               std::optional<double>,
               std::optional<double>
          >(),
          "com"_a = nb::none(),
          "span"_a = nb::none(),
          "halflife"_a = nb::none(),
          "alpha"_a = nb::none()
        )
        .def("__call__", &screamer::functor_call<screamer::EwAlpha>)
        .def("reset", &screamer::EwAlpha::reset, "Reset to the initial state.");

     nb::class_<screamer::EwResidualStd, screamer::EvalOp>(m, "EwResidualStd")
        .def(
          nb::init<
               std::optional<double>,
               std::optional<double>,
               std::optional<double>,
               std::optional<double>
          >(),
          "com"_a = nb::none(),
          "span"_a = nb::none(),
          "halflife"_a = nb::none(),
          "alpha"_a = nb::none()
        )
        .def("__call__", &screamer::functor_call<screamer::EwResidualStd>)
        .def("reset", &screamer::EwResidualStd::reset, "Reset to the initial state.");

}
