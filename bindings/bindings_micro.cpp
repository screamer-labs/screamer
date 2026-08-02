#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "screamer/common/dispatch.h"
#include "screamer/hawkes_intensity.h"
#include "screamer/ofi.h"
#include "screamer/tick_rule_sign.h"
#include "screamer/lee_ready_sign.h"
#include "screamer/amihud_illiquidity.h"
#include "screamer/bulk_volume_classifier.h"
#include "screamer/roll_spread.h"
#include "screamer/propagator.h"
#include "screamer/vpin.h"
#include "screamer/micro_price.h"
#include "screamer/cont_ofi.h"
#include "screamer/effective_spread.h"
#include "screamer/realized_spread.h"
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

// Microstructure and order-flow operators.
void init_bindings_micro(nb::module_& m) {

    nb::class_<screamer::HawkesIntensity, screamer::ScreamerBase>(m, "HawkesIntensity")
        .def(nb::init<double, double, double>(),
             "decay"_a = 0.9, "alpha"_a = 1.0, "mu"_a = 0.0)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::HawkesIntensity::reset, "Reset to the initial state.");

    nb::class_<screamer::OFI, screamer::EvalOp>(m, "OFI")
        .def(nb::init<>())
        .def("__call__", &screamer::functor_call<screamer::OFI>)
        .def("reset", &screamer::OFI::reset, "Reset to the initial state.");

    nb::class_<screamer::TickRuleSign, screamer::ScreamerBase>(m, "TickRuleSign")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::TickRuleSign::reset, "Reset to the initial state.");

    nb::class_<screamer::LeeReadySign, screamer::EvalOp>(m, "LeeReadySign")
        .def(nb::init<>())
        .def("__call__", &screamer::functor_call<screamer::LeeReadySign>)
        .def("reset", &screamer::LeeReadySign::reset, "Reset to the initial state.");

    nb::class_<screamer::AmihudIlliquidity, screamer::EvalOp>(m, "AmihudIlliquidity")
        .def(nb::init<int, const std::string&>(),
             "window_size"_a = 20, "start_policy"_a = "strict")
        .def("__call__", &screamer::functor_call<screamer::AmihudIlliquidity>)
        .def("reset", &screamer::AmihudIlliquidity::reset, "Reset to the initial state.");

    nb::class_<screamer::BulkVolumeClassifier, screamer::ScreamerBase>(m, "BulkVolumeClassifier")
        .def(nb::init<int, const std::string&>(),
             "window_size"_a = 20, "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::BulkVolumeClassifier::reset, "Reset to the initial state.");

    nb::class_<screamer::RollSpread, screamer::ScreamerBase>(m, "RollSpread")
        .def(nb::init<int, const std::string&>(),
             "window_size"_a = 20, "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollSpread::reset, "Reset to the initial state.");

    nb::class_<screamer::Propagator, screamer::ScreamerBase>(m, "Propagator")
        .def(nb::init<int, double, double>(),
             "window_size"_a = 20, "g0"_a = 1.0, "gamma"_a = 0.5)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Propagator::reset, "Reset to the initial state.");

    nb::class_<screamer::VPIN, screamer::EvalOp>(m, "VPIN")
        .def(nb::init<double, int>(),
             "bucket_volume"_a = 1.0, "n_buckets"_a = 50)
        .def("__call__", &screamer::functor_call<screamer::VPIN>)
        .def("reset", &screamer::VPIN::reset, "Reset to the initial state.");

    nb::class_<screamer::MicroPrice, screamer::EvalOp>(m, "MicroPrice")
        .def(nb::init<>())
        .def("__call__", &screamer::functor_call<screamer::MicroPrice>)
        .def("reset", &screamer::MicroPrice::reset, "Reset to the initial state.");

    nb::class_<screamer::ContOFI, screamer::EvalOp>(m, "ContOFI")
        .def(nb::init<>())
        .def("__call__", &screamer::functor_call<screamer::ContOFI>)
        .def("reset", &screamer::ContOFI::reset, "Reset to the initial state.");

    nb::class_<screamer::EffectiveSpread, screamer::EvalOp>(m, "EffectiveSpread")
        .def(nb::init<>())
        .def("__call__", &screamer::functor_call<screamer::EffectiveSpread>)
        .def("reset", &screamer::EffectiveSpread::reset, "Reset to the initial state.");

    nb::class_<screamer::RealizedSpread, screamer::EvalOp>(m, "RealizedSpread")
        .def(nb::init<int>(), "lag"_a = 1)
        .def("__call__", &screamer::functor_call<screamer::RealizedSpread>)
        .def("reset", &screamer::RealizedSpread::reset, "Reset to the initial state.");

}
