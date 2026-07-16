#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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

namespace py = pybind11;

// Microstructure and order-flow operators.
void init_bindings_micro(py::module& m) {

    py::class_<screamer::HawkesIntensity, screamer::ScreamerBase>(m, "HawkesIntensity")
        .def(py::init<double, double, double>(),
             py::arg("decay") = 0.9, py::arg("alpha") = 1.0, py::arg("mu") = 0.0)
        .def("__call__", &screamer::HawkesIntensity::operator(), py::arg("value"))
        .def("reset", &screamer::HawkesIntensity::reset, "Reset to the initial state.");

    py::class_<screamer::OFI, screamer::EvalOp>(m, "OFI")
        .def(py::init<>())
        .def("__call__", &screamer::OFI::handle_input)
        .def("reset", &screamer::OFI::reset, "Reset to the initial state.");

    py::class_<screamer::TickRuleSign, screamer::ScreamerBase>(m, "TickRuleSign")
        .def(py::init<>())
        .def("__call__", &screamer::TickRuleSign::operator(), py::arg("value"))
        .def("reset", &screamer::TickRuleSign::reset, "Reset to the initial state.");

    py::class_<screamer::LeeReadySign, screamer::EvalOp>(m, "LeeReadySign")
        .def(py::init<>())
        .def("__call__", &screamer::LeeReadySign::handle_input)
        .def("reset", &screamer::LeeReadySign::reset, "Reset to the initial state.");

    py::class_<screamer::AmihudIlliquidity, screamer::EvalOp>(m, "AmihudIlliquidity")
        .def(py::init<int, const std::string&>(),
             py::arg("window_size") = 20, py::arg("start_policy") = "strict")
        .def("__call__", &screamer::AmihudIlliquidity::handle_input)
        .def("reset", &screamer::AmihudIlliquidity::reset, "Reset to the initial state.");

    py::class_<screamer::BulkVolumeClassifier, screamer::ScreamerBase>(m, "BulkVolumeClassifier")
        .def(py::init<int, const std::string&>(),
             py::arg("window_size") = 20, py::arg("start_policy") = "strict")
        .def("__call__", &screamer::BulkVolumeClassifier::operator(), py::arg("value"))
        .def("reset", &screamer::BulkVolumeClassifier::reset, "Reset to the initial state.");

    py::class_<screamer::RollSpread, screamer::ScreamerBase>(m, "RollSpread")
        .def(py::init<int, const std::string&>(),
             py::arg("window_size") = 20, py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollSpread::operator(), py::arg("value"))
        .def("reset", &screamer::RollSpread::reset, "Reset to the initial state.");

    py::class_<screamer::Propagator, screamer::ScreamerBase>(m, "Propagator")
        .def(py::init<int, double, double>(),
             py::arg("window") = 20, py::arg("g0") = 1.0, py::arg("gamma") = 0.5)
        .def("__call__", &screamer::Propagator::operator(), py::arg("value"))
        .def("reset", &screamer::Propagator::reset, "Reset to the initial state.");

    py::class_<screamer::VPIN, screamer::EvalOp>(m, "VPIN")
        .def(py::init<double, int>(),
             py::arg("bucket_volume") = 1.0, py::arg("n_buckets") = 50)
        .def("__call__", &screamer::VPIN::handle_input)
        .def("reset", &screamer::VPIN::reset, "Reset to the initial state.");

    py::class_<screamer::MicroPrice, screamer::EvalOp>(m, "MicroPrice")
        .def(py::init<>())
        .def("__call__", &screamer::MicroPrice::handle_input)
        .def("reset", &screamer::MicroPrice::reset, "Reset to the initial state.");

    py::class_<screamer::ContOFI, screamer::EvalOp>(m, "ContOFI")
        .def(py::init<>())
        .def("__call__", &screamer::ContOFI::handle_input)
        .def("reset", &screamer::ContOFI::reset, "Reset to the initial state.");

    py::class_<screamer::EffectiveSpread, screamer::EvalOp>(m, "EffectiveSpread")
        .def(py::init<>())
        .def("__call__", &screamer::EffectiveSpread::handle_input)
        .def("reset", &screamer::EffectiveSpread::reset, "Reset to the initial state.");

    py::class_<screamer::RealizedSpread, screamer::EvalOp>(m, "RealizedSpread")
        .def(py::init<int>(), py::arg("lag") = 1)
        .def("__call__", &screamer::RealizedSpread::handle_input)
        .def("reset", &screamer::RealizedSpread::reset, "Reset to the initial state.");

}
