#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for std::vector
#include "screamer/common/base.h"
#include "screamer/butter.h"
#include "screamer/butter_highpass.h"
#include "screamer/butter_bandpass.h"
#include "screamer/butter_bandstop.h"
#include "screamer/super_smoother.h"
#include "screamer/decycler.h"
#include "screamer/roofing_filter.h"
#include "screamer/moving_average.h"
#include "screamer/kalman_filter.h"
#include "screamer/schmitt_trigger.h"
#include "screamer/hold.h"
#include "screamer/dominant_cycle.h"
#include "screamer/hilbert_phasor.h"
#include "screamer/cycle_phase.h"
#include "screamer/cycle_frequency.h"
#include "screamer/cycle_amplitude.h"
#include "screamer/cycle_sine.h"

namespace py = pybind11;

void init_bindings_signal(py::module& m) {

    py::class_<screamer::Butter, screamer::ScreamerBase>(m, "Butter")
        .def(py::init<int,double>(),  py::arg("order") = 2, py::arg("cutoff_freq") = 0.1)
        .def("__call__", &screamer::Butter::operator(), py::arg("value"))
        .def("reset", &screamer::Butter::reset, "Reset to the initial state.");

    // Butter family extensions: HP / BP / BS. Same scaling convention
    // (cutoff is a fraction of Nyquist in (0, 1)) as the existing
    // low-pass `Butter`.
    py::class_<screamer::ButterHighpass, screamer::ScreamerBase>(m, "ButterHighpass")
        .def(py::init<int, double>(), py::arg("order") = 2, py::arg("cutoff_freq") = 0.1)
        .def("__call__", &screamer::ButterHighpass::operator(), py::arg("value"))
        .def("reset", &screamer::ButterHighpass::reset, "Reset.");

    py::class_<screamer::ButterBandpass, screamer::ScreamerBase>(m, "ButterBandpass")
        .def(py::init<int, double, double>(),
             py::arg("order") = 2, py::arg("low_cutoff") = 0.05, py::arg("high_cutoff") = 0.2)
        .def("__call__", &screamer::ButterBandpass::operator(), py::arg("value"))
        .def("reset", &screamer::ButterBandpass::reset, "Reset.");

    py::class_<screamer::ButterBandstop, screamer::ScreamerBase>(m, "ButterBandstop")
        .def(py::init<int, double, double>(),
             py::arg("order") = 2, py::arg("low_cutoff") = 0.05, py::arg("high_cutoff") = 0.2)
        .def("__call__", &screamer::ButterBandstop::operator(), py::arg("value"))
        .def("reset", &screamer::ButterBandstop::reset, "Reset.");

    // SuperSmoother: Ehlers 2-pole low-lag lowpass. Exactly one of
    // period (samples) or cutoff (fraction of Nyquist).
    py::class_<screamer::SuperSmoother, screamer::ScreamerBase>(m, "SuperSmoother")
        .def(py::init<std::optional<double>, std::optional<double>>(),
             py::arg("period") = py::none(), py::arg("cutoff") = py::none())
        .def("__call__", &screamer::SuperSmoother::operator(), py::arg("value"))
        .def("reset", &screamer::SuperSmoother::reset, "Reset to the initial state.");

    // Decycler: Ehlers trend estimate (input minus a 1-pole highpass).
    py::class_<screamer::Decycler, screamer::ScreamerBase>(m, "Decycler")
        .def(py::init<std::optional<double>, std::optional<double>>(),
             py::arg("period") = py::none(), py::arg("cutoff") = py::none())
        .def("__call__", &screamer::Decycler::operator(), py::arg("value"))
        .def("reset", &screamer::Decycler::reset, "Reset to the initial state.");

    // RoofingFilter: Ehlers bandpass (2-pole highpass then SuperSmoother).
    py::class_<screamer::RoofingFilter, screamer::ScreamerBase>(m, "RoofingFilter")
        .def(py::init<std::optional<double>, std::optional<double>,
                      std::optional<double>, std::optional<double>>(),
             py::arg("hp_period") = py::none(), py::arg("lp_period") = py::none(),
             py::arg("hp_cutoff") = py::none(), py::arg("lp_cutoff") = py::none())
        .def("__call__", &screamer::RoofingFilter::operator(), py::arg("value"))
        .def("reset", &screamer::RoofingFilter::reset, "Reset to the initial state.");

    // MovingAverage: FIR filter with user-supplied taps. Pre-compute
    // taps via numpy / scipy (np.hamming, np.kaiser, scipy.signal.firwin,
    // ...) and pass the coefficient vector in.
    py::class_<screamer::MovingAverage, screamer::ScreamerBase>(m, "MovingAverage")
        .def(py::init<const std::vector<double>&>(), py::arg("taps") = std::vector<double>{0.25, 0.5, 0.25})
        .def("__call__", &screamer::MovingAverage::operator(), py::arg("value"))
        .def("reset", &screamer::MovingAverage::reset, "Reset.");

    // KalmanFilter: scalar 1-D random-walk-with-noise model.
    py::class_<screamer::KalmanFilter, screamer::ScreamerBase>(m, "KalmanFilter")
        .def(py::init<double, double, double, double>(),
             py::arg("process_var") = 0.01,
             py::arg("observation_var") = 1.0,
             py::arg("initial_state") = 0.0,
             py::arg("initial_variance") = 1.0)
        .def("__call__", &screamer::KalmanFilter::operator(), py::arg("value"))
        .def("reset", &screamer::KalmanFilter::reset, "Reset.");

    // SchmittTrigger: hysteresis comparator with latched binary output.
    py::class_<screamer::SchmittTrigger, screamer::ScreamerBase>(m, "SchmittTrigger")
        .def(py::init<double, double, double>(),
             py::arg("lower"), py::arg("upper"), py::arg("initial") = 0.0)
        .def("__call__", &screamer::SchmittTrigger::operator(), py::arg("value"))
        .def("reset", &screamer::SchmittTrigger::reset,
             "Reset to the initial latched state.");

    // Hold: time-latch operator. Latches a nonzero finite input and holds it
    // for n bars total; returns release after the hold expires.
    py::class_<screamer::Hold, screamer::ScreamerBase>(m, "Hold")
        .def(py::init<int, double>(),
             py::arg("n"), py::arg("release") = 0.0)
        .def("__call__", &screamer::Hold::operator(), py::arg("value"))
        .def("reset", &screamer::Hold::reset,
             "Reset to the initial state (remaining=0, held=release).");

    // DominantCycle: 1->1 dominant cycle period (samples) via Ehlers'
    // homodyne discriminator. See detail/hilbert_cycle.h.
    py::class_<screamer::DominantCycle, screamer::EvalOp>(m, "DominantCycle")
        .def(py::init<>())
        .def("__call__", &screamer::DominantCycle::handle_input)
        .def("reset", &screamer::DominantCycle::reset, "Reset to the initial state.");

    // HilbertPhasor: 1->2 in-phase / quadrature components of the analytic
    // signal via Ehlers' Hilbert transform. See detail/hilbert_cycle.h.
    py::class_<screamer::HilbertPhasor, screamer::EvalOp>(m, "HilbertPhasor")
        .def(py::init<>())
        .def("__call__", &screamer::HilbertPhasor::handle_input)
        .def("reset", &screamer::HilbertPhasor::reset, "Reset to the initial state.");

    // CyclePhase: 1->1 instantaneous phase (degrees, 0..360) of the analytic
    // signal via Ehlers' homodyne discriminator. See detail/hilbert_cycle.h.
    py::class_<screamer::CyclePhase, screamer::EvalOp>(m, "CyclePhase")
        .def(py::init<>())
        .def("__call__", &screamer::CyclePhase::handle_input)
        .def("reset", &screamer::CyclePhase::reset, "Reset to the initial state.");

    // CycleFrequency: 1->1 instantaneous frequency (cycles per sample), the
    // reciprocal of the dominant cycle period. See detail/hilbert_cycle.h.
    py::class_<screamer::CycleFrequency, screamer::EvalOp>(m, "CycleFrequency")
        .def(py::init<>())
        .def("__call__", &screamer::CycleFrequency::handle_input)
        .def("reset", &screamer::CycleFrequency::reset, "Reset to the initial state.");

    // CycleAmplitude: 1->1 instantaneous amplitude (envelope) of the analytic
    // signal, sqrt(I^2 + Q^2). See detail/hilbert_cycle.h.
    py::class_<screamer::CycleAmplitude, screamer::EvalOp>(m, "CycleAmplitude")
        .def(py::init<>())
        .def("__call__", &screamer::CycleAmplitude::handle_input)
        .def("reset", &screamer::CycleAmplitude::reset, "Reset to the initial state.");

    // CycleSine: 1->2 sinewave indicator (sine, leadsine) = sin(phase) and
    // sin(phase + 45 degrees), from the instantaneous phase of the analytic
    // signal. See detail/hilbert_cycle.h.
    py::class_<screamer::CycleSine, screamer::EvalOp>(m, "CycleSine")
        .def(py::init<>())
        .def("__call__", &screamer::CycleSine::handle_input)
        .def("reset", &screamer::CycleSine::reset, "Reset to the initial state.");
}
