#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for std::vector
#include "screamer/common/base.h"
#include "screamer/butter.h"
#include "screamer/butter_highpass.h"
#include "screamer/butter_bandpass.h"
#include "screamer/butter_bandstop.h"
#include "screamer/moving_average.h"
#include "screamer/kalman_filter.h"
#include "screamer/schmitt_trigger.h"

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
}
