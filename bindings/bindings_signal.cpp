#include <optional>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>   // std::optional ctor args
#include <nanobind/stl/vector.h>     // std::vector<double> taps
#include <nanobind/stl/string.h>     // std::string ctor arg
#include "screamer/common/base.h"
#include "screamer/common/dispatch.h"
#include "screamer/butter.h"
#include "screamer/butter_highpass.h"
#include "screamer/butter_bandpass.h"
#include "screamer/butter_bandstop.h"
#include "screamer/super_smoother.h"
#include "screamer/decycler.h"
#include "screamer/roofing_filter.h"
#include "screamer/moving_average.h"
#include "screamer/frac_diff.h"
#include "screamer/kalman_filter.h"
#include "screamer/schmitt_trigger.h"
#include "screamer/hold.h"
#include "screamer/dominant_cycle.h"
#include "screamer/hilbert_phasor.h"
#include "screamer/cycle_phase.h"
#include "screamer/cycle_frequency.h"
#include "screamer/cycle_amplitude.h"
#include "screamer/cycle_sine.h"
#include "screamer/trend_mode.h"
#include "screamer/mama.h"
#include "screamer/instantaneous_trendline.h"
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

void init_bindings_signal(nb::module_& m) {

    nb::class_<screamer::Butter, screamer::ScreamerBase>(m, "Butter")
        .def(nb::init<int,double>(),  "order"_a = 2, "cutoff_freq"_a = 0.1)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Butter::reset, "Reset to the initial state.");

    // Butter family extensions: HP / BP / BS. Same scaling convention
    // (cutoff is a fraction of Nyquist in (0, 1)) as the existing
    // low-pass `Butter`.
    nb::class_<screamer::ButterHighpass, screamer::ScreamerBase>(m, "ButterHighpass")
        .def(nb::init<int, double>(), "order"_a = 2, "cutoff_freq"_a = 0.1)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::ButterHighpass::reset, "Reset.");

    nb::class_<screamer::ButterBandpass, screamer::ScreamerBase>(m, "ButterBandpass")
        .def(nb::init<int, double, double>(),
             "order"_a = 2, "low_cutoff"_a = 0.05, "high_cutoff"_a = 0.2)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::ButterBandpass::reset, "Reset.");

    nb::class_<screamer::ButterBandstop, screamer::ScreamerBase>(m, "ButterBandstop")
        .def(nb::init<int, double, double>(),
             "order"_a = 2, "low_cutoff"_a = 0.05, "high_cutoff"_a = 0.2)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::ButterBandstop::reset, "Reset.");

    // SuperSmoother: Ehlers 2-pole low-lag lowpass. Exactly one of
    // period (samples) or cutoff (fraction of Nyquist).
    nb::class_<screamer::SuperSmoother, screamer::ScreamerBase>(m, "SuperSmoother")
        .def(nb::init<std::optional<double>, std::optional<double>>(),
             "period"_a = nb::none(), "cutoff"_a = nb::none())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::SuperSmoother::reset, "Reset to the initial state.");

    // Decycler: Ehlers trend estimate (input minus a 1-pole highpass).
    nb::class_<screamer::Decycler, screamer::ScreamerBase>(m, "Decycler")
        .def(nb::init<std::optional<double>, std::optional<double>>(),
             "period"_a = nb::none(), "cutoff"_a = nb::none())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Decycler::reset, "Reset to the initial state.");

    // RoofingFilter: Ehlers bandpass (2-pole highpass then SuperSmoother).
    nb::class_<screamer::RoofingFilter, screamer::ScreamerBase>(m, "RoofingFilter")
        .def(nb::init<std::optional<double>, std::optional<double>,
                      std::optional<double>, std::optional<double>>(),
             "hp_period"_a = nb::none(), "lp_period"_a = nb::none(),
             "hp_cutoff"_a = nb::none(), "lp_cutoff"_a = nb::none())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RoofingFilter::reset, "Reset to the initial state.");

    // MovingAverage: FIR filter with user-supplied taps. Pre-compute
    // taps via numpy / scipy (np.hamming, np.kaiser, scipy.signal.firwin,
    // ...) and pass the coefficient vector in.
    nb::class_<screamer::MovingAverage, screamer::ScreamerBase>(m, "MovingAverage")
        .def(nb::init<const std::vector<double>&>(), "taps"_a = std::vector<double>{0.25, 0.5, 0.25})
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::MovingAverage::reset, "Reset.");

    // FracDiff: fractional differentiation (Lopez de Prado, AFML ch. 5).
    // FIR filter with taps w_k = (-1)^k * binom(d, k), truncated at
    // window_size taps or at the first |w_k| < threshold.
    nb::class_<screamer::FracDiff, screamer::ScreamerBase>(m, "FracDiff")
        .def(nb::init<double, int, double, const std::string&>(),
             "d"_a = 0.4,
             "window_size"_a = 100,
             "threshold"_a = 1e-5,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::FracDiff::reset, "Reset to the initial state.");

    // KalmanFilter: scalar 1-D random-walk-with-noise model.
    nb::class_<screamer::KalmanFilter, screamer::ScreamerBase>(m, "KalmanFilter")
        .def(nb::init<double, double, double, double>(),
             "process_var"_a = 0.01,
             "observation_var"_a = 1.0,
             "initial_state"_a = 0.0,
             "initial_variance"_a = 1.0)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::KalmanFilter::reset, "Reset.");

    // SchmittTrigger: hysteresis comparator with latched binary output.
    nb::class_<screamer::SchmittTrigger, screamer::ScreamerBase>(m, "SchmittTrigger")
        .def(nb::init<double, double, double>(),
             "lower"_a, "upper"_a, "initial"_a = 0.0)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::SchmittTrigger::reset,
             "Reset to the initial latched state.");

    // Hold: time-latch operator. Latches a nonzero finite input and holds it
    // for n bars total; returns release after the hold expires.
    nb::class_<screamer::Hold, screamer::ScreamerBase>(m, "Hold")
        .def(nb::init<int, double>(),
             "n"_a, "release"_a = 0.0)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Hold::reset,
             "Reset to the initial state (remaining=0, held=release).");

    // DominantCycle: 1->1 dominant cycle period (samples) via Ehlers'
    // homodyne discriminator. See detail/hilbert_cycle.h.
    nb::class_<screamer::DominantCycle, screamer::EvalOp>(m, "DominantCycle")
        .def(nb::init<>())
        .def("__call__", &screamer::functor_call<screamer::DominantCycle>)
        .def("reset", &screamer::DominantCycle::reset, "Reset to the initial state.");

    // HilbertPhasor: 1->2 in-phase / quadrature components of the analytic
    // signal via Ehlers' Hilbert transform. See detail/hilbert_cycle.h.
    nb::class_<screamer::HilbertPhasor, screamer::EvalOp>(m, "HilbertPhasor")
        .def(nb::init<>())
        .def("__call__", &screamer::functor_call<screamer::HilbertPhasor>)
        .def("reset", &screamer::HilbertPhasor::reset, "Reset to the initial state.");

    // CyclePhase: 1->1 instantaneous phase (degrees, 0..360) of the analytic
    // signal via Ehlers' homodyne discriminator. See detail/hilbert_cycle.h.
    nb::class_<screamer::CyclePhase, screamer::EvalOp>(m, "CyclePhase")
        .def(nb::init<>())
        .def("__call__", &screamer::functor_call<screamer::CyclePhase>)
        .def("reset", &screamer::CyclePhase::reset, "Reset to the initial state.");

    // CycleFrequency: 1->1 instantaneous frequency (cycles per sample), the
    // reciprocal of the dominant cycle period. See detail/hilbert_cycle.h.
    nb::class_<screamer::CycleFrequency, screamer::EvalOp>(m, "CycleFrequency")
        .def(nb::init<>())
        .def("__call__", &screamer::functor_call<screamer::CycleFrequency>)
        .def("reset", &screamer::CycleFrequency::reset, "Reset to the initial state.");

    // CycleAmplitude: 1->1 instantaneous amplitude (envelope) of the analytic
    // signal, sqrt(I^2 + Q^2). See detail/hilbert_cycle.h.
    nb::class_<screamer::CycleAmplitude, screamer::EvalOp>(m, "CycleAmplitude")
        .def(nb::init<>())
        .def("__call__", &screamer::functor_call<screamer::CycleAmplitude>)
        .def("reset", &screamer::CycleAmplitude::reset, "Reset to the initial state.");

    // CycleSine: 1->2 sinewave indicator (sine, leadsine) = sin(phase) and
    // sin(phase + 45 degrees), from the instantaneous phase of the analytic
    // signal. See detail/hilbert_cycle.h.
    nb::class_<screamer::CycleSine, screamer::EvalOp>(m, "CycleSine")
        .def(nb::init<>())
        .def("__call__", &screamer::functor_call<screamer::CycleSine>)
        .def("reset", &screamer::CycleSine::reset, "Reset to the initial state.");

    // TrendMode: 1->1 trend-vs-cycle classifier. Outputs 1.0 when the
    // dominant-cycle phase advance per sample is a small fraction
    // (phase_rate_frac) of a full cycle's expected advance (trending), 0.0
    // when the phase rotates at the cycle rate (cycling). See
    // detail/hilbert_cycle.h.
    nb::class_<screamer::TrendMode, screamer::EvalOp>(m, "TrendMode")
        .def(nb::init<double>(), "phase_rate_frac"_a = 0.5)
        .def("__call__", &screamer::functor_call<screamer::TrendMode>)
        .def("reset", &screamer::TrendMode::reset, "Reset to the initial state.");

    // MAMA: 1->2 MESA Adaptive Moving Average (mama, fama). The smoothing
    // factor adapts to the instantaneous-phase rate of change. See
    // detail/hilbert_cycle.h.
    nb::class_<screamer::MAMA, screamer::EvalOp>(m, "MAMA")
        .def(nb::init<double, double>(),
             "fast_limit"_a = 0.5, "slow_limit"_a = 0.05)
        .def("__call__", &screamer::functor_call<screamer::MAMA>)
        .def("reset", &screamer::MAMA::reset, "Reset to the initial state.");

    // InstantaneousTrendline: 1->1 Ehlers adaptive 2-pole trendline. The
    // smoothing factor is set from the measured dominant cycle period, so
    // the trendline follows the trend and removes the dominant cycle. See
    // detail/hilbert_cycle.h.
    nb::class_<screamer::InstantaneousTrendline, screamer::EvalOp>(m, "InstantaneousTrendline")
        .def(nb::init<>())
        .def("__call__", &screamer::functor_call<screamer::InstantaneousTrendline>)
        .def("reset", &screamer::InstantaneousTrendline::reset, "Reset to the initial state.");
}
