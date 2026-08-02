#ifndef SCREAMER_SUPER_SMOOTHER_H
#define SCREAMER_SUPER_SMOOTHER_H

// SuperSmoother: Ehlers' 2-pole low-lag lowpass filter. A critically
// damped IIR designed to pass cycles longer than `period` samples with
// less lag and less ringing than a Butterworth of the same rolloff.
//
//   a1 = exp(-sqrt(2) pi / period)
//   b1 = 2 a1 cos(sqrt(2) pi / period)
//   y[t] = c1 (x[t] + x[t-1]) / 2 + b1 y[t-1] - a1^2 y[t-2]
//
// where c1 = 1 - b1 + a1^2. Reference: J. Ehlers, "Cycle Analytics for
// Traders". Give exactly one of `period` (samples) or `cutoff`
// (fraction of Nyquist in (0, 1)); period = 2 / cutoff.

#include <optional>
#include <vector>
#include "screamer/common/base.h"
#include "screamer/signal/ehlers.h"
#include "screamer/signal/signal.h"
#include <cstddef>

namespace screamer {

class SuperSmoother : public ScreamerBase {
public:
    explicit SuperSmoother(std::optional<double> period = std::nullopt,
                           std::optional<double> cutoff = std::nullopt) {
        const double p = ehlers_resolve_period(period, cutoff);
        std::vector<double> b, a;
        supersmoother_coeffs(p, b, a);
        iir_.init(b, a);
    }

    void reset() override { iir_.reset(); }

    double process_scalar(double x) override { return iir_.process_scalar(x); }

    void process_array_no_stride(double* y, const double* x, size_t size) override {
        iir_.process_array_no_stride(y, x, size);
    }

    void process_array_stride(double* y, size_t dyi, const double* x,
                              size_t dxi, size_t size) override {
        iir_.process_array_stride(y, dyi, x, dxi, size);
    }

private:
    IIRFilter iir_;
};

}  // namespace screamer

#endif  // SCREAMER_SUPER_SMOOTHER_H
