#ifndef SCREAMER_DECYCLER_H
#define SCREAMER_DECYCLER_H

// Decycler: Ehlers' trend estimate, the input with cycles at or below
// `period` removed. Equivalent to input minus a 1-pole highpass.
//
//   alpha = (cos(2 pi / period) + sin(2 pi / period) - 1) / cos(2 pi / period)
//   y[t] = (alpha / 2) (x[t] + x[t-1]) + (1 - alpha) y[t-1]
//
// Give exactly one of `period` (samples) or `cutoff` (fraction of
// Nyquist in (0, 1)); period = 2 / cutoff.

#include <optional>
#include <vector>
#include "screamer/common/base.h"
#include "screamer/signal/ehlers.h"
#include "screamer/signal/signal.h"
#include <cstddef>

namespace screamer {

class Decycler : public ScreamerBase {
public:
    explicit Decycler(std::optional<double> period = std::nullopt,
                      std::optional<double> cutoff = std::nullopt) {
        const double p = ehlers_resolve_period(period, cutoff);
        std::vector<double> b, a;
        decycler_coeffs(p, b, a);
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

#endif  // SCREAMER_DECYCLER_H
