#ifndef SCREAMER_BUTTER_BANDSTOP_H
#define SCREAMER_BUTTER_BANDSTOP_H

// ButterBandstop: digital band-stop (notch) Butterworth IIR filter.
// Produces a 2N-order filter from the order-N analog prototype.
//
//   ButterBandstop(order, low_cutoff, high_cutoff)
//
// Cutoffs are fractions of Nyquist in (0, 1). Frequencies in
// [low_cutoff, high_cutoff] are suppressed.

#include <stdexcept>
#include <vector>
#include "screamer/common/base.h"
#include "screamer/signal/butter.h"
#include "screamer/signal/signal.h"

namespace screamer {

class ButterBandstop : public ScreamerBase {
public:
    ButterBandstop(int order, double low_cutoff, double high_cutoff) {
        if (order < 1) throw std::invalid_argument("Order must be at least 1.");
        std::vector<double> b, a;
        butterworth_filter_bs(order, low_cutoff, high_cutoff, b, a);
        iir_.init(b, a);
    }

    void reset() override { iir_.reset(); }

    double process_scalar(double x) override {
        return iir_.process_scalar(x);
    }

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

#endif
