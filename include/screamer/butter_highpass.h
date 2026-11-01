#ifndef SCREAMER_BUTTER_HIGHPASS_H
#define SCREAMER_BUTTER_HIGHPASS_H

// ButterHighpass: digital high-pass Butterworth IIR filter, designed
// via the standard analog-prototype + lp2hp + bilinear pipeline.
//
// `cutoff_freq` is the normalised cutoff in (0, 1), where 1 is the
// Nyquist frequency. Same scaling as `Butter` (the low-pass class)
// and as scipy.signal.butter with Wn fraction-of-Nyquist convention.

#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/common/base.h"
#include "screamer/signal/butter.h"
#include "screamer/signal/signal.h"
#include <cstddef>

namespace screamer {

class ButterHighpass : public serial::Saveable<ButterHighpass, ScreamerBase> {
public:
    static constexpr const char* kTypeName = "ButterHighpass";
    static constexpr int kStateVersion = 1;

    template <class Ar>
    void settings(Ar& ar) {
        ar("order", order_);
        ar("cutoff_freq", cutoff_freq_);
        ar("start_policy", std::string(iir_.start_policy_name()));
    }

    template <class Ar>
    void state(Ar& ar) { ar.object("iir", iir_); }

    ButterHighpass(int order, double cutoff_freq,
                   const std::string& start_policy = "steady")
        : order_(order), cutoff_freq_(cutoff_freq) {
        const IIRStart start = parse_iir_start(start_policy, "ButterHighpass");
        if (order < 1) throw std::invalid_argument("Order must be at least 1.");
        if (!(cutoff_freq > 0.0 && cutoff_freq < 1.0)) {
            throw std::invalid_argument("cutoff_freq must be in (0, 1).");
        }
        std::vector<double> b, a;
        butterworth_filter_hp(order, cutoff_freq, b, a);
        iir_.init(b, a, start);
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
    int order_;
    double cutoff_freq_;
    IIRFilter iir_;
};

}  // namespace screamer

#endif
