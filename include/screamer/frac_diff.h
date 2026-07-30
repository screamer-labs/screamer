#ifndef SCREAMER_FRAC_DIFF_H
#define SCREAMER_FRAC_DIFF_H

// FracDiff(d, window_size, threshold, start_policy): fractional
// differentiation (Lopez de Prado, Advances in Financial Machine
// Learning, chapter 5).
//
// A fixed-width FIR filter whose taps come from the binomial expansion
// of (1 - B)^d, where B is the backshift operator:
//
//     w_0 = 1,   w_k = -w_{k-1} * (d - k + 1) / k
//     y[t] = sum_{k=0..L-1} w_k * x[t - k]
//
// which is w_k = (-1)^k * binom(d, k). Integer orders recover the
// ordinary differences: d = 0 gives [1] (identity), d = 1 gives
// [1, -1] (Diff(1)), d = 2 gives [1, -2, 1] (Diff2). Fractional orders
// in between decay as a power law rather than terminating, which is
// what preserves memory that an integer difference discards.
//
// The tap count L is the point at which the series is truncated:
// whichever comes first of `window_size` taps and the first weight with
// |w_k| < threshold. L bounds both the memory and the per-step cost,
// and it is the number of samples strict warmup waits for.
//
// Negative d is fractional integration (the weights do not alternate
// and decay more slowly); window_size is what bounds it.

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/fir_core.h"
#include "screamer/detail/start_policy.h"

namespace screamer {

class FracDiff : public ScreamerBase {
public:
    FracDiff(
        double d = 0.4,
        int window_size = 100,
        double threshold = 1e-5,
        const std::string& start_policy = "strict")
        // start_policy_ is declared first so an unknown policy raises
        // before the weight recursion runs.
        : start_policy_(detail::parse_start_policy(start_policy)),
          fir_(frac_weights(d, window_size, threshold))
    {}

    void reset() override {
        fir_.reset();
    }

    double process_scalar(double x) override {
        // nan_policy: ignore. The sample is not buffered and does not
        // advance warmup.
        if (isnan2(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        fir_.push(x);
        // strict withholds output until the filter is fully defined.
        // expanding and zero both emit the truncated convolution, which
        // for a linear filter is the same arithmetic: the samples that
        // have not arrived read as the zeros the buffer starts with.
        if (start_policy_ == detail::StartPolicy::Strict && !fir_.warm()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return fir_.convolve();
    }

private:
    // Runs inside the member initialiser list, so invalid arguments
    // raise before any state is built.
    static std::vector<double> frac_weights(double d, int window_size, double threshold) {
        if (window_size < 1) {
            throw std::invalid_argument("Window size must be positive.");
        }
        if (isnan2(d)) {
            throw std::invalid_argument("d must be a number.");
        }
        // Written as a negated comparison so a NaN threshold is rejected
        // too (every comparison against NaN is false).
        if (!(threshold >= 0.0)) {
            throw std::invalid_argument("threshold must be non-negative.");
        }

        std::vector<double> weights;
        weights.push_back(1.0);
        for (int k = 1; k < window_size; ++k) {
            const double next = -weights.back() * (d - k + 1) / k;
            if (std::abs(next) < threshold) {
                break;
            }
            weights.push_back(next);
        }
        return weights;
    }

    detail::StartPolicy start_policy_;
    detail::FirCore fir_;
};

}  // namespace screamer

#endif
