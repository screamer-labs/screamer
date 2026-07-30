#ifndef SCREAMER_MOVING_AVERAGE_H
#define SCREAMER_MOVING_AVERAGE_H

// MovingAverage(taps): finite-impulse-response filter with arbitrary
// coefficients.
//
//     y[t] = sum_{k=0..L-1} taps[k] * x[t - k]
//
// taps[0] is the coefficient on the current sample, taps[L-1] on the
// oldest. Pre-compute the coefficient vector with numpy
// (np.hamming(n) / np.bartlett(n) / np.blackman(n) / np.kaiser(n) /
// scipy.signal.firwin etc.) and pass it in. The user is responsible
// for any normalisation (e.g. taps /= taps.sum() for a unity-gain
// filter).
//
// 1 -> 1 stateless except for the circular buffer of the last
// len(taps) samples held by detail::FirCore. O(L) per step where
// L = len(taps). First valid output at sample index L - 1.

#include <limits>
#include <stdexcept>
#include <vector>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/fir_core.h"

namespace screamer {

class MovingAverage : public ScreamerBase {
public:
    explicit MovingAverage(const std::vector<double>& taps)
        : fir_(validated(taps))
    {}

    void reset() override {
        fir_.reset();
    }

    double process_scalar(double x) override {
        // nan_policy: ignore. The sample is not buffered and does not
        // advance warmup, so the next finite sample continues from the
        // state the filter already had.
        if (isnan2(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        fir_.push(x);
        if (!fir_.warm()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return fir_.convolve();
    }

private:
    // Runs before the member initialiser builds the filter, so invalid
    // taps raise before anything is allocated.
    static const std::vector<double>& validated(const std::vector<double>& taps) {
        if (taps.empty()) {
            throw std::invalid_argument("taps must be non-empty.");
        }
        return taps;
    }

    detail::FirCore fir_;
};

}  // namespace screamer

#endif
