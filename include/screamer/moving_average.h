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
// 1 -> 1 stateless except for an internal circular buffer of the
// last len(taps) samples. O(L) per step where L = len(taps). First
// valid output at sample index L - 1.

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
#include "screamer/common/base.h"

namespace screamer {

class MovingAverage : public ScreamerBase {
public:
    explicit MovingAverage(const std::vector<double>& taps)
        : taps_(taps)
    {
        if (taps_.empty()) {
            throw std::invalid_argument("taps must be non-empty.");
        }
        buffer_.assign(taps_.size(), 0.0);
        reset();
    }

    void reset() override {
        std::fill(buffer_.begin(), buffer_.end(), 0.0);
        index_ = 0;
        n_seen_ = 0;
    }

    double process_scalar(double x) override {
        buffer_[index_] = x;
        index_++;
        if (index_ == buffer_.size()) index_ = 0;

        if (n_seen_ < static_cast<int>(taps_.size())) {
            n_seen_++;
            if (n_seen_ < static_cast<int>(taps_.size())) {
                return std::numeric_limits<double>::quiet_NaN();
            }
        }

        // Convolve: most recent sample is at buffer_[(index_ - 1) mod L].
        // Iterate in time order from newest to oldest matching taps_.
        const int L = static_cast<int>(taps_.size());
        double acc = 0.0;
        int idx = (index_ - 1 + L) % L;
        for (int k = 0; k < L; ++k) {
            acc += taps_[k] * buffer_[idx];
            idx = (idx - 1 + L) % L;
        }
        return acc;
    }

private:
    const std::vector<double> taps_;
    std::vector<double> buffer_;
    int index_ = 0;
    int n_seen_ = 0;
};

}  // namespace screamer

#endif
