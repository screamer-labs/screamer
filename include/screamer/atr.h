#ifndef SCREAMER_ATR_H
#define SCREAMER_ATR_H

// ATR: Average True Range (J. Welles Wilder Jr. 1978). Wilder-smoothed
// rolling average of TrueRange:
//
//     TR[t]   = max(H - L, |H - C[t-1]|, |L - C[t-1]|)
//     ATR[w]  = (1/w) * sum_{i=1..w} TR[i]              (SMA seed)
//     ATR[t]  = ((w - 1) * ATR[t-1] + TR[t]) / w        (t > w)
//
// 3 -> 1 over (high, low, close). First valid output at sample
// index `window_size` (zero-indexed), matching TA-Lib's ATR.

#include <cmath>
#include <limits>
#include <stdexcept>
#include "screamer/common/float_info.h"
#include "screamer/common/functor_base.h"

namespace screamer {

class ATR : public FunctorBase<ATR, 3, 1> {
public:
    explicit ATR(int window_size = 14) : window_size_(window_size) {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be at least 2.");
        }
    }

    void reset() override {
        prev_close_ = std::numeric_limits<double>::quiet_NaN();
        seed_sum_ = 0.0;
        atr_ = 0.0;
        n_tr_ = 0;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double high  = inputs[0];
        const double low   = inputs[1];
        const double close = inputs[2];

        // Compute TR for this bar (TR is undefined at t=0).
        if (isnan2(prev_close_)) {
            prev_close_ = close;
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double tr = std::max({
            high - low,
            std::abs(high - prev_close_),
            std::abs(low  - prev_close_),
        });
        prev_close_ = close;

        n_tr_++;
        if (n_tr_ < window_size_) {
            // Warmup: accumulate TR sum.
            seed_sum_ += tr;
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (n_tr_ == window_size_) {
            // First valid output: seed ATR with the SMA of the first
            // `window_size` true ranges.
            seed_sum_ += tr;
            atr_ = seed_sum_ / window_size_;
        } else {
            // Wilder smoothing.
            const double w = static_cast<double>(window_size_);
            atr_ = ((w - 1.0) * atr_ + tr) / w;
        }
        return atr_;
    }

private:
    const int window_size_;
    double prev_close_ = std::numeric_limits<double>::quiet_NaN();
    double seed_sum_ = 0.0;
    double atr_ = 0.0;
    int n_tr_ = 0;
};

}  // namespace screamer

#endif
