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

#include <limits>
#include <stdexcept>
#include "screamer/common/float_info.h"
#include "screamer/common/functor_base.h"
#include "screamer/true_range.h"

namespace screamer {

class ATR : public FunctorBase<ATR, 3, 1> {
public:
    explicit ATR(int window_size = 14) : window_size_(window_size) {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be at least 2.");
        }
    }

    void reset() override {
        true_range_.reset();
        seed_sum_ = 0.0;
        atr_ = 0.0;
        n_tr_ = 0;
    }

    ResultTuple call(const InputArray& inputs) override {
        // TrueRange owns the TR definition, the previous-close carry, and the
        // `ignore` NaN policy for the bar. It returns NaN at t=0 and for any
        // bar with a missing field, in both cases without advancing state, so
        // the Wilder recursion below only ever sees a usable TR.
        const double tr = true_range_.call(inputs);
        if (isnan2(tr)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

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
    TrueRange true_range_;
    double seed_sum_ = 0.0;
    double atr_ = 0.0;
    int n_tr_ = 0;
};

}  // namespace screamer

#endif
