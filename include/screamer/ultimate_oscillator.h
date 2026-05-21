#ifndef SCREAMER_ULTIMATE_OSCILLATOR_H
#define SCREAMER_ULTIMATE_OSCILLATOR_H

// UltimateOscillator: Larry Williams (1976). Weighted average of
// the close-to-true-range ratio over three timeframes.
//
//     BP[t] = close - min(low, prev_close)
//     TR[t] = max(high, prev_close) - min(low, prev_close)
//     avg_k = SUM(BP, period_k) / SUM(TR, period_k)
//     UO[t] = 100 * (4*avg1 + 2*avg2 + avg3) / 7
//
// Defaults: (7, 14, 28). 3 -> 1 on (high, low, close).
//
// Composition: prev_close scalar + six detail::RollingSum buffers
// (BP and TR over each of the three periods). O(1) per step. First
// valid output at sample index max(period1, period2, period3).

#include <algorithm>
#include <limits>
#include <stdexcept>
#include "screamer/common/float_info.h"
#include "screamer/common/functor_base.h"
#include "screamer/detail/rolling_sum.h"

namespace screamer {

class UltimateOscillator : public FunctorBase<UltimateOscillator, 3, 1> {
public:
    UltimateOscillator(int period1 = 7, int period2 = 14, int period3 = 28)
        : period1_(period1),
          period2_(period2),
          period3_(period3),
          max_period_(std::max({period1, period2, period3})),
          bp_sum_1_(period1, "expanding"),
          bp_sum_2_(period2, "expanding"),
          bp_sum_3_(period3, "expanding"),
          tr_sum_1_(period1, "expanding"),
          tr_sum_2_(period2, "expanding"),
          tr_sum_3_(period3, "expanding")
    {
        if (period1 < 1 || period2 < 1 || period3 < 1) {
            throw std::invalid_argument("All periods must be positive.");
        }
    }

    void reset() override {
        bp_sum_1_.reset();
        bp_sum_2_.reset();
        bp_sum_3_.reset();
        tr_sum_1_.reset();
        tr_sum_2_.reset();
        tr_sum_3_.reset();
        prev_close_ = std::numeric_limits<double>::quiet_NaN();
        n_seen_ = 0;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double high  = inputs[0];
        const double low   = inputs[1];
        const double close = inputs[2];

        if (isnan2(high) || isnan2(low) || isnan2(close)) {
            // NaN policy "ignore": leave state alone, emit NaN.
            return std::numeric_limits<double>::quiet_NaN();
        }

        // First sample has no prev_close -> BP/TR undefined. Push 0
        // so the rolling-sum buffers stay aligned; will be flushed
        // before the first valid output.
        double bp = 0.0;
        double tr = 0.0;
        if (!isnan2(prev_close_)) {
            const double low_or_prev = std::min(low, prev_close_);
            const double high_or_prev = std::max(high, prev_close_);
            bp = close - low_or_prev;
            tr = high_or_prev - low_or_prev;
        }
        prev_close_ = close;

        const double bp1 = bp_sum_1_.append(bp);
        const double bp2 = bp_sum_2_.append(bp);
        const double bp3 = bp_sum_3_.append(bp);
        const double tr1 = tr_sum_1_.append(tr);
        const double tr2 = tr_sum_2_.append(tr);
        const double tr3 = tr_sum_3_.append(tr);

        // First sample's bp=0/tr=0 placeholder must roll out of all
        // buffers AND we need a full longest-window of real samples,
        // so the first valid output is at sample index max_period_
        // (zero-indexed), i.e. after max_period_ + 1 calls. Matches
        // TA-Lib's ULTOSC convention.
        if (n_seen_ <= max_period_) {
            n_seen_++;
            if (n_seen_ <= max_period_) {
                return std::numeric_limits<double>::quiet_NaN();
            }
        }

        const double avg1 = (tr1 > 0.0) ? bp1 / tr1 : 0.0;
        const double avg2 = (tr2 > 0.0) ? bp2 / tr2 : 0.0;
        const double avg3 = (tr3 > 0.0) ? bp3 / tr3 : 0.0;
        return 100.0 * (4.0 * avg1 + 2.0 * avg2 + avg3) / 7.0;
    }

private:
    const int period1_;
    const int period2_;
    const int period3_;
    const int max_period_;
    detail::RollingSum bp_sum_1_;
    detail::RollingSum bp_sum_2_;
    detail::RollingSum bp_sum_3_;
    detail::RollingSum tr_sum_1_;
    detail::RollingSum tr_sum_2_;
    detail::RollingSum tr_sum_3_;
    double prev_close_ = std::numeric_limits<double>::quiet_NaN();
    int n_seen_ = 0;
};

}  // namespace screamer

#endif
