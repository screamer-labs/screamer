#ifndef SCREAMER_MFI_H
#define SCREAMER_MFI_H

// MFI: Money Flow Index (Avrum Soudack & Gene Quong, 1989). Volume-
// weighted analogue of RSI on the typical price:
//
//     TP[t]      = (high + low + close) / 3
//     raw_MF[t]  = TP * volume
//     pos_MF     = sum over the window of raw_MF where TP > prev_TP
//     neg_MF     = sum over the window of raw_MF where TP < prev_TP
//     MFI[t]     = 100 * pos_MF / (pos_MF + neg_MF)
//
// 4 -> 1 over (high, low, close, volume). First valid output at
// sample index `window_size` (need one prev_TP plus a full window of
// directional money-flow values). Matches TA-Lib's MFI.

#include <limits>
#include <stdexcept>
#include "screamer/common/float_info.h"
#include "screamer/common/functor_base.h"
#include "screamer/detail/rolling_sum.h"

namespace screamer {

class MFI : public FunctorBase<MFI, 4, 1> {
public:
    explicit MFI(int window_size = 14)
        : window_size_(window_size),
          pos_mf_(window_size, "expanding"),
          neg_mf_(window_size, "expanding")
    {
        if (window_size < 1) {
            throw std::invalid_argument("Window size must be positive.");
        }
    }

    void reset() override {
        pos_mf_.reset();
        neg_mf_.reset();
        prev_tp_ = std::numeric_limits<double>::quiet_NaN();
        n_directional_ = 0;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double high   = inputs[0];
        const double low    = inputs[1];
        const double close  = inputs[2];
        const double volume = inputs[3];
        if (isnan2(high) || isnan2(low) || isnan2(close) || isnan2(volume)) {
            // NaN policy "ignore": leave state alone.
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double tp = (high + low + close) / 3.0;
        const double mf = tp * volume;

        if (isnan2(prev_tp_)) {
            // First bar: cannot direct money flow yet.
            prev_tp_ = tp;
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double pos = (tp > prev_tp_) ? mf : 0.0;
        const double neg = (tp < prev_tp_) ? mf : 0.0;
        prev_tp_ = tp;

        const double pos_sum = pos_mf_.append(pos);
        const double neg_sum = neg_mf_.append(neg);
        n_directional_++;

        if (n_directional_ < window_size_) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double total = pos_sum + neg_sum;
        if (total <= 0.0) {
            return 100.0;
        }
        return 100.0 * pos_sum / total;
    }

private:
    const int window_size_;
    screamer::detail::RollingSum pos_mf_;
    screamer::detail::RollingSum neg_mf_;
    double prev_tp_ = std::numeric_limits<double>::quiet_NaN();
    int n_directional_ = 0;
};

}  // namespace screamer

#endif
