#ifndef SCREAMER_KELTNER_CHANNELS_H
#define SCREAMER_KELTNER_CHANNELS_H

// KeltnerChannels (Chester Keltner; modernised by Linda Bradford
// Raschke). Volatility-adapted price envelope:
//
//     mid[t]   = EMA(close, span)
//     atr[t]   = ATR(high, low, close, window)
//     upper[t] = mid + num_atr * atr
//     lower[t] = mid - num_atr * atr
//
// 3 -> 3 functor over (high, low, close). Returns (lower, mid, upper)
// per step. The first 3->M consumer in screamer (uses the Plan E
// N->M dispatcher).
//
// Composition: holds one EwMean (the EMA on close) and one ATR (the
// Wilder-smoothed True Range). First valid output at sample index
// `window_size` (ATR's warmup; EwMean has no warmup).

#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include "screamer/atr.h"
#include "screamer/common/float_info.h"
#include "screamer/common/functor_base.h"
#include "screamer/ew_mean.h"

namespace screamer {

class KeltnerChannels : public FunctorBase<KeltnerChannels, 3, 3> {
public:
    explicit KeltnerChannels(int window_size = 20, double num_atr = 2.0)
        : num_atr_(num_atr),
          mid_(std::nullopt, static_cast<double>(window_size),
               std::nullopt, std::nullopt),
          atr_(window_size)
    {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be at least 2.");
        }
    }

    void reset() override {
        mid_.reset();
        atr_.reset();
    }

    ResultTuple call(const InputArray& inputs) override {
        const double close = inputs[2];
        const double m = mid_.process_scalar(close);
        const double a = atr_.call(inputs);
        if (isnan2(a)) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            return std::make_tuple(nan, nan, nan);
        }
        return std::make_tuple(m - num_atr_ * a, m, m + num_atr_ * a);
    }

private:
    const double num_atr_;
    EwMean mid_;
    ATR atr_;
};

}  // namespace screamer

#endif
