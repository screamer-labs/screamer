#ifndef SCREAMER_ADOSC_H
#define SCREAMER_ADOSC_H

// ADOSC: Chaikin Accumulation/Distribution Oscillator (Marc Chaikin).
// Difference of fast and slow EMAs of the A/D line:
//
//     ADOSC[t] = EMA(AD, fast) - EMA(AD, slow)
//
// 4 -> 1 over (high, low, close, volume). TA-Lib defaults: fast=3,
// slow=10. The underlying EMA is screamer.EwMean (pandas adjust=True);
// TA-Lib uses adjust=False, so this class inherits the same documented
// divergence as DEMA / TEMA / MACD.

#include <optional>
#include <stdexcept>
#include "screamer/ad.h"
#include "screamer/common/functor_base.h"
#include "screamer/ew_mean.h"

namespace screamer {

class ADOSC : public FunctorBase<ADOSC, 4, 1> {
public:
    explicit ADOSC(int fast = 3, int slow = 10)
        : ema_fast_(std::nullopt, static_cast<double>(fast),
                    std::nullopt, std::nullopt),
          ema_slow_(std::nullopt, static_cast<double>(slow),
                    std::nullopt, std::nullopt)
    {
        if (fast < 1 || slow < 1) {
            throw std::invalid_argument("fast and slow must be positive.");
        }
        if (fast >= slow) {
            throw std::invalid_argument("fast must be strictly less than slow.");
        }
    }

    void reset() override {
        ad_.reset();
        ema_fast_.reset();
        ema_slow_.reset();
    }

    ResultTuple call(const InputArray& inputs) override {
        const double ad_val = ad_.call(inputs);
        return ema_fast_.process_scalar(ad_val) - ema_slow_.process_scalar(ad_val);
    }

private:
    AD ad_;
    EwMean ema_fast_;
    EwMean ema_slow_;
};

}  // namespace screamer

#endif
