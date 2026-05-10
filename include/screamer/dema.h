#ifndef SCREAMER_DEMA_H
#define SCREAMER_DEMA_H

// DEMA: Double Exponential Moving Average. Patrick Mulloy (1994).
//
//     DEMA[t] = 2 * EMA(x)[t] - EMA(EMA(x))[t]
//
// Pure composition of two chained EwMean instances; no extra state.
// Same constructor (com / span / halflife / alpha mutex) as EwMean.
// O(1) per step, no warmup (EwMean is well-defined from t=0).

#include <optional>
#include "screamer/common/base.h"
#include "screamer/ew_mean.h"

namespace screamer {

class DEMA : public ScreamerBase {
public:
    explicit DEMA(
        std::optional<double> com = std::nullopt,
        std::optional<double> span = std::nullopt,
        std::optional<double> halflife = std::nullopt,
        std::optional<double> alpha = std::nullopt)
        : ema1_(com, span, halflife, alpha),
          ema2_(com, span, halflife, alpha)
    {}

    void reset() override {
        ema1_.reset();
        ema2_.reset();
    }

    double process_scalar(double x) override {
        const double e1 = ema1_.process_scalar(x);
        const double e2 = ema2_.process_scalar(e1);
        return 2.0 * e1 - e2;
    }

private:
    EwMean ema1_;
    EwMean ema2_;
};

}  // namespace screamer

#endif
