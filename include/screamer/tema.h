#ifndef SCREAMER_TEMA_H
#define SCREAMER_TEMA_H

// TEMA: Triple Exponential Moving Average. Patrick Mulloy (1994).
//
//     TEMA[t] = 3*EMA(x) - 3*EMA(EMA(x)) + EMA(EMA(EMA(x)))   [at time t]
//
// Pure composition of three chained EwMean instances. Same constructor
// (com / span / halflife / alpha mutex) as EwMean. O(1) per step, no
// warmup (EwMean is well-defined from t=0).

#include <optional>
#include "screamer/common/base.h"
#include "screamer/ew_mean.h"

namespace screamer {

class TEMA : public ScreamerBase {
public:
    explicit TEMA(
        std::optional<double> com = std::nullopt,
        std::optional<double> span = std::nullopt,
        std::optional<double> halflife = std::nullopt,
        std::optional<double> alpha = std::nullopt)
        : ema1_(com, span, halflife, alpha),
          ema2_(com, span, halflife, alpha),
          ema3_(com, span, halflife, alpha)
    {}

    void reset() override {
        ema1_.reset();
        ema2_.reset();
        ema3_.reset();
    }

    double process_scalar(double x) override {
        const double e1 = ema1_.process_scalar(x);
        const double e2 = ema2_.process_scalar(e1);
        const double e3 = ema3_.process_scalar(e2);
        return 3.0 * e1 - 3.0 * e2 + e3;
    }

private:
    EwMean ema1_;
    EwMean ema2_;
    EwMean ema3_;
};

}  // namespace screamer

#endif
