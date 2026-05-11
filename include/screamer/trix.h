#ifndef SCREAMER_TRIX_H
#define SCREAMER_TRIX_H

// TRIX: 1-period rate of change of a triple-smoothed EMA.
//
//     ema1[t] = EwMean(span)(x)[t]
//     ema2[t] = EwMean(span)(ema1)[t]
//     ema3[t] = EwMean(span)(ema2)[t]
//     TRIX[t] = 100 * (ema3[t] - ema3[t-1]) / ema3[t-1]
//
// Pure composition of three EwMean instances plus a 1-step delay
// buffer for the final ratio. O(1) per step.
//
// The underlying EMA is pandas's adjust=True form (matching our
// EwMean and the rest of the codebase). TA-Lib's TRIX uses
// adjust=False with an SMA seed, so our TRIX differs from TA-Lib
// by a few percent during early samples; see docs/conventions.md.

#include <limits>
#include <optional>
#include <stdexcept>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/ew_mean.h"

namespace screamer {

class TRIX : public ScreamerBase {
public:
    explicit TRIX(int span)
        : ema1_(std::nullopt, static_cast<double>(span),
                std::nullopt, std::nullopt),
          ema2_(std::nullopt, static_cast<double>(span),
                std::nullopt, std::nullopt),
          ema3_(std::nullopt, static_cast<double>(span),
                std::nullopt, std::nullopt)
    {
        if (span < 1) {
            throw std::invalid_argument("span must be at least 1.");
        }
    }

    void reset() override {
        ema1_.reset();
        ema2_.reset();
        ema3_.reset();
        prev_ema3_ = std::numeric_limits<double>::quiet_NaN();
    }

    double process_scalar(double x) override {
        const double e1 = ema1_.process_scalar(x);
        const double e2 = ema2_.process_scalar(e1);
        const double e3 = ema3_.process_scalar(e2);

        double trix;
        if (isnan2(prev_ema3_)) {
            trix = std::numeric_limits<double>::quiet_NaN();
        } else {
            trix = 100.0 * (e3 - prev_ema3_) / prev_ema3_;
        }
        prev_ema3_ = e3;
        return trix;
    }

private:
    EwMean ema1_;
    EwMean ema2_;
    EwMean ema3_;
    double prev_ema3_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace screamer

#endif
