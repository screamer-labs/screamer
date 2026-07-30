#ifndef SCREAMER_ROOFING_FILTER_H
#define SCREAMER_ROOFING_FILTER_H

// RoofingFilter: Ehlers' bandpass preprocessor. A 2-pole highpass at
// `hp_period` removes the trend, then a SuperSmoother at `lp_period`
// removes aliasing noise, leaving the tradeable cycle band. Chains two
// IIR stages in one node. Reference: J. Ehlers, "Predictive and
// Successful Indicators". Give exactly one of hp_period / hp_cutoff and
// exactly one of lp_period / lp_cutoff.

#include <optional>
#include <vector>
#include "screamer/common/base.h"
#include "screamer/signal/ehlers.h"
#include "screamer/signal/signal.h"

namespace screamer {

class RoofingFilter : public ScreamerBase {
public:
    RoofingFilter(std::optional<double> hp_period = std::nullopt,
                  std::optional<double> lp_period = std::nullopt,
                  std::optional<double> hp_cutoff = std::nullopt,
                  std::optional<double> lp_cutoff = std::nullopt) {
        const double hp = ehlers_resolve_period(hp_period, hp_cutoff);
        const double lp = ehlers_resolve_period(lp_period, lp_cutoff);
        std::vector<double> b, a;
        ehlers_highpass2_coeffs(hp, b, a);
        hp_.init(b, a);
        supersmoother_coeffs(lp, b, a);
        lp_.init(b, a);
    }

    void reset() override {
        hp_.reset();
        lp_.reset();
    }

    double process_scalar(double x) override {
        return lp_.process_scalar(hp_.process_scalar(x));
    }

private:
    IIRFilter hp_;
    IIRFilter lp_;
};

}  // namespace screamer

#endif  // SCREAMER_ROOFING_FILTER_H
