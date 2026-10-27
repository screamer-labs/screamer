#ifndef SCREAMER_ROOFING_FILTER_H
#define SCREAMER_ROOFING_FILTER_H

// RoofingFilter: Ehlers' bandpass preprocessor. A 2-pole highpass at
// `hp_period` removes the trend, then a SuperSmoother at `lp_period`
// removes aliasing noise, leaving the tradeable cycle band. Chains two
// IIR stages in one node. Reference: J. Ehlers, "Predictive and
// Successful Indicators". Give exactly one of hp_period / hp_cutoff and
// exactly one of lp_period / lp_cutoff.

#include <optional>
#include <cstddef>
#include <string>
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
                  std::optional<double> lp_cutoff = std::nullopt,
                  const std::string& start_policy = "steady") {
        const IIRStart start = parse_iir_start(start_policy, "RoofingFilter");
        const double hp = ehlers_resolve_period(hp_period, hp_cutoff);
        const double lp = ehlers_resolve_period(lp_period, lp_cutoff);
        std::vector<double> b, a;
        ehlers_highpass2_coeffs(hp, b, a);
        hp_.init(b, a, start);
        supersmoother_coeffs(lp, b, a);
        lp_.init(b, a, start);
    }

    void reset() override {
        hp_.reset();
        lp_.reset();
    }

    double process_scalar(double x) override {
        return lp_.process_scalar(hp_.process_scalar(x));
    }

    void process_array_no_stride(double* y, const double* x, size_t size) override {
        run_column(y, 1, x, 1, size);
    }

    void process_array_stride(double* y, size_t dyi, const double* x, size_t dxi,
                              size_t size) override {
        run_column(y, static_cast<std::ptrdiff_t>(dyi), x, static_cast<std::ptrdiff_t>(dxi), size);
    }

private:
    // The batch loop: both second-order filters as IIRStates (state in
    // locals), one step of each per value. A NaN from the high-pass (an
    // infinite input) skips the smoother, as process_scalar does. Other
    // orders cannot occur today; they would take the per-value path.
    void run_column(double* y, std::ptrdiff_t dy, const double* x, std::ptrdiff_t dx,
                    size_t size) {
        if (hp_.n != 3 || lp_.n != 3) {
            for (size_t m = 0; m < size; ++m, x += dx, y += dy) *y = process_scalar(*x);
            return;
        }
        // Values up to the first one both stages have started on take the
        // per-value path, which sets the start states.
        for (; size > 0 && !(hp_.started() && lp_.started()); --size, x += dx, y += dy) {
            *y = process_scalar(*x);
        }
        if (size == 0) return;
        IIRState<2> hp(hp_), lp(lp_);
        skip_nan_column(y, dy, x, dx, size, [&hp, &lp](double v) {
            const double h = hp.step(v);
            return h != h ? h : lp.step(h);
        });
        hp.store(hp_);
        lp.store(lp_);
    }

    IIRFilter hp_;
    IIRFilter lp_;
};

}  // namespace screamer

#endif  // SCREAMER_ROOFING_FILTER_H
