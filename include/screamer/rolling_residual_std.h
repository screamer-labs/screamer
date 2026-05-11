#ifndef SCREAMER_ROLLING_RESIDUAL_STD_H
#define SCREAMER_ROLLING_RESIDUAL_STD_H

// RollingResidualStd: rolling std of the per-bar spread x - beta*y.
//
// Useful for pairs-trading normalisation -- the z-score of a current
// spread is (spread - rolling_mean(spread)) / RollingResidualStd.
//
// Convention matches RollingBeta / RollingSpread: FIRST argument is
// the dependent (target), SECOND is the regressor. Composes
// RollingSpread + RollingStd. O(1) per step. ddof=1, matching the
// rest of the library.

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/functor_base.h"
#include "screamer/rolling_spread.h"
#include "screamer/rolling_std.h"

namespace screamer {

class RollingResidualStd : public FunctorBase<RollingResidualStd, 2, 1> {
public:
    RollingResidualStd(int window_size, const std::string& start_policy = "strict")
        : spread_(window_size, start_policy),
          std_(window_size, start_policy)
    {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be at least 2.");
        }
    }

    void reset() override {
        spread_.reset();
        std_.reset();
    }

    ResultTuple call(const InputArray& inputs) override {
        const double spread_val = spread_.call(inputs);
        // RollingSpread emits NaN during its warmup; feeding NaN into
        // RollingStd's running sums would poison them permanently.
        // Skip the feed until spread is valid.
        if (std::isnan(spread_val)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return std_.process_scalar(spread_val);
    }

private:
    RollingSpread spread_;
    RollingStd std_;
};

}  // namespace screamer

#endif
