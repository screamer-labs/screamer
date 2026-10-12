#ifndef SCREAMER_ROLLING_RESIDUAL_STD_H
#define SCREAMER_ROLLING_RESIDUAL_STD_H

// RollingResidualStd: rolling std of the per-bar spread x - beta*y.
//
// Useful for pairs-trading normalisation -- the z-score of a current
// spread is (spread - rolling_mean(spread)) / RollingResidualStd.
//
// Convention matches RollingBeta / RollingSpread: FIRST argument is
// the dependent (target), SECOND is the regressor. Composes
// RollingSpread with the RollingStd of the spreads. O(1) per step.
// ddof=1, matching the rest of the library.

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/functor_base.h"
#include "screamer/rolling_spread.h"
#include "screamer/detail/shifted_window_sums.h"

namespace screamer {

class RollingResidualStd : public FunctorBase<RollingResidualStd, 2, 1> {
    using StdSums = detail::ShiftedWindowSums<2>;

public:
    RollingResidualStd(int window_size, const std::string& start_policy = "strict")
        : spread_(window_size, start_policy),
          start_policy_(detail::parse_start_policy(start_policy)),
          sums_(window_size >= 2 ? window_size : 2, start_policy_)
    {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be at least 2.");
        }
        reset();
    }

    void reset() override {
        spread_.reset();
        sums_.reset();
    }

    // The spread's step, then the RollingStd of the spreads on their own
    // window of sums; the batch column loop runs it on a local copy
    // (FunctorBase, has_stepper).
    struct Stepper {
        RollingSpread::Stepper spread;
        StdSums::Cursor c;
        bool strict;

        inline ResultTuple call(const InputArray& inputs) {
            const double spread_val = spread.call(inputs);
            if (std::isnan(spread_val)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            c.push(spread_val);
            const std::size_t n_ = c.st.count;
            if (strict && n_ < c.cap) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const double n = static_cast<double>(n_);
            const double var = (c.st.s[1] - c.st.s[0] * c.st.s[0] / n) / (n - 1);
            return std::sqrt(var);
        }
    };
    Stepper stepper() {
        return Stepper{spread_.stepper(), sums_.cursor(),
                       start_policy_ == detail::StartPolicy::Strict};
    }
    void store(const Stepper& k) {
        spread_.store(k.spread);
        sums_.set_state(k.c.st);
    }

    ResultTuple call(const InputArray& inputs) override {
        Stepper k = stepper();
        const ResultTuple r = k.call(inputs);
        store(k);
        return r;
    }

private:
    RollingSpread spread_;
    const detail::StartPolicy start_policy_;
    StdSums sums_;
};

}  // namespace screamer

#endif
