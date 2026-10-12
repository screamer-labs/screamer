#ifndef SCREAMER_EXPANDING_SKEW_H
#define SCREAMER_EXPANDING_SKEW_H

// ExpandingSkew: bias-corrected sample skewness (G1) over the whole history
// since the last reset(). O(1) memory, no window. Keeps power sums about an
// origin that follows the mean (detail::RunningMomentsOp) and applies
// RollingSkew's estimator.
//
// Convention: the adjusted Fisher-Pearson standardized moment G1, identical to
// RollingSkew and pandas Series.expanding().skew(). Undefined (NaN) for n < 3.
//
// NaN policy "ignore": a NaN input is skipped -- output is NaN at that index
// and internal state is unchanged.

#include <cmath>
#include <limits>
#include "screamer/common/math.h"
#include "screamer/detail/running_moments_op.h"
#include "screamer/detail/moment_formulas.h"

namespace screamer {

class ExpandingSkew : public detail::RunningMomentsOp<ExpandingSkew, 3, false> {
public:
    ExpandingSkew() { reset(); }

    inline double value(const State& st, double) const {
        if (st.sums.w < 3.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const int n = static_cast<int>(st.sums.w);
        double c0;
        skew_n_const(n, c0);
        return detail::skew_from_sums(st.sums.s1, st.sums.s2, st.sums.s3, n, c0);
    }
};

}  // namespace screamer

#endif  // SCREAMER_EXPANDING_SKEW_H
