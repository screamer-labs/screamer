#ifndef SCREAMER_EXPANDING_KURT_H
#define SCREAMER_EXPANDING_KURT_H

// ExpandingKurt: bias-corrected sample excess kurtosis (Fisher) over the whole
// history since the last reset(). O(1) memory, no window. Keeps power sums about
// an origin that follows the mean (detail::RunningMomentsOp) and applies
// RollingKurt's estimator.
//
// Convention: the bias-corrected Fisher excess kurtosis, identical to
// RollingKurt and pandas Series.expanding().kurt(). Undefined (NaN) for n < 4.
//
// NaN policy "ignore": a NaN input is skipped -- output is NaN at that index
// and internal state is unchanged.

#include <cmath>
#include <limits>
#include "screamer/common/math.h"
#include "screamer/detail/running_moments_op.h"
#include "screamer/detail/moment_formulas.h"

namespace screamer {

class ExpandingKurt : public detail::RunningMomentsOp<ExpandingKurt, 4, false> {
public:
    ExpandingKurt() { reset(); }

    inline double value(const State& st, double) const {
        if (st.sums.w < 4.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const int n = static_cast<int>(st.sums.w);
        double c0, c1, c2;
        kurt_n_const(n, c0, c1, c2);
        return detail::kurt_from_sums(st.sums.s1, st.sums.s2, st.sums.s3, st.sums.s4, n,
                                      c0, c1, c2);
    }
};

}  // namespace screamer

#endif  // SCREAMER_EXPANDING_KURT_H
