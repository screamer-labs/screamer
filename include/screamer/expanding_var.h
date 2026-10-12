#ifndef SCREAMER_EXPANDING_VAR_H
#define SCREAMER_EXPANDING_VAR_H

// ExpandingVar: sample variance (ddof=1) over the whole history since the last
// reset(). O(1) memory, no window, no start_policy. Keeps power sums about an
// origin that follows the mean (detail::RunningMomentsOp).
//
// Convention: ddof=1 (sample variance), matching RollingVar / RollingStd and
// pandas Series.expanding().var() (default ddof=1). Undefined (NaN) for n < 2.
//
// NaN policy "ignore" (see docs/nan_policy.md): a NaN input is skipped --
// output is NaN at that index and internal state is unchanged.

#include <cmath>
#include <limits>
#include "screamer/common/math.h"
#include "screamer/detail/running_moments_op.h"

namespace screamer {

class ExpandingVar : public detail::RunningMomentsOp<ExpandingVar, 2, false> {
public:
    ExpandingVar() { reset(); }

    inline double value(const State& st, double) const {
        if (st.sums.w < 2.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const int n = static_cast<int>(st.sums.w);
        double var;
        var_from_stats(st.sums.s1, st.sums.s2, n, var);
        return var;
    }
};

}  // namespace screamer

#endif  // SCREAMER_EXPANDING_VAR_H
