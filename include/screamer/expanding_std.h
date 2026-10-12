#ifndef SCREAMER_EXPANDING_STD_H
#define SCREAMER_EXPANDING_STD_H

// ExpandingStd: sample standard deviation (ddof=1) over the whole history
// since the last reset(). sqrt of ExpandingVar. O(1) memory, no window.
//
// Convention: ddof=1, matching RollingStd and pandas
// Series.expanding().std() (default ddof=1). Undefined (NaN) for n < 2.
//
// NaN policy "ignore": a NaN input is skipped -- output is NaN at that index
// and internal state is unchanged.

#include <cmath>
#include <limits>
#include "screamer/common/math.h"
#include "screamer/detail/running_moments_op.h"

namespace screamer {

class ExpandingStd : public detail::RunningMomentsOp<ExpandingStd, 2, false> {
public:
    ExpandingStd() { reset(); }

    inline double value(const State& st, double) const {
        if (st.sums.w < 2.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const int n = static_cast<int>(st.sums.w);
        double var;
        var_from_stats(st.sums.s1, st.sums.s2, n, var);
        return std::sqrt(var);
    }
};

}  // namespace screamer

#endif  // SCREAMER_EXPANDING_STD_H
