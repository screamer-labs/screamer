#ifndef SCREAMER_ROLLING_ZSCORE_H
#define SCREAMER_ROLLING_ZSCORE_H

#include <limits>
#include "screamer/detail/window_moments_op.h"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace screamer {

    // (x - window mean) / window standard deviation (ddof=1). The window sums
    // are taken about an origin near the mean (detail::ShiftedWindowSums), so
    // the result stays accurate when the mean is large next to the spread.
    class RollingZscore : public detail::WindowMomentsOp<RollingZscore, 2> {
    public:

        // The output is NaN until the window is full, whatever the start
        // policy, so the window always runs with the strict policy.
        RollingZscore(int window_size, const std::string& start_policy = "strict") :
            WindowMomentsOp(window_size, detail::parse_start_policy(start_policy),
                            detail::StartPolicy::Strict),
            c0(1.0 / (static_cast<double>(window_size) * (window_size - 1)))
        {
            if (window_size_ < 2) {
                throw std::invalid_argument("Window size must be 2 or more.");
            }
            reset();
        }

        inline double value(const State& st, double x) const {
            if (st.count != static_cast<std::size_t>(window_size_)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const double n = static_cast<double>(window_size_);
            const double ns2 = n * st.s[1];
            const double s1s1 = st.s[0] * st.s[0];
            const double var = (ns2 - s1s1) * c0;
            const double mean_d = st.s[0] / n;
            return ((x - st.origin) - mean_d) / std::sqrt(var);
        }

    private:
        const double c0;
    }; // end of class

} // end of namespace

#endif // end of include guards
