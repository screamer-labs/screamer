#ifndef SCREAMER_ROLLING_SKEW_H
#define SCREAMER_ROLLING_SKEW_H

#include <limits>
#include "screamer/common/math.h"
#include "screamer/detail/moment_formulas.h"
#include "screamer/detail/window_moments_op.h"
#include <stdexcept>
#include <string>
namespace screamer {

    // Adjusted Fisher-Pearson skewness of the window. The window sums are taken
    // about an origin near the mean (detail::ShiftedWindowSums), so the result
    // stays accurate when the mean is large next to the spread.
    class RollingSkew : public detail::WindowMomentsOp<RollingSkew, 3> {
    public:

        RollingSkew(
            int window_size,
            const std::string& start_policy = "strict"
        ) :
            WindowMomentsOp(window_size, detail::parse_start_policy(start_policy),
                            detail::parse_start_policy(start_policy))
        {
            if (window_size <= 2) {
                throw std::invalid_argument("Window size must be at least 2.");
            }
            skew_n_const(window_size_, c0_full_);
            reset();
        }

        inline double value(const State& st, double) const {
            if (start_policy_ == detail::StartPolicy::Strict &&
                st.count != static_cast<std::size_t>(window_size_)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const int n_ = static_cast<int>(st.count);
            double c0 = c0_full_;
            if (n_ != window_size_) skew_n_const(n_, c0);
            return detail::skew_from_sums(st.s[0], st.s[1], st.s[2], n_, c0);
        }

    private:
        double c0_full_ = 0.0;
    }; // end of class

} // end of namespace

#endif // end of include guards
