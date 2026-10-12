#ifndef SCREAMER_ROLLING_SIGMA_CLIP_H
#define SCREAMER_ROLLING_SIGMA_CLIP_H

#include "screamer/common/base.h"
#include "screamer/detail/shifted_window_sums.h"
#include "screamer/common/float_info.h"
#include "screamer/common/math.h"
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
namespace screamer {

    class RollingSigmaClip : public ScreamerBase {
    public:

        static int parse_output(const std::string& s) {
            if (s == "clipped") return 0;
            if (s == "mean")    return 1;
            if (s == "std")     return 2;
            if (s == "nan")     return 3;
            throw std::invalid_argument(
                "output must be \"clipped\", \"mean\", \"std\", or \"nan\".");
        }

        RollingSigmaClip(
            int window_size,
            std::optional<double> lower = std::nullopt,
            std::optional<double> upper = std::nullopt,
            const std::string& output = "clipped",
            const std::string& start_policy = "strict"
        ) :
            window_size_(window_size),
            lower_bound_(lower.value_or(std::numeric_limits<double>::lowest())),
            upper_bound_(upper.value_or(std::numeric_limits<double>::max())),
            output_(parse_output(output)),
            start_policy_(detail::parse_start_policy(start_policy)),
            sums_(window_size > 0 ? window_size : 1, detail::parse_start_policy(start_policy))
        {
            if (window_size_ <= 0) {
                throw std::invalid_argument("Window size must be positive.");
            }

            if (lower_bound_ >= upper_bound_ ) {
                throw std::invalid_argument("Lower bound must be below the upper bound.");
            }

            standard_truncated_normal_mean_variance(
                lower_bound_,
                upper_bound_,
                mu_trunc,
                sigma_trunc
            );
            reset();
        }

        void reset() override {
            sums_.reset();
            mean_ = std::numeric_limits<double>::quiet_NaN();
            std_ = std::numeric_limits<double>::quiet_NaN(); 
            n_ = (start_policy_ != detail::StartPolicy::Zero) ? 0 : window_size_;         
        }
        
        void _update_mean_std(double newValue,size_t n_) {
            // Sums about an origin near the window mean, stable when the mean
            // is large next to the spread. The step runs on a copy of the
            // state, which is then stored back whole.
            auto c = sums_.cursor();
            c.push(newValue);
            sums_.set_state(c.st);

            // Compute the raw mean and std (NaN until the window is full
            // under the strict policy, so nothing is clipped before then)
            double observed_mean = std::numeric_limits<double>::quiet_NaN();
            double observed_std = std::numeric_limits<double>::quiet_NaN();
            if (start_policy_ != detail::StartPolicy::Strict || c.st.count == c.cap) {
                const double sum_d = c.st.s[0];
                const double sum_dd = c.st.s[1];
                observed_mean = c.st.origin + sum_d / n_;
                observed_std = std::sqrt((sum_dd - sum_d * sum_d / n_) / (n_ - 1));
            }

            // Correct to the mean_ and std_ for truncation bias
            estimate_true_mean_std(
                observed_mean, 
                observed_std, 
                mu_trunc, 
                sigma_trunc, 
                mean_, 
                std_
            );
        }

        double process_scalar(double newValue) override {

            double zscore, clippedValue;

            // if we have a mean and std then we can do clipping
            clippedValue = newValue;
            if ( (!isnan2(mean_)) && (!isnan2(std_))) {
                zscore = (newValue - mean_) / std_;
                if (zscore < lower_bound_) {
                    clippedValue =  mean_ + lower_bound_ * std_;
                }
                if (zscore > upper_bound_) {
                    clippedValue = mean_ + upper_bound_ * std_;
                }
            }

            // did we clip? If so we don't update stats, we return
            if (clippedValue != newValue) {
                if (n_ < window_size_) {
                    if (start_policy_ == detail::StartPolicy::Strict) {
                        return std::numeric_limits<double>::quiet_NaN();
                    }
                    if (n_ < 2) return std::numeric_limits<double>::quiet_NaN();
                }
                if (output_ == 0) return clippedValue;
                if (output_ == 1) return mean_;
                if (output_ == 2) return std_;
                return std::numeric_limits<double>::quiet_NaN();
            }

            // we didn't clip: update 
            if (n_ < window_size_) {
                n_++;
            } 
            _update_mean_std(newValue, n_);

            // return the final result
            if (n_ < window_size_) {
                if (start_policy_ == detail::StartPolicy::Strict) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                if (n_ < 2) return std::numeric_limits<double>::quiet_NaN();
            }        
            if (output_ == 1) return mean_;
            if (output_ == 2) return std_;
            return newValue;
        }

    private:
        detail::ShiftedWindowSums<2> sums_;
        const detail::StartPolicy start_policy_;

        const double lower_bound_;
        const double upper_bound_;
        const int window_size_;
        const int output_;
        int n_ = 0;
        double mu_trunc = 0;
        double sigma_trunc = 0;
        double mean_ = 0;
        double std_ = 0;
    }; // end of class

} // end of namespace

#endif // end of include guards

