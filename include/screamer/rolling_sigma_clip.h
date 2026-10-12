#ifndef SCREAMER_ROLLING_SIGMA_CLIP_H
#define SCREAMER_ROLLING_SIGMA_CLIP_H

#include "screamer/common/base.h"
#include "screamer/detail/moment_formulas.h"
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
            sums_(window_size > 0 ? window_size : 1, detail::parse_start_policy(start_policy)),
            start_policy_(detail::parse_start_policy(start_policy)),
            lower_bound_(lower.value_or(std::numeric_limits<double>::lowest())),
            upper_bound_(upper.value_or(std::numeric_limits<double>::max())),
            window_size_(window_size),
            output_(parse_output(output))
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
        }
        
        double process_scalar(double newValue) override {
            Clip k{sums_.cursor(), mean_, std_};
            const double r = step(k, newValue);
            store(k);
            return r;
        }

        // Batch paths: the same step on a local copy of the state.
        void process_array_no_stride(double* y, const double* x, size_t size) override {
            run(y, 1, x, 1, size);
        }
        void process_array_stride(double* y, size_t dy, const double* x, size_t dx,
                                  size_t size) override {
            run(y, dy, x, dx, size);
        }

    private:
        // The running state: the window (sums about an origin near its mean,
        // stable when the mean is large next to the spread) and the corrected
        // mean and standard deviation.
        struct Clip {
            detail::ShiftedWindowSums<2>::Cursor c;
            double mean;
            double sd;
        };

        void store(const Clip& k) {
            sums_.set_state(k.c.st);
            mean_ = k.mean;
            std_ = k.sd;
        }

        void run(double* y, size_t dy, const double* x, size_t dx, size_t size) {
            Clip k{sums_.cursor(), mean_, std_};
            for (size_t i = 0; i < size; ++i, y += dy, x += dx) *y = step(k, *x);
            store(k);
        }

        // One event. Each product is its own statement (see
        // detail/shifted_window_sums.h).
        SCREAMER_WINDOW_INLINE double step(Clip& k, double newValue) const {
            const double nan = std::numeric_limits<double>::quiet_NaN();

            // if we have a mean and std then we can do clipping
            double clippedValue = newValue;
            if ((!isnan2(k.mean)) && (!isnan2(k.sd))) {
                const double zscore = (newValue - k.mean) / k.sd;
                if (zscore < lower_bound_) {
                    const double offset = lower_bound_ * k.sd;
                    clippedValue = k.mean + offset;
                }
                if (zscore > upper_bound_) {
                    const double offset = upper_bound_ * k.sd;
                    clippedValue = k.mean + offset;
                }
            }

            // did we clip? If so we don't update stats, we return
            if (clippedValue != newValue) {
                const int n_ = static_cast<int>(k.c.st.count);
                if (n_ < window_size_) {
                    if (start_policy_ == detail::StartPolicy::Strict) return nan;
                    if (n_ < 2) return nan;
                }
                if (output_ == 0) return clippedValue;
                if (output_ == 1) return k.mean;
                if (output_ == 2) return k.sd;
                return nan;
            }

            // we didn't clip: update
            k.c.push(newValue);
            const int n_ = static_cast<int>(k.c.st.count);

            // The raw mean and std (NaN until the window is full under the
            // strict policy, so nothing is clipped before then), corrected
            // for the truncation.
            double observed_mean = nan;
            double observed_std = nan;
            if (start_policy_ != detail::StartPolicy::Strict || k.c.st.count == k.c.cap) {
                const double n = static_cast<double>(n_);
                observed_mean = k.c.st.origin + k.c.st.s[0] / n;
                observed_std = std::sqrt(detail::var_from_sums(k.c.st.s[0], k.c.st.s[1], n));
            }
            double mean = nan, sd = nan;
            estimate_true_mean_std(observed_mean, observed_std, mu_trunc, sigma_trunc, mean, sd);
            k.mean = mean;
            k.sd = sd;

            // return the final result
            if (n_ < window_size_) {
                if (start_policy_ == detail::StartPolicy::Strict) return nan;
                if (n_ < 2) return nan;
            }
            if (output_ == 1) return k.mean;
            if (output_ == 2) return k.sd;
            return newValue;
        }

        detail::ShiftedWindowSums<2> sums_;
        const detail::StartPolicy start_policy_;

        const double lower_bound_;
        const double upper_bound_;
        const int window_size_;
        const int output_;
        double mu_trunc = 0;
        double sigma_trunc = 0;
        double mean_ = 0;
        double std_ = 0;
    }; // end of class

} // end of namespace

#endif // end of include guards

