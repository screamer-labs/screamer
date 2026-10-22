#ifndef SCREAMER_SPREAD_ESTIMATORS_H
#define SCREAMER_SPREAD_ESTIMATORS_H

// Bid-ask spread estimators from bar high, low and close prices, averaged over
// a trailing window of two-bar estimates. Both are 3 -> 1 over
// (high, low, close) and both read a pair of consecutive bars, so the
// estimate for the pair (t-1, t) is reported at bar t.
//
// CorwinSchultzSpread (Corwin and Schultz 2012). For the pair of bars
// (t-1, t), with bar t's high and low shifted for the gap from the close of
// bar t-1 (optional, the paper's overnight adjustment):
//
//     beta  = ln(H[t-1]/L[t-1])^2 + ln(H[t]/L[t])^2
//     gamma = ln(max(H[t-1], H[t]) / min(L[t-1], L[t]))^2
//     alpha = (sqrt(2 beta) - sqrt(beta)) / (3 - 2 sqrt 2) - sqrt(gamma / (3 - 2 sqrt 2))
//     S     = 2 (e^alpha - 1) / (1 + e^alpha)
//
// A negative two-bar S is set to 0, then S is averaged over the window.
//
// AbdiRanaldoSpread (Abdi and Ranaldo 2017). With c = ln C and the log
// mid-range eta = (ln H + ln L) / 2,
//
//     x[t] = 4 (c[t-1] - eta[t-1]) (c[t-1] - eta[t])
//     S    = sqrt(max(mean of x over the window, 0))
//
// which is the paper's monthly estimator: the products are averaged first and
// a negative average is floored at 0.
//
// nan_policy: ignore. A bar with a NaN high, low or close gives NaN and is
// skipped whole: the next bar pairs with the last usable bar. Each operator
// keeps the previous bar in a few doubles and the window in
// detail::RollingMean, O(1) per bar.

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/float_info.h"
#include "screamer/common/functor_base.h"
#include "screamer/detail/rolling_mean.h"

namespace screamer {

namespace detail {

// Two-bar Corwin-Schultz spread from the squared log ranges of the two bars
// (lr2_prev, lr2) and the log of the two-bar range (g). With k = 3 - 2 sqrt 2,
// (sqrt 2 - 1) / k = 1 / sqrt k = 1 + sqrt 2, so the paper's alpha is
// (1 + sqrt 2) (sqrt(beta) - sqrt(gamma)). A negative estimate is returned
// as 0.
inline double corwin_schultz_pair(double lr2_prev, double lr2, double g) {
    static constexpr double kOnePlusSqrt2 = 2.4142135623730951;
    double alpha = kOnePlusSqrt2 * (std::sqrt(lr2_prev + lr2) - std::fabs(g));
    // A negative estimate becomes 0 through alpha = 0, without a branch on
    // the sign; (alpha < 0) is false for a NaN alpha (a price outside the log
    // domain), which stays NaN.
    alpha = (alpha < 0.0) ? 0.0 : alpha;
    // The paper's form. expm1 or tanh would be exact to the last bit for a
    // small alpha, but measure 1.5x slower here; the cancellation in e - 1
    // costs a relative 1e-16 / alpha (1e-13 at a 10 bp spread).
    const double e = std::exp(alpha);
    return 2.0 * (e - 1.0) / (1.0 + e);
}

inline std::size_t spread_window(int window_size) {
    if (window_size < 1) {
        throw std::invalid_argument("Window size must be positive.");
    }
    return static_cast<std::size_t>(window_size);
}

}  // namespace detail


class CorwinSchultzSpread : public StepperFunctorBase<CorwinSchultzSpread, 3, 1> {
public:
    CorwinSchultzSpread(int window_size, bool adjust_overnight = true,
                        const std::string& start_policy = "strict")
        : adjust_overnight_(adjust_overnight),
          mean_(detail::spread_window(window_size), start_policy)
    {
        reset();
    }

    void reset() override {
        mean_.reset();
        have_prev_ = false;
        prev_high_ = 0.0;
        prev_low_ = 0.0;
        prev_close_ = 0.0;
        prev_lr2_ = 0.0;
    }

    // One bar on plain state, for the batch column loop and call()
    // (StepperFunctorBase). The window step is detail::rolling_mean_step.
    struct Stepper {
        double* buf;
        std::size_t cap;
        std::size_t index;
        std::size_t count;
        double sum;
        detail::StartPolicy policy;
        bool adjust;
        bool have_prev;
        double prev_high;
        double prev_low;
        double prev_close;
        double prev_lr2;      // ln(H/L)^2 of the previous bar, unadjusted

        inline ResultTuple call(const InputArray& inputs) {
            const double high = inputs[0];
            const double low = inputs[1];
            const double close = inputs[2];
            if (any_nan(high, low, close)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const double r = std::log(high / low);
            double lr2 = r * r;       // squared log range, unadjusted
            double s = std::numeric_limits<double>::quiet_NaN();
            if (have_prev) {
                double h = high;
                double l = low;
                double pair_lr2 = lr2;
                // Corwin-Schultz overnight adjustment: a bar whose low is above
                // (high is below) the previous close is shifted down (up) by
                // the gap, so the pair does not count the gap as range.
                if (adjust) {
                    if (l > prev_close) {
                        h -= l - prev_close;
                        l = prev_close;
                        const double ra = std::log(h / l);
                        pair_lr2 = ra * ra;
                    } else if (h < prev_close) {
                        l += prev_close - h;
                        h = prev_close;
                        const double ra = std::log(h / l);
                        pair_lr2 = ra * ra;
                    }
                }
                const double h2 = (h > prev_high) ? h : prev_high;
                const double l2 = (l < prev_low) ? l : prev_low;
                s = detail::corwin_schultz_pair(prev_lr2, pair_lr2, std::log(h2 / l2));
                s = detail::rolling_mean_step(buf, cap, index, count, sum, policy, s);
            }
            have_prev = true;
            prev_high = high;
            prev_low = low;
            prev_close = close;
            prev_lr2 = lr2;
            return s;
        }
    };
    Stepper stepper() {
        return Stepper{mean_.buffer_data(), mean_.capacity(), mean_.index(), mean_.count(),
                       mean_.sum(), mean_.start_policy(), adjust_overnight_, have_prev_,
                       prev_high_, prev_low_, prev_close_, prev_lr2_};
    }
    void store(const Stepper& k) {
        mean_.restore(k.index, k.count, k.sum);
        have_prev_ = k.have_prev;
        prev_high_ = k.prev_high;
        prev_low_ = k.prev_low;
        prev_close_ = k.prev_close;
        prev_lr2_ = k.prev_lr2;
    }

private:
    const bool adjust_overnight_;
    detail::RollingMean mean_;
    bool have_prev_ = false;
    double prev_high_ = 0.0;
    double prev_low_ = 0.0;
    double prev_close_ = 0.0;
    double prev_lr2_ = 0.0;
};


class AbdiRanaldoSpread : public StepperFunctorBase<AbdiRanaldoSpread, 3, 1> {
public:
    AbdiRanaldoSpread(int window_size, const std::string& start_policy = "strict")
        : mean_(detail::spread_window(window_size), start_policy)
    {
        reset();
    }

    void reset() override {
        mean_.reset();
        have_prev_ = false;
        prev_c_ = 0.0;
        prev_dev_ = 0.0;
    }

    struct Stepper {
        double* buf;
        std::size_t cap;
        std::size_t index;
        std::size_t count;
        double sum;
        detail::StartPolicy policy;
        bool have_prev;
        double prev_c;        // ln C of the previous bar
        double prev_dev;      // c - eta of the previous bar

        inline ResultTuple call(const InputArray& inputs) {
            const double high = inputs[0];
            const double low = inputs[1];
            const double close = inputs[2];
            if (any_nan(high, low, close)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const double eta = 0.5 * std::log(high * low);   // (ln H + ln L) / 2
            const double c = std::log(close);
            double s = std::numeric_limits<double>::quiet_NaN();
            if (have_prev) {
                const double x = 4.0 * prev_dev * (prev_c - eta);
                const double m = detail::rolling_mean_step(buf, cap, index, count, sum,
                                                           policy, x);
                // A NaN m (warmup) stays NaN; a negative mean is floored at 0.
                s = isnan2(m) ? m : std::sqrt(m > 0.0 ? m : 0.0);
            }
            have_prev = true;
            prev_c = c;
            prev_dev = c - eta;
            return s;
        }
    };
    Stepper stepper() {
        return Stepper{mean_.buffer_data(), mean_.capacity(), mean_.index(), mean_.count(),
                       mean_.sum(), mean_.start_policy(), have_prev_, prev_c_, prev_dev_};
    }
    void store(const Stepper& k) {
        mean_.restore(k.index, k.count, k.sum);
        have_prev_ = k.have_prev;
        prev_c_ = k.prev_c;
        prev_dev_ = k.prev_dev;
    }

private:
    detail::RollingMean mean_;
    bool have_prev_ = false;
    double prev_c_ = 0.0;
    double prev_dev_ = 0.0;
};

}  // namespace screamer

#endif  // SCREAMER_SPREAD_ESTIMATORS_H
