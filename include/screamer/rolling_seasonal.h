#ifndef SCREAMER_ROLLING_SEASONAL_H
#define SCREAMER_ROLLING_SEASONAL_H

// RollingSeasonalMean, RollingSeasonalStd, RollingSeasonalZscore: the mean,
// sample standard deviation (ddof 1) and z-score of the same slot of the last
// `window_size` cycles, x_t, x_{t - period}, ..., x_{t - (window_size - 1)
// period}. Step t (counted from the first step) belongs to slot t mod period,
// so the input must be on a regular clock. Each slot keeps its own window
// (detail::ShiftedWindowSums<2>, sums about an origin near the slot mean).
// The z-score includes x_t in its window, as RollingZscore does.
//
// start_policy: "strict" gives NaN until the slot holds window_size values,
// "expanding" uses the values the slot holds (the standard deviation needs
// two), and "zero" starts each slot with window_size zeros.
//
// nan_policy: nan-aware. A NaN step advances the slot clock, enters no
// window, and gives NaN, so a missing value on the clock does not shift the
// later values into the wrong slots. O(1) per step; memory period *
// window_size values.

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/moment_formulas.h"
#include "screamer/detail/shifted_window_sums.h"
#include "screamer/detail/start_policy.h"

namespace screamer {
namespace detail {

template <class Derived>
class SeasonalMomentsOp : public ScreamerBase {
protected:
    using Sums = ShiftedWindowSums<2>;
    using State = Sums::State;

    SeasonalMomentsOp(int period, int window_size, const std::string& start_policy,
                      int min_window)
        : window_size_(window_size),
          start_policy_(parse_start_policy(start_policy))
    {
        if (period < 1) throw std::invalid_argument("period must be 1 or more.");
        if (window_size < min_window)
            throw std::invalid_argument(min_window == 1 ? "Window size must be 1 or more."
                                                        : "Window size must be 2 or more.");
        slots_.reserve(static_cast<std::size_t>(period));
        for (int i = 0; i < period; ++i)
            slots_.emplace_back(static_cast<std::size_t>(window_size), start_policy_);
        reset();
    }

public:
    void reset() override {
        for (Sums& s : slots_) s.reset();
        slot_ = 0;
    }

    double process_scalar(double x) override {
        Sums& s = slots_[slot_];
        if (++slot_ == slots_.size()) slot_ = 0;
        if (isnan2(x)) return std::numeric_limits<double>::quiet_NaN();
        Sums::Cursor c = s.cursor();
        c.push(x);
        s.set_state(c.st);
        return static_cast<const Derived&>(*this).value(c.st, x);
    }

protected:
    bool ready(const State& st, std::size_t minimum) const {
        if (start_policy_ == StartPolicy::Strict)
            return st.count == static_cast<std::size_t>(window_size_);
        return st.count >= minimum;
    }

    const int window_size_;
    const StartPolicy start_policy_;
    std::vector<Sums> slots_;
    std::size_t slot_ = 0;
};

}  // namespace detail

class RollingSeasonalMean : public detail::SeasonalMomentsOp<RollingSeasonalMean> {
public:
    RollingSeasonalMean(int period = 24, int window_size = 20,
                        const std::string& start_policy = "strict")
        : SeasonalMomentsOp(period, window_size, start_policy, 1) {}

    double value(const State& st, double) const {
        if (!ready(st, 1)) return std::numeric_limits<double>::quiet_NaN();
        return st.origin + st.s[0] / static_cast<double>(st.count);
    }
};

class RollingSeasonalStd : public detail::SeasonalMomentsOp<RollingSeasonalStd> {
public:
    RollingSeasonalStd(int period = 24, int window_size = 20,
                       const std::string& start_policy = "strict")
        : SeasonalMomentsOp(period, window_size, start_policy, 2) {}

    double value(const State& st, double) const {
        if (!ready(st, 2)) return std::numeric_limits<double>::quiet_NaN();
        const double n = static_cast<double>(st.count);
        return std::sqrt(detail::var_from_sums(st.s[0], st.s[1], n));
    }
};

class RollingSeasonalZscore : public detail::SeasonalMomentsOp<RollingSeasonalZscore> {
public:
    RollingSeasonalZscore(int period = 24, int window_size = 20,
                          const std::string& start_policy = "strict")
        : SeasonalMomentsOp(period, window_size, start_policy, 2) {}

    double value(const State& st, double x) const {
        if (!ready(st, 2)) return std::numeric_limits<double>::quiet_NaN();
        const double n = static_cast<double>(st.count);
        const double var = detail::var_from_sums(st.s[0], st.s[1], n);
        const double mean_d = st.s[0] / n;
        return ((x - st.origin) - mean_d) / std::sqrt(var);
    }
};

}  // namespace screamer

#endif  // SCREAMER_ROLLING_SEASONAL_H
