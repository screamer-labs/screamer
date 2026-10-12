#ifndef SCREAMER_ROLLING_OU_H
#define SCREAMER_ROLLING_OU_H

// RollingOU: a discrete Ornstein-Uhlenbeck (AR(1)) fit over the window.
//
// The window_size most recent values x_0 .. x_{n-1} give m = n - 1 pairs
// (x_{i-1}, x_i), and ordinary least squares fits
//
//     x_i = a + b x_{i-1} + e_i
//
// Writing the pair sums about the pair means (u = x_{i-1}, v = x_i):
//
//     b = sum (u - mean u)(v - mean v) / sum (u - mean u)^2
//     a = mean v - b mean u
//
// and the outputs are
//
//     mrr     = 1 - b                        (mean-reversion rate per step)
//     mean    = a / (1 - b)                  (long-run mean)
//     relmean = mean - x_{n-1}
//     std     = sqrt(SSE / (m - 2)),  SSE = sum (v - mean v)^2 - b sum (u - mean u)(v - mean v)
//
// A level L added to the input adds L to mean and leaves mrr, relmean and std
// unchanged: value() writes mean as an origin near the data plus a term of
// the size of the spread. The pair sums are kept about
// origins near the means (detail::ShiftedWindowCoSums over the m pairs), so
// the fit stays accurate when the level is large next to the spread. O(1) per
// step.
//
// start_policy: "strict" gives NaN until the window holds window_size values;
// "expanding" fits the pairs seen so far (mrr and mean from 2 pairs, std from
// 3); "zero" fits a window pre-filled with zeros. A NaN input gives NaN and
// leaves the state untouched.

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/moment_formulas.h"
#include "screamer/detail/shifted_window_sums.h"
#include "screamer/detail/start_policy.h"

namespace screamer {

class RollingOU : public ScreamerBase {
    using Sums = detail::ShiftedWindowCoSums;

public:
    static int parse_output(const std::string& s) {
        if (s == "mrr")     return 0;
        if (s == "mean")    return 1;
        if (s == "relmean") return 2;
        if (s == "std")     return 3;
        throw std::invalid_argument(
            "output must be \"mrr\", \"mean\", \"relmean\", or \"std\".");
    }

    RollingOU(
        int window_size,
        const std::string& output = "mrr",
        const std::string& start_policy = "strict"
    ) :
        output_(parse_output(output)),
        start_policy_(detail::parse_start_policy(start_policy)),
        sums_(window_size >= 2 ? static_cast<std::size_t>(window_size - 1) : 1, start_policy_)
    {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be 2 or more.");
        }
        reset();
    }

    void reset() override {
        sums_.reset();
        // The zero policy's window is all zeros, so the value before the first
        // input is 0.
        st_ = Step{0.0, start_policy_ == detail::StartPolicy::Zero};
    }

    double process_scalar(double x) override {
        if (isnan2(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        Sums::Cursor c = sums_.cursor();
        Step k = st_;
        const double out = step(c, k, x);
        sums_.set_state(c.st);
        st_ = k;
        return out;
    }

    void process_array_no_stride(double* y, const double* x, size_t size) override {
        run(y, 1, x, 1, size);
    }

    void process_array_stride(double* y, size_t dy, const double* x, size_t dx,
                              size_t size) override {
        run(y, dy, x, dx, size);
    }

private:
    // The previous input, and whether there is one.
    struct Step {
        double prev;
        bool has_prev;
    };

    SCREAMER_WINDOW_INLINE double step(Sums::Cursor& c, Step& k, double x) const {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (!k.has_prev) {
            k.prev = x;
            k.has_prev = true;
            return nan;
        }
        c.push(k.prev, x);  // (u, v) = (x_{i-1}, x_i)
        k.prev = x;
        const std::size_t m = c.st.count;
        if (start_policy_ == detail::StartPolicy::Strict && m != c.cap) return nan;
        return value(c.st, x);
    }

    // Each product is its own statement (see shifted_window_sums.h).
    SCREAMER_WINDOW_INLINE double value(const Sums::State& st, double x) const {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const double m = static_cast<double>(st.count);
        const double den = detail::n2_comoment(st.sx, st.sx, st.sxx, m);
        if (!(den > 0.0)) return nan;
        const double num = detail::n2_comoment(st.sx, st.sy, st.sxy, m);
        const double b = num / den;
        const double mrr = 1.0 - b;
        if (output_ == 0) return mrr;
        if (output_ == 3) {
            if (st.count < 3) return nan;
            // m SSE = m Syy - (sum v)^2 - b (m Sxy - sum u sum v).
            const double den_y = detail::n2_comoment(st.sy, st.sy, st.syy, m);
            const double fit = b * num;
            const double m_sse = den_y - fit;
            const double mm2 = m * (m - 2.0);
            return m_sse > 0.0 ? std::sqrt(m_sse / mm2) : 0.0;
        }
        // a = mean v - b mean u. Relative to the origin ou of u, with
        // mean u = ou + su / m and mean v = ov + sv / m:
        //   mean = a / (1 - b) = ou + ((ov - ou) + (sv - b su) / m) / (1 - b),
        // where ov - ou is exact and small, so no term carries the level.
        const double b_sx = b * st.sx;
        const double a_rel = (st.oy - st.ox) + (st.sy - b_sx) / m;
        const double dev = a_rel / mrr;  // mean - ou
        if (output_ == 1) return st.ox + dev;
        return (st.ox - x) + dev;
    }

    void run(double* y, size_t dy, const double* x, size_t dx, size_t size) {
        Sums::Cursor c = sums_.cursor();
        Step k = st_;
        for (size_t i = 0; i < size; ++i, y += dy, x += dx) {
            const double v = *x;
            if (isnan2(v)) {
                *y = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            *y = step(c, k, v);
        }
        sums_.set_state(c.st);
        st_ = k;
    }

    const int output_;
    const detail::StartPolicy start_policy_;
    Sums sums_;
    Step st_{0.0, false};
};

}  // namespace screamer

#endif  // SCREAMER_ROLLING_OU_H
