#ifndef SCREAMER_ROLLING_ADF_H
#define SCREAMER_ROLLING_ADF_H

// RollingADF: the augmented Dickey-Fuller statistic over a rolling window.
//
// For each window of `window_size` regression rows, ordinary least squares fits
//
//     dy_t = c + gamma * y_{t-1} + phi_1 dy_{t-1} + ... + phi_p dy_{t-p} + e_t
//
// with p = `lags`, and the output is the t-statistic of gamma. A strongly
// negative value says the series reverts to a mean (no unit root). Applied to
// the spread of two assets, it is the second step of the Engle-Granger
// cointegration test. Compare it with the Dickey-Fuller critical values
// (MacKinnon 2010), not with the normal distribution.
//
// Each step adds one regression row and drops the oldest; the cross-product
// sums are updated in O(d^2) with d = lags + 2 and recomputed from the stored
// rows once per window to stop rounding drift, and the d x d normal equations
// are solved by Gauss-Jordan elimination with partial pivoting. The first row
// needs lags + 1 earlier values, so the first output is at step
// window_size + lags; before that the output is NaN. NaN when the regression
// is singular. A NaN input skips the step.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"

namespace screamer {

class RollingADF : public ScreamerBase {
public:
    RollingADF(int window_size = 100, int lags = 1)
        : w_(window_size), p_(lags), d_(static_cast<std::size_t>(lags) + 2)
    {
        if (lags < 0) throw std::invalid_argument("lags must be non-negative.");
        if (window_size < lags + 4)
            throw std::invalid_argument("window_size must be at least lags + 4.");
        rows_.assign(static_cast<std::size_t>(w_) * (d_ + 1), 0.0);
        xtx_.assign(d_ * d_, 0.0);
        xty_.assign(d_, 0.0);
        a_.assign(d_ * 2 * d_, 0.0);
        hist_.assign(static_cast<std::size_t>(p_) + 1, 0.0);
        reset();
    }

    void reset() override {
        std::fill(xtx_.begin(), xtx_.end(), 0.0);
        std::fill(xty_.begin(), xty_.end(), 0.0);
        yty_ = 0.0;
        count_ = head_ = since_refresh_ = 0;
        seen_ = 0;
        prev_ = 0.0;
    }

private:
    double process_scalar(double y) override {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (isnan2(y)) return nan;
        if (seen_ == 0) {
            prev_ = y;
            seen_ = 1;
            return nan;
        }
        const double dy = y - prev_;
        // hist_ holds the last p + 1 differences, newest first, after prev_.
        if (seen_ >= static_cast<std::size_t>(p_) + 1) {
            double* row = rows_.data() + head_ * (d_ + 1);
            if (count_ == static_cast<std::size_t>(w_)) accumulate(row, -1.0);
            row[0] = 1.0;
            row[1] = prev_;
            for (int i = 0; i < p_; ++i) row[2 + i] = hist_[i];
            row[d_] = dy;
            accumulate(row, 1.0);
            head_ = (head_ + 1) % static_cast<std::size_t>(w_);
            if (count_ < static_cast<std::size_t>(w_)) ++count_;
            if (++since_refresh_ >= static_cast<std::size_t>(w_)) refresh();
        }
        for (int i = p_; i > 0; --i) hist_[i] = hist_[i - 1];
        hist_[0] = dy;
        prev_ = y;
        ++seen_;

        if (count_ < static_cast<std::size_t>(w_)) return nan;   // window not full
        return t_stat();
    }

    void accumulate(const double* row, double sign) {
        for (std::size_t i = 0; i < d_; ++i) {
            for (std::size_t j = 0; j < d_; ++j) xtx_[i * d_ + j] += sign * row[i] * row[j];
            xty_[i] += sign * row[i] * row[d_];
        }
        yty_ += sign * row[d_] * row[d_];
    }

    void refresh() {
        std::fill(xtx_.begin(), xtx_.end(), 0.0);
        std::fill(xty_.begin(), xty_.end(), 0.0);
        yty_ = 0.0;
        for (std::size_t r = 0; r < count_; ++r) accumulate(rows_.data() + r * (d_ + 1), 1.0);
        since_refresh_ = 0;
    }

    // t-statistic of the y_{t-1} coefficient: invert X'X with Gauss-Jordan.
    double t_stat() {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const std::size_t m = 2 * d_;
        for (std::size_t i = 0; i < d_; ++i)
            for (std::size_t j = 0; j < m; ++j)
                a_[i * m + j] = j < d_ ? xtx_[i * d_ + j] : (j - d_ == i ? 1.0 : 0.0);
        double scale = 0.0;
        for (std::size_t i = 0; i < d_; ++i) scale = std::max(scale, std::abs(xtx_[i * d_ + i]));
        for (std::size_t c = 0; c < d_; ++c) {
            std::size_t piv = c;
            for (std::size_t r = c + 1; r < d_; ++r)
                if (std::abs(a_[r * m + c]) > std::abs(a_[piv * m + c])) piv = r;
            if (!(std::abs(a_[piv * m + c]) > 1e-13 * scale)) return nan;
            if (piv != c)
                for (std::size_t j = 0; j < m; ++j) std::swap(a_[c * m + j], a_[piv * m + j]);
            const double inv = 1.0 / a_[c * m + c];
            for (std::size_t j = 0; j < m; ++j) a_[c * m + j] *= inv;
            for (std::size_t r = 0; r < d_; ++r) {
                if (r == c) continue;
                const double f = a_[r * m + c];
                if (f != 0.0)
                    for (std::size_t j = 0; j < m; ++j) a_[r * m + j] -= f * a_[c * m + j];
            }
        }
        // beta = (X'X)^-1 X'y; rss = y'y - beta' X'y.
        double gamma = 0.0, fit = 0.0;
        for (std::size_t i = 0; i < d_; ++i) {
            double bi = 0.0;
            for (std::size_t j = 0; j < d_; ++j) bi += a_[i * m + d_ + j] * xty_[j];
            if (i == 1) gamma = bi;
            fit += bi * xty_[i];
        }
        const double dof = static_cast<double>(count_) - static_cast<double>(d_);
        const double rss = std::max(yty_ - fit, 0.0);
        const double var = rss / dof * a_[1 * m + d_ + 1];
        if (!(var > 0.0)) return nan;
        return gamma / std::sqrt(var);
    }

    const int w_, p_;
    const std::size_t d_;
    std::vector<double> rows_, xtx_, xty_, a_, hist_;
    double yty_ = 0.0, prev_ = 0.0;
    std::size_t count_ = 0, head_ = 0, since_refresh_ = 0, seen_ = 0;
};

}  // namespace screamer

#endif
