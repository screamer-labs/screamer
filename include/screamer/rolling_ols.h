#ifndef SCREAMER_ROLLING_OLS_H
#define SCREAMER_ROLLING_OLS_H

// RollingOLS and ExpandingOLS: ordinary least squares of y on k regressors,
//
//     y = b0 + b1 x1 + ... + bk xk + e      (b0 only with an intercept)
//
// over a trailing window of rows (RollingOLS) or over every row so far
// (ExpandingOLS). Inputs are y, x1, ..., xk (1 <= k <= 16). Outputs, per row:
//
//     pred   b0 + b . x_t with the coefficients fitted on the rows before t,
//            a one-step-ahead forecast of y_t
//     resid  y_t - pred
//     r2     R^2 of the fit after row t enters
//     b0     the intercept after row t enters (with an intercept only)
//     b1..bk the coefficients after row t enters
//
// A fit needs a full window (RollingOLS) or p + 1 rows (ExpandingOLS), with
// p = k + 1 parameters with an intercept and p = k without; r2 and the
// coefficients are NaN before that, and pred is NaN until the row after the
// first fit. A row with a NaN in any input is skipped: it does not enter the
// window and its outputs are NaN.
//
// With an intercept the operator keeps the means of the window's columns and
// their centered cross-products, updated in O(k^2) per row (Welford's add and
// remove), and solves the k x k centered normal equations by Cholesky; the
// intercept is mean_y - b . mean_x and r2 = 1 - SSE / sum (y - mean_y)^2.
// Each column enters as z - origin, with the origin near the column's mean
// (moved there when the mean drifts more than four standard deviations from
// it, and at every exact recomputation), so the fit stays accurate when the
// levels are large next to the spread.
// Without an intercept it keeps the raw cross-products, and r2 is the
// uncentered 1 - SSE / sum y^2 (the statsmodels convention for a model
// without a constant). The solve costs O(k^3) per row.
//
// RollingOLS recomputes its cross-products exactly from the stored rows every
// window_size rows, and at once when the row that left was far out of the
// rest (its squared deviation more than 16 times the remaining sum of squares
// in some column, or not finite), so rounding from the removals stays
// bounded. A window whose regressors are collinear (a pivot of the Cholesky
// factor below 1e-12 of its diagonal, for example a regressor that is constant
// in the window with an intercept) is singular: r2 and the coefficients are
// NaN, as is the next pred.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
#include "screamer/common/float_info.h"
#include "screamer/common/var_arity_op.h"

namespace screamer {
namespace detail {

constexpr std::size_t kMaxOLSRegressors = 16;

// The fit shared by RollingOLS (window > 0) and ExpandingOLS (window 0).
template <class Derived>
class OLSFit : public VarArityOp<Derived> {
    static constexpr std::size_t D = kMaxOLSRegressors + 1;   // x1..xk, y

public:
    OLSFit(std::size_t window, int n_regressors, bool intercept)
        : VarArityOp<Derived>(check_k(n_regressors) + 1,
                              static_cast<std::size_t>(n_regressors) + (intercept ? 4 : 3)),
          k_(static_cast<std::size_t>(n_regressors)), d_(k_ + 1), m_(k_ + (intercept ? 4 : 3)),
          icpt_(intercept), w_(window),
          n_min_(window ? window : k_ + (intercept ? 2 : 1))
    {
        if (w_) ring_.assign(w_ * d_, 0.0);
        reset_fit();
    }

    // One row: in = (y, x1..xk); out = (pred, resid, r2, [b0], b1..bk).
    inline void step(const double* in, double* out) {
        constexpr double nan = std::numeric_limits<double>::quiet_NaN();
        const std::size_t m = m_;
        for (std::size_t i = 0; i <= k_; ++i) {
            if (isnan2(in[i])) {
                for (std::size_t j = 0; j < m; ++j) out[j] = nan;
                return;
            }
        }
        // The row as z = (x1..xk, y).
        std::array<double, D> z;
        for (std::size_t j = 0; j < k_; ++j) z[j] = in[j + 1];
        z[k_] = in[0];

        double pred = nan;
        if (fitted_) {
            pred = b0_;
            for (std::size_t j = 0; j < k_; ++j) {
                const double term = b_[j] * z[j];
                pred += term;
            }
        }
        out[0] = pred;
        out[1] = z[k_] - pred;

        if (w_) push_rolling(z.data());
        else add(z.data());
        fit();

        out[2] = fitted_ ? r2_ : nan;
        std::size_t o = 3;
        if (icpt_) out[o++] = fitted_ ? b0_ : nan;
        for (std::size_t j = 0; j < k_; ++j) out[o + j] = fitted_ ? b_[j] : nan;
    }

    void reset() override { reset_fit(); }

protected:
    static std::size_t check_k(int k) {
        if (k < 1 || k > static_cast<int>(kMaxOLSRegressors))
            throw std::invalid_argument("n_regressors must be between 1 and 16.");
        return static_cast<std::size_t>(k);
    }

private:
    void reset_fit() {
        n_ = 0;
        head_ = 0;
        since_refresh_ = 0;
        mean_.fill(0.0);
        origin_.fill(0.0);
        c_.fill(0.0);
        fitted_ = false;
        b0_ = r2_ = 0.0;
        b_.fill(0.0);
    }

    double& c(std::size_t i, std::size_t j) { return c_[i * D + j]; }

    // Adds a row to the sums (upper triangle, i <= j). With an intercept the
    // row enters as z - origin, which is exact while z is within a factor 2
    // of the origin, and mean_ is the mean of z - origin.
    void add(const double* z) {
        if (icpt_ && n_ == 0)
            for (std::size_t i = 0; i < d_; ++i) origin_[i] = z[i];
        n_ += 1;
        if (icpt_) {
            const double nd = static_cast<double>(n_);
            std::array<double, D> dz, e;
            for (std::size_t i = 0; i < d_; ++i) {
                const double v = z[i] - origin_[i];
                dz[i] = v - mean_[i];
                mean_[i] += dz[i] / nd;
                e[i] = v - mean_[i];
            }
            for (std::size_t i = 0; i < d_; ++i)
                for (std::size_t j = i; j < d_; ++j) {
                    const double t = dz[i] * e[j];
                    c(i, j) += t;
                }
            recenter();
        } else {
            for (std::size_t i = 0; i < d_; ++i)
                for (std::size_t j = i; j < d_; ++j) {
                    const double t = z[i] * z[j];
                    c(i, j) += t;
                }
        }
    }

    // Removes a row from the sums. Returns true when the row was far out of
    // the rows that remain, or not finite: its rounding may then dominate.
    bool remove(const double* z) {
        n_ -= 1;
        bool far = false;
        if (n_ == 0) {
            mean_.fill(0.0);
            c_.fill(0.0);
            return false;
        }
        if (icpt_) {
            const double nd = static_cast<double>(n_);
            std::array<double, D> dz, e;
            for (std::size_t i = 0; i < d_; ++i) {
                const double v = z[i] - origin_[i];
                dz[i] = v - mean_[i];
                mean_[i] -= dz[i] / nd;
                e[i] = v - mean_[i];
            }
            for (std::size_t i = 0; i < d_; ++i)
                for (std::size_t j = i; j < d_; ++j) {
                    const double t = dz[i] * e[j];
                    c(i, j) -= t;
                }
            for (std::size_t i = 0; i < d_; ++i) {
                const double sq = dz[i] * e[i];
                if (!(sq <= kFar * c(i, i))) far = true;
            }
        } else {
            for (std::size_t i = 0; i < d_; ++i)
                for (std::size_t j = i; j < d_; ++j) {
                    const double t = z[i] * z[j];
                    c(i, j) -= t;
                }
            for (std::size_t i = 0; i < d_; ++i) {
                const double sq = z[i] * z[i];
                if (!(sq <= kFar * c(i, i))) far = true;
            }
        }
        return far;
    }

    void push_rolling(const double* z) {
        double* slot = ring_.data() + head_ * d_;
        bool far = false;
        if (n_ == w_) far = remove(slot);
        std::copy(z, z + d_, slot);
        add(z);
        head_ = head_ + 1 == w_ ? 0 : head_ + 1;
        if (far || ++since_refresh_ >= w_) refresh();
    }

    // Recomputes the sums from the stored rows, in two passes with an
    // intercept (the means, then the centered cross-products).
    void refresh() {
        since_refresh_ = 0;
        const std::size_t n = n_;
        // The stored rows are the n slots before head_ (all slots once full).
        const std::size_t first = n == w_ ? 0 : (head_ + w_ - n) % w_;
        auto row = [&](std::size_t r) { return ring_.data() + ((first + r) % w_) * d_; };
        c_.fill(0.0);
        if (icpt_) {
            // The origin moves to the window mean, taken about the old origin
            // (or a finite stored value, after an infinity), then the means of
            // z - origin and the centered cross-products in a second pass.
            const double nd = static_cast<double>(n);
            for (std::size_t i = 0; i < d_; ++i) {
                double base = origin_[i];
                for (std::size_t r = 0; r < n && !std::isfinite(base); ++r)
                    if (std::isfinite(row(r)[i])) base = row(r)[i];
                if (!std::isfinite(base)) base = 0.0;
                double s1 = 0.0;
                for (std::size_t r = 0; r < n; ++r) s1 += row(r)[i] - base;
                const double shift = s1 / nd;
                origin_[i] = std::isfinite(shift) ? base + shift : base;
                double s2 = 0.0;
                for (std::size_t r = 0; r < n; ++r) s2 += row(r)[i] - origin_[i];
                mean_[i] = s2 / nd;
            }
            for (std::size_t r = 0; r < n; ++r) {
                const double* z = row(r);
                std::array<double, D> e;
                for (std::size_t i = 0; i < d_; ++i) e[i] = (z[i] - origin_[i]) - mean_[i];
                for (std::size_t i = 0; i < d_; ++i)
                    for (std::size_t j = i; j < d_; ++j) {
                        const double t = e[i] * e[j];
                        c(i, j) += t;
                    }
            }
        } else {
            for (std::size_t r = 0; r < n; ++r) {
                const double* z = row(r);
                for (std::size_t i = 0; i < d_; ++i)
                    for (std::size_t j = i; j < d_; ++j) {
                        const double t = z[i] * z[j];
                        c(i, j) += t;
                    }
            }
        }
    }

    // Moves the origin of a column to its mean when the mean has drifted more
    // than four standard deviations from it, so z - origin stays near the
    // spread. The centered cross-products do not depend on the origin.
    void recenter() {
        const double nd = static_cast<double>(n_);
        for (std::size_t i = 0; i < d_; ++i) {
            const double m = mean_[i];
            const double m2 = m * m;
            if (m2 * nd > kDrift * c(i, i) && std::isfinite(m)) {
                const double o = origin_[i] + m;
                const double moved = o - origin_[i];
                origin_[i] = o;
                mean_[i] = m - moved;
            }
        }
    }

    // Solves the k x k normal equations by Cholesky: A b = r with A = c(x, x)
    // and r = c(x, y).
    void fit() {
        fitted_ = false;
        if (n_ < n_min_) return;
        const std::size_t k = k_;
        // l_ holds the lower factor, row-major with stride D.
        for (std::size_t j = 0; j < k; ++j) {
            const double ajj = c(j, j);
            double s = ajj;
            for (std::size_t q = 0; q < j; ++q) {
                const double t = l_[j * D + q] * l_[j * D + q];
                s -= t;
            }
            if (!(s > kSingular * ajj)) return;   // collinear, constant, or NaN
            const double ljj = std::sqrt(s);
            l_[j * D + j] = ljj;
            for (std::size_t i = j + 1; i < k; ++i) {
                double v = c(j, i);
                for (std::size_t q = 0; q < j; ++q) {
                    const double t = l_[i * D + q] * l_[j * D + q];
                    v -= t;
                }
                l_[i * D + j] = v / ljj;
            }
        }
        // Forward then back substitution.
        std::array<double, D> u;
        for (std::size_t i = 0; i < k; ++i) {
            double v = c(i, k);
            for (std::size_t q = 0; q < i; ++q) {
                const double t = l_[i * D + q] * u[q];
                v -= t;
            }
            u[i] = v / l_[i * D + i];
        }
        for (std::size_t i = k; i-- > 0;) {
            double v = u[i];
            for (std::size_t q = i + 1; q < k; ++q) {
                const double t = l_[q * D + i] * b_[q];
                v -= t;
            }
            b_[i] = v / l_[i * D + i];
        }
        // SSE = c(y, y) - b . c(x, y); r2 = 1 - SSE / c(y, y).
        const double syy = c(k, k);
        double explained = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
            const double t = b_[i] * c(i, k);
            explained += t;
        }
        const double sse = std::max(syy - explained, 0.0);
        r2_ = syy > 0.0 ? 1.0 - sse / syy : std::numeric_limits<double>::quiet_NaN();
        b0_ = 0.0;
        if (icpt_) {
            // mean_y - b . mean_x, with the means restored from the origins.
            double v = origin_[k] + mean_[k];
            for (std::size_t i = 0; i < k; ++i) {
                const double mi = origin_[i] + mean_[i];
                const double t = b_[i] * mi;
                v -= t;
            }
            b0_ = v;
        }
        fitted_ = std::isfinite(b0_);
        for (std::size_t i = 0; i < k && fitted_; ++i) fitted_ = std::isfinite(b_[i]);
    }

    static constexpr double kFar = 16.0;
    static constexpr double kDrift = 16.0;
    static constexpr double kSingular = 1e-12;

    const std::size_t k_, d_, m_;   // regressors, row width, outputs
    const bool icpt_;
    const std::size_t w_, n_min_;
    std::vector<double> ring_;
    std::size_t n_ = 0, head_ = 0, since_refresh_ = 0;
    std::array<double, D> mean_{};     // mean of z - origin (with an intercept)
    std::array<double, D> origin_{};
    std::array<double, D * D> c_{};
    std::array<double, D * D> l_{};
    std::array<double, kMaxOLSRegressors> b_{};
    double b0_ = 0.0, r2_ = 0.0;
    bool fitted_ = false;
};

}  // namespace detail

class RollingOLS : public detail::OLSFit<RollingOLS> {
public:
    RollingOLS(int window_size, int n_regressors, bool intercept = true)
        : detail::OLSFit<RollingOLS>(check_window(window_size, n_regressors, intercept),
                                     n_regressors, intercept) {}

private:
    static std::size_t check_window(int w, int k, bool intercept) {
        check_k(k);
        const int p = k + (intercept ? 1 : 0);
        if (w < p + 1)
            throw std::invalid_argument(
                "window_size must be at least n_regressors + 1, plus 1 with an intercept.");
        return static_cast<std::size_t>(w);
    }
};

class ExpandingOLS : public detail::OLSFit<ExpandingOLS> {
public:
    ExpandingOLS(int n_regressors, bool intercept = true)
        : detail::OLSFit<ExpandingOLS>(0, n_regressors, intercept) {}
};

}  // namespace screamer

#endif
