#ifndef SCREAMER_MULTI_REGRESSION_H
#define SCREAMER_MULTI_REGRESSION_H

// RollingMultiRegression, ExpandingMultiRegression and EwMultiRegression:
// least squares of y on k regressors,
//
//     y = b0 + b1 x1 + ... + bk xk + e      (b0 only with an intercept)
//
// over a trailing window of rows (RollingMultiRegression), over every row so
// far (ExpandingMultiRegression), or over every row so far with exponentially
// decaying weights (EwMultiRegression: the row i rows back has weight
// (1 - alpha)^i, the pandas adjust=True convention of the other Ew operators).
// Inputs are y, x1, ..., xk (1 <= k <= 16). Outputs, per row:
//
//     pred   b0 + b . x_t with the coefficients fitted on the rows before t,
//            a one-step-ahead forecast of y_t
//     resid  y_t - pred
//     r2     R^2 of the fit after row t enters
//     b0     the intercept after row t enters (with an intercept only)
//     b1..bk the coefficients after row t enters
//
// A fit needs a full window (RollingMultiRegression) or p + 1 rows (the other
// two), with p = k + 1 parameters with an intercept and p = k without; r2 and
// the coefficients are NaN before that, and pred is NaN until the row after
// the first fit. A row with a NaN in any input is skipped: it does not enter
// the window (nor decay the weights) and its outputs are NaN.
//
// The operator keeps the (weighted) means of the columns and their centered
// cross-products, updated in O(k^2) per row (Welford's add and remove; West's
// weighted add, after scaling the sums by 1 - alpha, for EwMultiRegression).
// Each column enters as z - origin, with the origin near the column's mean
// (moved there when the mean drifts more than four standard deviations from
// it, and at every exact recomputation), so the sums stay accurate when the
// levels are large next to the spread. Each row then solves the k x k normal
// equations by Cholesky, O(k^3), so a step costs O(k^3) in all. With an
// intercept they are the centered ones; the intercept is mean_y - b . mean_x
// and r2 = 1 - SSE / sum w (y - mean_y)^2. Without an intercept they are the
// raw ones, X'WX = C + W m m' (W the total weight), formed at the solve, and
// r2 is the uncentered 1 - SSE / sum w y^2 (the statsmodels convention for a
// model without a constant). The weights w are 1 except in
// EwMultiRegression. The normal equations square the condition number of the
// regressors: without an intercept on levels far from zero (prices), the
// coefficients keep about 16 - 2 log10(cond) digits, where cond is the
// condition number of the (weighted) regressor matrix.
//
// RollingMultiRegression recomputes its cross-products exactly from the stored
// rows every window_size rows, and at once when the row that left was far out
// of the rest (its squared deviation more than 16 times the remaining sum of
// squares in some column, or not finite), so rounding from the removals stays
// bounded. The other two never remove a row and need no recomputation. A fit
// whose regressors are collinear (a pivot of the Cholesky factor below 1e-12
// of its diagonal, for example a regressor that is constant in the window with
// an intercept) is singular: r2 and the coefficients are NaN, as is the next
// pred. In EwMultiRegression a regressor that stops varying keeps a spread
// that decays geometrically toward the rounding of its mean, which the pivot
// test does not see; the fit is also singular once a regressor's spread (its
// root mean square without an intercept) falls below 2.3e-10 of the largest
// root mean square it has had (kFlat).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>
#include "screamer/common/float_info.h"
#include "screamer/common/var_arity_op.h"

namespace screamer {
namespace detail {

constexpr std::size_t kMaxOLSRegressors = 16;

// The fit shared by RollingMultiRegression (window > 0),
// ExpandingMultiRegression (window 0) and, with kEw, EwMultiRegression
// (window 0, every weight scaled by decay before each row enters).
template <class Derived, bool kEw = false>
class OLSFit : public VarArityOp<Derived> {
    static constexpr std::size_t D = kMaxOLSRegressors + 1;   // x1..xk, y

public:
    OLSFit(std::size_t window, int n_regressors, bool intercept, double decay = 1.0)
        : VarArityOp<Derived>(check_k(n_regressors) + 1,
                              static_cast<std::size_t>(n_regressors) + (intercept ? 4 : 3)),
          k_(static_cast<std::size_t>(n_regressors)), d_(k_ + 1), m_(k_ + (intercept ? 4 : 3)),
          icpt_(intercept), w_(window),
          n_min_(window ? window : k_ + (intercept ? 2 : 1)), decay_(decay)
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

        if (!kEw && w_) push_rolling(z.data());
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
        wsum_ = 0.0;
        head_ = 0;
        since_refresh_ = 0;
        mean_.fill(0.0);
        origin_.fill(0.0);
        c_.fill(0.0);
        fitted_ = false;
        b0_ = r2_ = 0.0;
        b_.fill(0.0);
        peak_.fill(0.0);
    }

    double& c(std::size_t i, std::size_t j) { return c_[i * D + j]; }

    // The total weight of the rows in the sums: their count, or with kEw the
    // sum of their decayed weights.
    double weight() const {
        if constexpr (kEw) return wsum_;
        else return static_cast<double>(n_);
    }

    // Adds a row with weight 1 to the sums (upper triangle, i <= j). The row
    // enters as z - origin, which is exact while z is within a factor 2 of
    // the origin; mean_ is the weighted mean of z - origin and c_ holds the
    // centered cross-products, with or without an intercept. With kEw every
    // earlier weight is first multiplied by decay_: the total weight and the
    // cross-products scale with it (in the same pass), the means do not.
    void add(const double* z) {
        if (n_ == 0)
            for (std::size_t i = 0; i < d_; ++i) origin_[i] = z[i];
        n_ += 1;
        if constexpr (kEw) {
            const double scaled = wsum_ * decay_;
            wsum_ = scaled + 1.0;
        }
        const double nd = weight();
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
                if constexpr (kEw) {
                    const double scaled = c(i, j) * decay_;
                    c(i, j) = scaled + t;
                } else {
                    c(i, j) += t;
                }
            }
        recenter();
    }

    // Removes a row from the sums. Returns true when the row was far out of
    // the rows that remain, or not finite: its rounding may then dominate.
    bool remove(const double* z) {
        n_ -= 1;
        if (n_ == 0) {
            mean_.fill(0.0);
            c_.fill(0.0);
            return false;
        }
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
        bool far = false;
        for (std::size_t i = 0; i < d_; ++i) {
            const double sq = dz[i] * e[i];
            if (!(sq <= kFar * c(i, i))) far = true;
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

    // Recomputes the sums from the stored rows in two passes. The origin
    // moves to the window mean, taken about the old origin (or a finite
    // stored value, after an infinity), then the means of z - origin and the
    // centered cross-products.
    void refresh() {
        since_refresh_ = 0;
        const std::size_t n = n_;
        // The stored rows are the n slots before head_ (all slots once full).
        const std::size_t first = n == w_ ? 0 : (head_ + w_ - n) % w_;
        auto row = [&](std::size_t r) { return ring_.data() + ((first + r) % w_) * d_; };
        c_.fill(0.0);
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
    }

    // Moves the origin of a column to its mean when the mean has drifted more
    // than four standard deviations from it, so z - origin stays near the
    // spread. The centered cross-products do not depend on the origin.
    void recenter() {
        const double nd = weight();
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

    // Solves the k x k normal equations A b = r by Cholesky. With an
    // intercept A = c(x, x) and r = c(x, y), the centered cross-products.
    // Without one A and r are the raw cross-products, formed here as
    // c + n mean mean' from the centered sums, so no cancellation builds up
    // in them; their condition number is that of the regressors squared.
    void fit() {
        fitted_ = false;
        const std::size_t k = k_;
        const double nd = weight();
        if constexpr (kEw) {
            // The largest second moment per unit weight, sum w z^2 / W, that
            // each regressor has had, the reference of the test below.
            for (std::size_t j = 0; j < k; ++j) {
                const double level = origin_[j] + mean_[j];
                const double l2 = level * level;
                const double m2 = c(j, j) / nd + l2;
                if (m2 > peak_[j]) peak_[j] = m2;
            }
        }
        if (n_ < n_min_) return;
        std::array<double, D> mu{};          // the column means, without an intercept
        if (!icpt_)
            for (std::size_t i = 0; i <= k; ++i) mu[i] = origin_[i] + mean_[i];
        auto a = [&](std::size_t i, std::size_t j) {   // i <= j
            double v = c(i, j);
            if (!icpt_) {
                const double t = nd * mu[i] * mu[j];
                v += t;
            }
            return v;
        };
        // l_ holds the lower factor, row-major with stride D, and inv_ the
        // reciprocals of its diagonal.
        for (std::size_t j = 0; j < k; ++j) {
            const double ajj = a(j, j);
            double s = ajj;
            for (std::size_t q = 0; q < j; ++q) {
                const double t = l_[j * D + q] * l_[j * D + q];
                s -= t;
            }
            if (!(s > kSingular * ajj)) return;   // collinear, constant, or NaN
            if constexpr (kEw) {
                // A regressor that has stopped varying: its weighted spread
                // decays geometrically while the sums keep the rounding of
                // its mean (a few ulps of its level), or underflow, so the
                // pivot test above cannot see it. It counts as constant once
                // its diagonal per unit weight is below kFlat^2 times the
                // largest second moment it has had; see kFlat.
                const double ref = kFlat2 * nd;
                const double floor = ref * peak_[j];
                if (!(ajj > floor)) return;
            }
            const double inv = 1.0 / std::sqrt(s);
            inv_[j] = inv;
            for (std::size_t i = j + 1; i < k; ++i) {
                double v = a(j, i);
                for (std::size_t q = 0; q < j; ++q) {
                    const double t = l_[i * D + q] * l_[j * D + q];
                    v -= t;
                }
                l_[i * D + j] = v * inv;
            }
        }
        // Forward then back substitution.
        std::array<double, D> r, u;
        for (std::size_t i = 0; i < k; ++i) r[i] = a(i, k);
        for (std::size_t i = 0; i < k; ++i) {
            double v = r[i];
            for (std::size_t q = 0; q < i; ++q) {
                const double t = l_[i * D + q] * u[q];
                v -= t;
            }
            u[i] = v * inv_[i];
        }
        for (std::size_t i = k; i-- > 0;) {
            double v = u[i];
            for (std::size_t q = i + 1; q < k; ++q) {
                const double t = l_[q * D + i] * b_[q];
                v -= t;
            }
            b_[i] = v * inv_[i];
        }
        // SSE = a(y, y) - b . r; r2 = 1 - SSE / a(y, y).
        const double syy = a(k, k);
        double explained = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
            const double t = b_[i] * r[i];
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
    // EwMultiRegression: a regressor whose weighted standard deviation (with
    // an intercept; its root mean square without one) is below kFlat times
    // the root of the largest second moment it has had, sum w z^2 / W, is
    // constant. Once a regressor stops varying, the weighted spread of its
    // earlier rows decays geometrically while the sums keep the rounding of
    // its running mean, and the slope loses a digit for every factor 10 the
    // spread falls toward that rounding. kFlat = 2^20 eps (2.3e-10) stops
    // the fit while it still has about 7 digits (in tests, a varying
    // regressor that then stays at 3.7, 1, 0 or -5); a real spread that
    // small next to the column's level or history is rare: a regressor at
    // level 3.7 moving by 1e-5 over the effective window is thousands of
    // times above it.
    static constexpr double kFlat = 1048576.0 * std::numeric_limits<double>::epsilon();
    static constexpr double kFlat2 = kFlat * kFlat;

    const std::size_t k_, d_, m_;   // regressors, row width, outputs
    const bool icpt_;
    const std::size_t w_, n_min_;
    const double decay_;             // 1 - alpha with kEw
    std::vector<double> ring_;
    std::size_t n_ = 0, head_ = 0, since_refresh_ = 0;
    double wsum_ = 0.0;              // the total weight, with kEw
    std::array<double, D> mean_{};     // mean of z - origin
    std::array<double, D> origin_{};
    std::array<double, D * D> c_{};
    std::array<double, D * D> l_{};
    std::array<double, D> inv_{};
    std::array<double, kMaxOLSRegressors> b_{};
    std::array<double, kMaxOLSRegressors> peak_{};   // with kEw
    double b0_ = 0.0, r2_ = 0.0;
    bool fitted_ = false;
};

}  // namespace detail

class RollingMultiRegression : public detail::OLSFit<RollingMultiRegression> {
public:
    RollingMultiRegression(int window_size, int n_regressors, bool intercept = true)
        : detail::OLSFit<RollingMultiRegression>(check_window(window_size, n_regressors, intercept),
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

class ExpandingMultiRegression : public detail::OLSFit<ExpandingMultiRegression> {
public:
    ExpandingMultiRegression(int n_regressors, bool intercept = true)
        : detail::OLSFit<ExpandingMultiRegression>(0, n_regressors, intercept) {}
};

class EwMultiRegression : public detail::OLSFit<EwMultiRegression, true> {
public:
    EwMultiRegression(int n_regressors, bool intercept = true,
                      std::optional<double> com = std::nullopt,
                      std::optional<double> span = std::nullopt,
                      std::optional<double> halflife = std::nullopt,
                      std::optional<double> alpha = std::nullopt)
        : detail::OLSFit<EwMultiRegression, true>(
              0, n_regressors, intercept, decay(n_regressors, com, span, halflife, alpha)) {}

private:
    // 1 - alpha, with alpha from exactly one of com, span, halflife and alpha
    // as in EwBeta.
    static double decay(int k, std::optional<double> com, std::optional<double> span,
                        std::optional<double> halflife, std::optional<double> alpha) {
        check_k(k);
        const int provided = (com.has_value() ? 1 : 0) + (span.has_value() ? 1 : 0)
                           + (halflife.has_value() ? 1 : 0) + (alpha.has_value() ? 1 : 0);
        if (provided != 1)
            throw std::invalid_argument("Exactly one of com, span, halflife, or alpha must be provided");
        double a;
        if (alpha.has_value())          a = alpha.value();
        else if (com.has_value())       a = 1.0 / (1.0 + com.value());
        else if (span.has_value())      a = 2.0 / (span.value() + 1.0);
        else                            a = 1.0 - std::exp(-std::log(2.0) / halflife.value());
        if (!std::isfinite(a) || a <= 0.0 || a >= 1.0)
            throw std::invalid_argument("Alpha must be a finite value between 0 and 1 (exclusive)");
        return 1.0 - a;
    }
};

}  // namespace screamer

#endif
