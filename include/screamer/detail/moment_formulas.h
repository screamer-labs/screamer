#ifndef SCREAMER_DETAIL_MOMENT_FORMULAS_H
#define SCREAMER_DETAIL_MOMENT_FORMULAS_H

// The sample moment estimators shared by the rolling and expanding operators,
// from power sums s1..s4 of n samples taken about any origin; the results do
// not depend on the origin. Each product is its own statement (see
// shifted_window_sums.h), so the formulas round the same wherever they are
// inlined, and every operator that reports a variance rounds it the same way.

#include <cmath>

namespace screamer {
namespace detail {

// Sum of squared deviations from the mean, M2 = s2 - s1^2 / n.
inline double m2_from_sums(double s1, double s2, double n) {
    const double s1s1 = s1 * s1;
    return s2 - s1s1 / n;
}

// Sample variance (ddof = 1).
inline double var_from_sums(double s1, double s2, double n) {
    return m2_from_sums(s1, s2, n) / (n - 1);
}

// n^2 times the population covariance, n sxy - sx sy; n sxx - sx^2 for a
// variance. The rolling two-input operators form their ratios from these.
inline double n2_comoment(double sx, double sy, double sxy, double n) {
    const double a = n * sxy;
    const double b = sx * sy;
    return a - b;
}

// Adjusted Fisher-Pearson skewness; c0 from skew_n_const(n).
inline double skew_from_sums(double s1, double s2, double s3, double n, double c0) {
    const double mean = s1 / n;
    const double var = var_from_sums(s1, s2, n);
    const double std_dev = std::sqrt(var);
    const double t1 = 3 * mean * s2;
    const double t2 = 2 * n * mean * mean * mean;
    const double m3 = s3 - t1 + t2;
    const double g1 = m3 / (var * std_dev);
    return g1 * c0;
}

// Bias-corrected excess kurtosis; c0, c1, c2 from kurt_n_const(n).
inline double kurt_from_sums(double s1, double s2, double s3, double s4, double n,
                             double c0, double c1, double c2) {
    const double mean = s1 / n;
    const double var = var_from_sums(s1, s2, n);
    const double mean2 = mean * mean;
    const double t1 = 4 * mean * s3;
    const double t2 = 6 * mean2 * s2;
    const double t3 = 3 * n * mean2 * mean2;
    const double m4 = s4 - t1 + t2 - t3;
    const double numerator = c0 * m4;
    const double denominator = c1 * var * var;
    return (numerator / denominator) - c2;
}

}  // namespace detail
}  // namespace screamer

#endif  // SCREAMER_DETAIL_MOMENT_FORMULAS_H
