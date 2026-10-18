#ifndef SCREAMER_DETAIL_EW_FLAT_H
#define SCREAMER_DETAIL_EW_FLAT_H

// The test for an exponentially weighted series that has stopped varying,
// shared by the operators whose ratio with its weighted variance turns into
// rounding noise (EwBeta for its regressor, EwSkew, EwKurt). EwCorr and
// EwZscore do not use it: with one input flat their values stay exact (they
// decay toward 0 with the exact value), so they only give NaN where their
// denominator underflows.
//
// Once a series stops varying, the weighted spread of its earlier rows decays
// as (1 - alpha)^n while the sums keep the rounding of its running mean, so a
// ratio with its variance in the denominator loses a digit for every factor
// 10 the variance falls, and in the end returns rounding noise (or, after the
// variance underflows, values such as 1, 2 or -1). The exact value is set by
// rows whose weight is below (1 - alpha)^n; it is not identified by the data.
//
// The series counts as flat when its weighted variance (per unit weight) is
// at most kEwFlat^2 times a slowly decaying peak of it. The peak is a running
// maximum that decays by (1 - alpha)^(1/8) per row, eight times slower than
// the weights. A flat series' variance falls away from it as
// (1 - alpha)^(7n/8), so it is flagged after about
// 2 ln(1 / kEwFlat) * 8 / 7 / alpha, some 45 / alpha rows (about 430 at
// alpha = 0.1); up to that row the ratios keep 5 digits or more. An old value
// R times the current variance is forgotten: it stops marking the series
// after about 8 ln(R kEwFlat^2) / alpha rows (about 1070 rows at alpha = 0.1
// after one 1e10 tick in returns of spread 0.01), and never marks it when
// R kEwFlat^2 < 1 (a 1e8 tick there).
//
// kEwFlat = 2^24 eps (3.7e-9). A real series whose variance falls more than
// 16 orders of magnitude below its decaying peak is also flagged, until the
// peak has decayed. A series that is constant from the start has variance 0
// and is flagged from the first row.

#include <cmath>
#include <limits>

namespace screamer {
namespace detail {

inline constexpr double kEwFlat = 16777216.0 * std::numeric_limits<double>::epsilon();
inline constexpr double kEwFlat2 = kEwFlat * kEwFlat;

// The per-row decay of the peak, (1 - alpha)^(1/8), by three square roots:
// correctly rounded, so every binding gets the same value. Set once.
inline double ew_flat_peak_decay(double one_minus_alpha) {
    return std::sqrt(std::sqrt(std::sqrt(one_minus_alpha)));
}

// The decaying peak of one series' weighted variance.
struct EwFlatPeak {
    double peak = 0.0;

    void reset() { peak = 0.0; }

    // Decays the peak, raises it to var, and returns true when var is at most
    // kEwFlat^2 times the peak (also when var is NaN or not positive; a NaN
    // leaves the peak decayed). Call once per row, with the variance of that
    // row.
    inline bool flat(double var, double peak_decay) {
        const double old = peak * peak_decay;
        peak = std::fmax(var, old);
        const double floor = kEwFlat2 * peak;
        return !(var > floor);
    }
};

}  // namespace detail
}  // namespace screamer

#endif  // SCREAMER_DETAIL_EW_FLAT_H
