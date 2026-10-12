#ifndef SCREAMER_DETAIL_SHIFTED_POWER_SUMS_H
#define SCREAMER_DETAIL_SHIFTED_POWER_SUMS_H

// Weighted power sums about an origin that follows the mean, for the
// expanding and exponentially weighted moment operators.
//
// The moments follow from the weighted power sums w = sum w_i and
// s_p = sum w_i (x_i - origin)^p. Taken about origin 0 (raw sums), they cancel
// when the mean is large next to the spread: on a price near 1000 with unit
// moves the fourth central moment keeps about 4 of 16 digits. Here the origin
// starts at the first sample and moves to the current mean whenever the mean
// has drifted from it by more than the (weighted) standard deviation, that is
// when 2 s_1^2 > s_2 w (tested every fourth sample; the two-input version
// allows three standard deviations). The sums are then shifted to the new origin with the
// binomial formulas; at that point the shift is at most about one standard
// deviation, so the shift itself loses no precision. Between shifts every
// term is within about a standard deviation of the mean, and the formulas
// that turn power sums into central moments (unchanged by the shift) do not
// cancel. The mean is origin + s_1 / w.
//
// decay(a) multiplies every weight by a: the exponentially weighted operators
// call it before each add; the expanding operators never call it, and their
// weight is the sample count.
//
// Each product is its own statement. Clang fuses a * b + c into one multiply-
// add only within an expression, and whether it does there can depend on where
// the code is inlined; with no product inside a sum, an update inlined into a
// batch loop rounds exactly like the same update called once per event.

#include <algorithm>

namespace screamer {
namespace detail {

// One series, powers 1..P with P in 2..4.
template <int P>
struct ShiftedPowerSums {
    static_assert(P >= 2 && P <= 4, "ShiftedPowerSums supports powers 2 to 4");

    double w = 0.0;
    double origin = 0.0;
    double s1 = 0.0;
    double s2 = 0.0;
    double s3 = 0.0;
    double s4 = 0.0;
    unsigned tick = 0;

    void reset() {
        w = origin = s1 = s2 = s3 = s4 = 0.0;
        tick = 0;
    }

    inline void decay(double a) {
        w *= a;
        s1 *= a;
        s2 *= a;
        if constexpr (P >= 3) s3 *= a;
        if constexpr (P >= 4) s4 *= a;
    }

    // Adds x with weight 1.
    inline void add(double x) {
        if (w == 0.0) {  // the first sample since reset: it is the origin
            origin = x;
            w = 1.0;
            return;
        }
        const double d = x - origin;
        const double dd = d * d;
        s1 += d;
        s2 += dd;
        if constexpr (P >= 3) {
            const double ddd = dd * d;
            s3 += ddd;
        }
        if constexpr (P >= 4) {
            const double dddd = dd * dd;
            s4 += dddd;
        }
        w += 1.0;
        // The test runs every fourth sample, which keeps its cost low; the
        // mean moves little in four samples.
        if ((++tick & 3u) != 0) return;
        const double s1s1 = s1 * s1;
        const double s2w = s2 * w;
        if (s1s1 + s1s1 > s2w) *this = recentered(*this);
    }

private:
    // Moves the origin to the mean and shifts the sums with it. By value, so
    // a batch loop's local sums do not escape and stay in registers.
    static ShiftedPowerSums recentered(ShiftedPowerSums p) {
        const double w = p.w;
        double& s1 = p.s1;
        double& s2 = p.s2;
        [[maybe_unused]] double& s3 = p.s3;
        [[maybe_unused]] double& s4 = p.s4;
        // The shift is the exact step of the stored origin, which differs from
        // the mean s1 / w by the rounding of origin + mean; shifting the sums
        // by the mean instead would leave them about a point a rounding of
        // the level away from the origin.
        const double new_origin = p.origin + s1 / w;
        const double c = new_origin - p.origin;
        const double c2 = c * c;
        const double cs1 = c * s1;
        const double wc2 = w * c2;
        if constexpr (P >= 4) {
            const double c3 = c2 * c;
            const double t1 = 4.0 * c * s3;
            const double t2 = 6.0 * c2 * s2;
            const double t3 = 4.0 * c3 * s1;
            const double t4 = wc2 * c2;
            s4 = s4 - t1 + t2 - t3 + t4;
        }
        if constexpr (P >= 3) {
            const double c3 = c2 * c;
            const double t1 = 3.0 * c * s2;
            const double t2 = 3.0 * c2 * s1;
            const double t3 = w * c3;
            s3 = s3 - t1 + t2 - t3;
        }
        const double t = 2.0 * cs1;
        s2 = s2 - t + wc2;
        const double wc = w * c;
        s1 = s1 - wc;
        p.origin = new_origin;
        return p;
    }
};

// Two series: the weights, origins and sums sx, sy, sxy, and sxx and syy when
// enabled (kXX, kYY), which is what the operator's output needs. The drift
// test needs a spread for both inputs; for an input whose sum of squares is
// not kept, a coarse one does: vxx (vyy) decays every sample like the sums but
// adds 4 dx^2 only on the samples that run the test, every fourth.
template <bool kXX, bool kYY>
struct ShiftedCoPowerSums {
    double w = 0.0;
    double origin_x = 0.0;
    double origin_y = 0.0;
    double sx = 0.0;
    double sy = 0.0;
    double sxx = 0.0;  // the sum, or the coarse spread vxx when !kXX
    double syy = 0.0;  // likewise
    double sxy = 0.0;
    unsigned tick = 0;

    void reset() {
        w = origin_x = origin_y = sx = sy = sxx = syy = sxy = 0.0;
        tick = 0;
    }

    inline void decay(double a) {
        w *= a;
        sx *= a;
        sy *= a;
        sxx *= a;
        syy *= a;
        sxy *= a;
    }

    inline void add(double x, double y) {
        if (w == 0.0) {  // the first pair since reset: it is the origin
            origin_x = x;
            origin_y = y;
            w = 1.0;
            return;
        }
        const double dx = x - origin_x;
        const double dy = y - origin_y;
        const double dxy = dx * dy;
        sx += dx;
        sy += dy;
        sxy += dxy;
        if constexpr (kXX) {
            const double dxx = dx * dx;
            sxx += dxx;
        }
        if constexpr (kYY) {
            const double dyy = dy * dy;
            syy += dyy;
        }
        w += 1.0;
        // The test runs every fourth sample and allows the means three
        // standard deviations from the origins: second moments lose at most a
        // factor 1 + 3^2 = 10 to cancellation, and the test and the shift
        // cost little.
        if ((++tick & 3u) != 0) return;
        if constexpr (!kXX) {
            const double dxx4 = 4.0 * dx * dx;
            sxx += dxx4;
        }
        if constexpr (!kYY) {
            const double dyy4 = 4.0 * dy * dy;
            syy += dyy4;
        }
        const double sxsx = sx * sx;
        const double sysy = sy * sy;
        const double sxxw = 0.9 * sxx * w;
        const double syyw = 0.9 * syy * w;
        if (sxsx > sxxw || sysy > syyw) {
            // Move the origins to the means and shift the sums by the exact
            // step of each origin (see ShiftedPowerSums::recentered).
            const double new_ox = origin_x + sx / w;
            const double new_oy = origin_y + sy / w;
            const double cx = new_ox - origin_x;
            const double cy = new_oy - origin_y;
            const double cxsx = cx * sx;
            const double cysy = cy * sy;
            const double cxsy = cx * sy;
            const double cysx = cy * sx;
            const double wcx = w * cx;
            const double wcy = w * cy;
            const double wcxcx = wcx * cx;
            const double wcycy = wcy * cy;
            const double cxcy = cx * cy;
            const double wcxcy = w * cxcy;
            const double txx = 2.0 * cxsx;
            const double tyy = 2.0 * cysy;
            const double txy = cxsy + cysx;  // symmetric in x and y
            // The same shift for a coarse spread; it is only a test input, so
            // it is kept non-negative.
            sxx = sxx - txx + wcxcx;
            if constexpr (!kXX) sxx = std::max(sxx, 0.0);
            syy = syy - tyy + wcycy;
            if constexpr (!kYY) syy = std::max(syy, 0.0);
            sxy = sxy - txy + wcxcy;
            sx = sx - wcx;
            sy = sy - wcy;
            origin_x = new_ox;
            origin_y = new_oy;
        }
    }
};

}  // namespace detail
}  // namespace screamer

#endif  // SCREAMER_DETAIL_SHIFTED_POWER_SUMS_H
