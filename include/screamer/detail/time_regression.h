#ifndef SCREAMER_DETAIL_TIME_REGRESSION_H
#define SCREAMER_DETAIL_TIME_REGRESSION_H

// Least-squares fits of a line or a parabola against the time index
// t = 0, 1, ..., n - 1, from the sums that ShiftedWindowSums<P, T> and
// ShiftedPowerSums<P, true> keep: s0 = sum d and ts1 = sum t d (and
// ts2 = sum t^2 d for the parabola), with d = x - origin. The sums of powers of
// t alone have closed forms, which the constants below hold for one n.
//
// The fit is written about the mean time tbar = (n - 1) / 2 and the mean of d,
// so that no term carries the level:
//
//   line:      slope = sum (t - tbar) d / sum (t - tbar)^2
//                    = (ts1 - tbar s0) / (n (n^2 - 1) / 12)
//              value at t = origin + s0 / n + slope (t - tbar)
//
//   parabola:  y = a t^2 + b t + c, with u = t - tbar and mu2 = mean u^2,
//              a = sum (u^2 - mu2) d / sum (u^2 - mu2)^2
//                = (ts2 - m2 s0 - (n - 1) (ts1 - tbar s0)) / Q
//              b = (ts1 - tbar s0) / sum u^2 - (n - 1) a
//              with m2 = mean t^2 = (n - 1)(2n - 1) / 6 and
//              Q = sum (u^2 - mu2)^2 = n (n^2 - 1)(n^2 - 4) / 180.
//              value at t = n - 1: origin + s0 / n + a ((n - 1)^2 - m2) + b tbar
//
// The slope and curvature do not depend on the origin; the fitted values are
// the origin plus terms of the size of the spread. Each product is its own
// statement (see shifted_window_sums.h).

namespace screamer {
namespace detail {

// The constants of a line fit over n points.
struct TimeLine {
    double inv_n = 0.0;
    double tbar = 0.0;     // (n - 1) / 2
    double inv_stt = 0.0;  // 1 / sum (t - tbar)^2 = 12 / (n (n^2 - 1))

    static TimeLine of(double n) {
        TimeLine c;
        c.inv_n = 1.0 / n;
        c.tbar = 0.5 * (n - 1.0);
        const double nn = n * n;
        c.inv_stt = 12.0 / (n * (nn - 1.0));
        return c;
    }
};

// sum (t - tbar) d = ts1 - tbar s0.
inline double centered_time_sum(double s0, double ts1, double tbar) {
    const double tb_s0 = tbar * s0;
    return ts1 - tb_s0;
}

inline double line_slope(double s0, double ts1, const TimeLine& c) {
    const double num = centered_time_sum(s0, ts1, c.tbar);
    return num * c.inv_stt;
}

// The fitted line at t, relative to the origin: s0 / n + slope (t - tbar).
inline double line_value_d(double s0, double slope, double t_minus_tbar, const TimeLine& c) {
    const double mean_d = s0 * c.inv_n;
    const double rise = slope * t_minus_tbar;
    return mean_d + rise;
}

// The constants of a parabola fit over n >= 3 points.
struct TimeParabola {
    TimeLine line;
    double nm1 = 0.0;    // n - 1
    double m2 = 0.0;     // mean t^2 = (n - 1)(2n - 1) / 6
    double inv_q = 0.0;  // 180 / (n (n^2 - 1)(n^2 - 4))
    double c_end = 0.0;  // (n - 1)^2 - m2

    static TimeParabola of(double n) {
        TimeParabola c;
        c.line = TimeLine::of(n);
        c.nm1 = n - 1.0;
        const double two_n = 2.0 * n;
        c.m2 = (n - 1.0) * (two_n - 1.0) / 6.0;
        const double n2 = n * n;
        c.inv_q = 180.0 / (n * (n2 - 1.0) * (n2 - 4.0));
        const double nm1_sq = c.nm1 * c.nm1;
        c.c_end = nm1_sq - c.m2;
        return c;
    }
};

struct ParabolaFit {
    double a;  // t^2 coefficient
    double b;  // t coefficient
};

inline ParabolaFit parabola_fit(double s0, double ts1, double ts2, const TimeParabola& c) {
    const double zxy = centered_time_sum(s0, ts1, c.line.tbar);   // sum (t - tbar) d
    const double m2_s0 = c.m2 * s0;
    const double zx2y = ts2 - m2_s0;                              // sum (t^2 - m2) d
    const double nm1_zxy = c.nm1 * zxy;
    const double num = zx2y - nm1_zxy;                            // sum (u^2 - mu2) d
    const double a = num * c.inv_q;
    const double b_line = zxy * c.line.inv_stt;
    const double nm1_a = c.nm1 * a;
    return ParabolaFit{a, b_line - nm1_a};
}

// The fitted parabola at t = n - 1, relative to the origin.
inline double parabola_end_d(double s0, const ParabolaFit& f, const TimeParabola& c) {
    const double mean_d = s0 * c.line.inv_n;
    const double ta = f.a * c.c_end;
    const double tb = f.b * c.line.tbar;
    return mean_d + ta + tb;
}

// The parabola's derivative at t = n - 1: 2 a (n - 1) + b.
inline double parabola_end_slope(const ParabolaFit& f, const TimeParabola& c) {
    const double two_a = 2.0 * f.a;
    const double rise = two_a * c.nm1;
    return rise + f.b;
}

}  // namespace detail
}  // namespace screamer

#endif  // SCREAMER_DETAIL_TIME_REGRESSION_H
