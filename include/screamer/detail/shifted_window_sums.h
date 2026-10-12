#ifndef SCREAMER_DETAIL_SHIFTED_WINDOW_SUMS_H
#define SCREAMER_DETAIL_SHIFTED_WINDOW_SUMS_H

// Power sums of a sliding window, taken about an origin near the window's
// current values.
//
// The central moments of a window follow from its power sums, but sums of raw
// x^p cancel when the mean is large next to the spread: on a price near 1000
// with unit moves the fourth central moment keeps about 4 of 16 digits. The
// classes here sum powers of d = x - origin instead, with the origin close to
// the data, so every term is of the size of the spread and nothing cancels.
// The formulas that turn power sums into central moments are unchanged by the
// shift, and the window mean is origin + s1 / n.
//
// Re-centering moves the origin to the window mean and recomputes the sums
// exactly from the buffer. It runs when any of these holds after a step:
//
//   - the mean has drifted from the origin by more than four standard
//     deviations (s1^2 > (16/17) n s2), as after a level jump;
//   - the sample that left dominates what remains: o^2 > K s2 for the second
//     and third moments, o^4 > K s4 with the fourth, K = 16. Its rounding was of the size of its own power and
//     would dominate the smaller sums that remain, as after a spike or after
//     the last sample of an old level leaves;
//   - the sample that left was not finite (an infinity has left the window);
//   - window_size steps have passed since the last re-centering. This bounds
//     the rounding that the add-remove updates accumulate.
//
// With the origin within four standard deviations of the mean and no removed
// term larger than K times what remains, the sums carry relative errors of
// tens of times the machine precision (about a hundred for the fourth
// moment), as on stationary data. On
// stationary data the first two tests almost never fire, and the last fires
// once per window, so the cost per step is O(1) amortized; the step that
// re-centers costs one pass over the window (two after a far sample left,
// whose rounding makes the running mean inexact).
//
// "zero" start policy: the window starts full of zeros (count == capacity) and
// the origin starts at 0.
//
// Time sums (T = 1 or 2): with t = 0 for the oldest sample in the window and
// count - 1 for the newest, ts[q - 1] = sum t^q (x_i - origin) for q = 1..T.
// The time-index regressions (RollingPoly1, RollingPoly2, RollingTSF) take
// their fit from these and s[0]; the sums of t^q alone have closed forms. When
// the window slides, every t drops by one, so ts follows from the old ts, the
// new s[0] and the new sample; a re-centering recomputes them from the buffer
// with the rest.
//
// The running values are a small State, and the step is a static function of
// it. Both the batch loop and the per-event call run the step on a Cursor, a
// local copy of the State with the buffer pointer, and store the State back.
// Callers skip NaN inputs (the "ignore" NaN policy) before stepping.
//
// Each product is its own statement. Clang fuses a * b + c into one multiply-
// add only within an expression, and whether it does there can depend on where
// the code is inlined; with no product inside a sum, the step inlined into a
// batch loop rounds exactly like the same step called once per event.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>
#include "screamer/detail/start_policy.h"

namespace screamer {
namespace detail {

// The bound K of the far-out test above.
constexpr double kWindowRecenterTerm = 16.0;
// The drift test: s1^2 > D n s2, with D = 16/17, means |mean - origin| > 4 sd.
// Second moments then lose at most a factor 1 + 4^2 = 17 to cancellation, the
// fourth about 120.
constexpr double kWindowRecenterDrift = 16.0 / 17.0;

// The re-centering runs rarely and loops over the window; kept out of line it
// leaves the step small enough for the batch loop to keep its state in
// registers.
// The step itself is always inlined, so the batch loop's state stays in
// registers.
#if defined(__GNUC__) || defined(__clang__)
#define SCREAMER_WINDOW_NOINLINE __attribute__((noinline))
#define SCREAMER_WINDOW_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define SCREAMER_WINDOW_NOINLINE __declspec(noinline)
#define SCREAMER_WINDOW_INLINE __forceinline
#else
#define SCREAMER_WINDOW_NOINLINE
#define SCREAMER_WINDOW_INLINE inline
#endif

// The State member of an operator without time sums.
struct NoTimeSums {};

// Sums s[p - 1] = sum (x_i - origin)^p for p = 1..P, with P in 2..4, and the
// time sums ts[q - 1] = sum t^q (x_i - origin) for q = 1..T, with T in 0..2.
template <int P, int T = 0>
class ShiftedWindowSums {
    static_assert(P >= 2 && P <= 4, "ShiftedWindowSums supports powers 2 to 4");
    static_assert(T >= 0 && T <= 2, "ShiftedWindowSums supports time powers 0 to 2");

public:
    struct State {
        double s[P] = {};
        [[no_unique_address]] std::conditional_t<(T > 0), std::array<double, T>, NoTimeSums> ts{};
        std::size_t index = 0;   // next slot of the ring buffer
        std::size_t count = 0;   // samples in the window, zero padding included
        std::size_t age = 0;     // steps since the last re-centering
        double origin = 0.0;
    };

    ShiftedWindowSums(std::size_t capacity, StartPolicy policy)
        : capacity_(capacity), policy_(policy), buffer_(capacity, 0.0) {
        reset();
    }

    void reset() {
        std::fill(buffer_.begin(), buffer_.end(), 0.0);
        st_ = State{};
        st_.count = (policy_ == StartPolicy::Zero) ? capacity_ : 0;
    }

    // Adds x to the window held in (st, buf), dropping the oldest sample once
    // the window is full.
    static SCREAMER_WINDOW_INLINE void step(State& st, double* buf, std::size_t cap, double x) {
        constexpr double K = kWindowRecenterTerm;
        bool far = false;  // the removed sample calls for a re-centering
        if (st.count == cap) {
            const double d = x - st.origin;
            const double o = buf[st.index] - st.origin;
            st.s[0] += d - o;
            const double dd = d * d;
            const double oo = o * o;
            st.s[1] += dd - oo;
            if constexpr (P >= 3) {
                const double ddd = dd * d;
                const double ooo = oo * o;
                st.s[2] += ddd - ooo;
            }
            if constexpr (P == 2) {
                const double ks2 = K * st.s[1];
                far = oo > ks2;
            } else if constexpr (P == 3) {
                // The variance's dominance test. A removed term that does not
                // dominate the remaining spread leaves rounding of about
                // eps |o|^3 in s3, which the once-per-window recompute clears;
                // testing |o|^3 against the third-moment scale instead fired
                // on every outlier that left, even with others still in the
                // window.
                const double ks2 = K * st.s[1];
                far = oo > ks2;
            } else {
                const double dddd = dd * dd;
                const double oooo = oo * oo;
                st.s[3] += dddd - oooo;
                const double ks4 = K * st.s[3];
                far = oooo > ks4;
            }
            if constexpr (T >= 1) {
                // Every t drops by one and x enters at t = cap - 1:
                //   ts1' = ts1 + cap d - s0'
                //   ts2' = ts2 + (cap^2 - 2 cap) d - 2 ts1 + s0'
                // with s0' the new sum of d.
                const double nd = static_cast<double>(cap) * d;
                if constexpr (T >= 2) {
                    const double c2 = static_cast<double>(cap) * static_cast<double>(cap - 2);
                    const double c2d = c2 * d;
                    const double two_ts1 = 2.0 * st.ts[0];
                    st.ts[1] = st.ts[1] + c2d - two_ts1 + st.s[0];
                }
                st.ts[0] = st.ts[0] + nd - st.s[0];
            }
            far = far || !std::isfinite(o);
        } else {
            if (st.count == 0) st.origin = x;
            const double d = x - st.origin;
            if constexpr (T >= 1) {
                // x enters at t = count.
                const double t = static_cast<double>(st.count);
                const double td = t * d;
                st.ts[0] += td;
                if constexpr (T >= 2) {
                    const double ttd = t * td;
                    st.ts[1] += ttd;
                }
            }
            st.s[0] += d;
            const double dd = d * d;
            st.s[1] += dd;
            if constexpr (P >= 3) {
                const double ddd = dd * d;
                st.s[2] += ddd;
            }
            if constexpr (P >= 4) {
                const double dddd = dd * dd;
                st.s[3] += dddd;
            }
            ++st.count;
        }
        buf[st.index] = x;
        if (++st.index == cap) st.index = 0;
        // The mean has drifted more than four standard deviations from the
        // origin.
        const double s1s1 = st.s[0] * st.s[0];
        const double ns2 = kWindowRecenterDrift * static_cast<double>(st.count) * st.s[1];
        const bool drift = s1s1 > ns2;
        if (far || drift || ++st.age == cap) {
            st = recentered(st, buf, cap, far);
        }
    }

    // The state with the buffer it indexes, as one small value that a loop
    // keeps in registers; set_state(cursor.st) stores the state back.
    struct Cursor {
        State st;
        double* buf;
        std::size_t cap;
        SCREAMER_WINDOW_INLINE void push(double x) { step(st, buf, cap, x); }
    };
    Cursor cursor() { return Cursor{st_, buffer_.data(), capacity_}; }
    void set_state(const State& st) { st_ = st; }
    // Samples in the window, zero padding included.
    std::size_t count() const { return st_.count; }

private:
    // Moves the origin to the window mean and recomputes the sums exactly
    // from the buffer. When the sums may be inaccurate (`refresh`), the mean
    // is first recomputed from the buffer too; if the origin is not finite
    // (an infinity was in the window), a finite sample becomes the base. The
    // state goes by value, so a batch loop's local state does not escape and
    // stays in registers.
    SCREAMER_WINDOW_NOINLINE
    static State recentered(State st, const double* buf, std::size_t cap, bool refresh) {
        // The samples in the window: all slots once full (or zero padded),
        // else the first count slots.
        const std::size_t m = (st.count == cap) ? cap : st.count;
        const double n = static_cast<double>(st.count);
        double base = st.origin;
        double s1 = st.s[0];
        if (refresh || !std::isfinite(base)) {
            if (!std::isfinite(base)) {
                base = 0.0;
                for (std::size_t i = 0; i < m; ++i) {
                    if (std::isfinite(buf[i])) {
                        base = buf[i];
                        break;
                    }
                }
            }
            s1 = 0.0;
            for (std::size_t i = 0; i < m; ++i) s1 += buf[i] - base;
        }
        const double shift = s1 / n;
        st.origin = std::isfinite(shift) ? base + shift : base;
        double a1 = 0.0, a2 = 0.0, a3 = 0.0, a4 = 0.0;
        if constexpr (T == 0) {
            for (std::size_t i = 0; i < m; ++i) {
                const double d = buf[i] - st.origin;
                a1 += d;
                const double dd = d * d;
                a2 += dd;
                if constexpr (P >= 3) {
                    const double ddd = dd * d;
                    a3 += ddd;
                }
                if constexpr (P >= 4) {
                    const double dddd = dd * dd;
                    a4 += dddd;
                }
            }
        } else {
            // Oldest first: a full window starts at the next slot, a filling
            // one at slot 0.
            double b1 = 0.0, b2 = 0.0;
            std::size_t i = (st.count == cap) ? st.index : 0;
            for (std::size_t j = 0; j < m; ++j) {
                const double d = buf[i] - st.origin;
                if (++i == cap) i = 0;
                a1 += d;
                const double dd = d * d;
                a2 += dd;
                if constexpr (P >= 3) {
                    const double ddd = dd * d;
                    a3 += ddd;
                }
                if constexpr (P >= 4) {
                    const double dddd = dd * dd;
                    a4 += dddd;
                }
                const double t = static_cast<double>(j);
                const double td = t * d;
                b1 += td;
                if constexpr (T >= 2) {
                    const double ttd = t * td;
                    b2 += ttd;
                }
            }
            st.ts[0] = b1;
            if constexpr (T >= 2) st.ts[1] = b2;
        }
        // Zero padding (count == cap before cap samples arrived) is part of
        // the buffer, so the loop above already covers it.
        st.age = 0;
        st.s[0] = a1;
        st.s[1] = a2;
        if constexpr (P >= 3) st.s[2] = a3;
        if constexpr (P >= 4) st.s[3] = a4;
        return st;
    }

    std::size_t capacity_;
    StartPolicy policy_;
    std::vector<double> buffer_;
    State st_;
};

// The two-series version: with dx = x - origin_x and dy = y - origin_y, the
// sums sx, sy, sxx, syy and sxy. Every operator keeps all five: the tests that
// trigger a re-centering need the spread of each input.
class ShiftedWindowCoSums {
public:
    struct State {
        double sx = 0.0, sy = 0.0, sxy = 0.0, sxx = 0.0, syy = 0.0;
        std::size_t index = 0;
        std::size_t count = 0;
        std::size_t age = 0;
        double ox = 0.0, oy = 0.0;
    };

    ShiftedWindowCoSums(std::size_t capacity, StartPolicy policy)
        : capacity_(capacity), policy_(policy), bx_(capacity, 0.0), by_(capacity, 0.0) {
        reset();
    }

    void reset() {
        std::fill(bx_.begin(), bx_.end(), 0.0);
        std::fill(by_.begin(), by_.end(), 0.0);
        st_ = State{};
        st_.count = (policy_ == StartPolicy::Zero) ? capacity_ : 0;
    }

    static SCREAMER_WINDOW_INLINE void step(State& st, double* bx, double* by, std::size_t cap,
                                            double x, double y) {
        constexpr double K = kWindowRecenterTerm;
        bool far = false;
        if (st.count == cap) {
            const double dx = x - st.ox, dy = y - st.oy;
            const double qx = bx[st.index] - st.ox, qy = by[st.index] - st.oy;
            st.sx += dx - qx;
            st.sy += dy - qy;
            const double dxy = dx * dy, qxy = qx * qy;
            st.sxy += dxy - qxy;
            const double dxx = dx * dx, qxx = qx * qx;
            st.sxx += dxx - qxx;
            const double dyy = dy * dy, qyy = qy * qy;
            st.syy += dyy - qyy;
            const double ksxx = K * st.sxx, ksyy = K * st.syy;
            far = qxx > ksxx || qyy > ksyy || !std::isfinite(qxy);
        } else {
            if (st.count == 0) {
                st.ox = x;
                st.oy = y;
            }
            const double dx = x - st.ox, dy = y - st.oy;
            st.sx += dx;
            st.sy += dy;
            const double dxy = dx * dy;
            st.sxy += dxy;
            const double dxx = dx * dx;
            st.sxx += dxx;
            const double dyy = dy * dy;
            st.syy += dyy;
            ++st.count;
        }
        bx[st.index] = x;
        by[st.index] = y;
        if (++st.index == cap) st.index = 0;
        const double n = kWindowRecenterDrift * static_cast<double>(st.count);
        const double sxsx = st.sx * st.sx, sysy = st.sy * st.sy;
        const double nsxx = n * st.sxx, nsyy = n * st.syy;
        const bool drift = sxsx > nsxx || sysy > nsyy;
        if (far || drift || ++st.age == cap) {
            st = recentered(st, bx, by, cap, far);
        }
    }

    struct Cursor {  // as ShiftedWindowSums::Cursor
        State st;
        double* bx;
        double* by;
        std::size_t cap;
        SCREAMER_WINDOW_INLINE void push(double x, double y) { step(st, bx, by, cap, x, y); }
    };
    Cursor cursor() { return Cursor{st_, bx_.data(), by_.data(), capacity_}; }
    void set_state(const State& st) { st_ = st; }

private:
    // One input's new origin (see ShiftedWindowSums::recentered).
    static double new_origin(double origin, double s1, const double* b, std::size_t m,
                             double n, bool refresh) {
        double base = origin;
        if (refresh || !std::isfinite(base)) {
            if (!std::isfinite(base)) {
                base = 0.0;
                for (std::size_t i = 0; i < m; ++i) {
                    if (std::isfinite(b[i])) {
                        base = b[i];
                        break;
                    }
                }
            }
            s1 = 0.0;
            for (std::size_t i = 0; i < m; ++i) s1 += b[i] - base;
        }
        const double shift = s1 / n;
        return std::isfinite(shift) ? base + shift : base;
    }

    SCREAMER_WINDOW_NOINLINE
    static State recentered(State st, const double* bx, const double* by, std::size_t cap,
                            bool refresh) {
        const std::size_t m = (st.count == cap) ? cap : st.count;
        const double n = static_cast<double>(st.count);
        st.ox = new_origin(st.ox, st.sx, bx, m, n, refresh);
        st.oy = new_origin(st.oy, st.sy, by, m, n, refresh);
        double ax = 0.0, ay = 0.0, axx = 0.0, ayy = 0.0, axy = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            const double dx = bx[i] - st.ox, dy = by[i] - st.oy;
            ax += dx;
            ay += dy;
            const double dxy = dx * dy;
            axy += dxy;
            const double dxx = dx * dx;
            axx += dxx;
            const double dyy = dy * dy;
            ayy += dyy;
        }
        st.age = 0;
        st.sx = ax;
        st.sy = ay;
        st.sxy = axy;
        st.sxx = axx;
        st.syy = ayy;
        return st;
    }

    std::size_t capacity_;
    StartPolicy policy_;
    std::vector<double> bx_, by_;
    State st_;
};

}  // namespace detail
}  // namespace screamer

#endif  // SCREAMER_DETAIL_SHIFTED_WINDOW_SUMS_H
