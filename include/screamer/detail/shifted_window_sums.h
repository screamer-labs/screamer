#ifndef SCREAMER_DETAIL_SHIFTED_WINDOW_SUMS_H
#define SCREAMER_DETAIL_SHIFTED_WINDOW_SUMS_H

// Power sums of a sliding window, taken about an origin near the window mean.
//
// The central moments of a window follow from its power sums, but sums of raw
// x^p cancel when the mean is large next to the spread: on a price near 1000
// with unit moves the fourth central moment keeps about 4 of 16 digits. The
// classes here sum powers of d = x - origin instead, with the origin close to
// the mean, so every term is of the size of the spread and nothing cancels.
// The formulas that turn power sums into central moments are unchanged by the
// shift, and the window mean is origin + s1 / n.
//
// The origin is the first sample. Each time the window has been completely
// replaced (the ring index wraps), the origin moves to the window mean and the
// sums are recomputed from the buffer. That bounds the distance of the samples
// from the origin by the spread of two adjacent windows, and it discards the
// rounding the add-remove updates accumulated. The recompute is one pass over
// the window every window_size steps, so the cost per step stays O(1).
//
// "zero" start policy: the window starts full of zeros (count == capacity) and
// the origin starts at 0.
//
// The running values are a small State, and the step is a static function of
// it. Both the batch loop and the per-event call run the step on a Cursor, a
// local copy of the State with the buffer pointer, and store the State back. Callers skip NaN
// inputs (the "ignore" NaN policy) before stepping.
//
// Each product is its own statement. Clang fuses a * b + c into one multiply-
// add only within an expression, and whether it does there can depend on where
// the code is inlined; with no product inside a sum, the step inlined into a
// batch loop rounds exactly like the same step called once per event.

#include <algorithm>
#include <cstddef>
#include <vector>
#include "screamer/detail/start_policy.h"

namespace screamer {
namespace detail {

// Sums s[p - 1] = sum (x_i - origin)^p for p = 1..P, with P in 1..4.
template <int P>
class ShiftedWindowSums {
    static_assert(P >= 1 && P <= 4, "ShiftedWindowSums supports powers 1 to 4");

public:
    struct State {
        double s[P] = {};
        std::size_t index = 0;   // next slot of the ring buffer
        std::size_t count = 0;   // samples in the window, zero padding included
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
    static inline void step(State& st, double* buf, std::size_t cap, double x) {
        if (st.count == cap) {
            const double d = x - st.origin;
            const double o = buf[st.index] - st.origin;
            st.s[0] += d - o;
            if constexpr (P >= 2) {
                const double dd = d * d;
                const double oo = o * o;
                st.s[1] += dd - oo;
                if constexpr (P >= 3) {
                    const double ddd = dd * d;
                    const double ooo = oo * o;
                    st.s[2] += ddd - ooo;
                }
                if constexpr (P >= 4) {
                    const double dddd = dd * dd;
                    const double oooo = oo * oo;
                    st.s[3] += dddd - oooo;
                }
            }
        } else {
            if (st.count == 0) st.origin = x;
            const double d = x - st.origin;
            st.s[0] += d;
            if constexpr (P >= 2) {
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
            }
            ++st.count;
        }
        buf[st.index] = x;
        if (++st.index == cap) {
            st.index = 0;
            st = reanchored(st, buf, cap);
        }
    }

    // The state with the buffer it indexes, as one small value that a loop
    // keeps in registers; set_state(cursor.st) stores the state back.
    struct Cursor {
        State st;
        double* buf;
        std::size_t cap;
        inline void push(double x) { step(st, buf, cap, x); }
    };
    Cursor cursor() { return Cursor{st_, buffer_.data(), capacity_}; }
    void set_state(const State& st) { st_ = st; }

private:
    // For a full window: moves the origin to the window mean and recomputes
    // the sums exactly from the buffer. The state goes by value, so a batch
    // loop's local state does not escape and stays in registers.
    static State reanchored(State st, const double* buf, std::size_t cap) {
        st.origin += st.s[0] / static_cast<double>(st.count);
        double a1 = 0.0, a2 = 0.0, a3 = 0.0, a4 = 0.0;
        for (std::size_t i = 0; i < cap; ++i) {
            const double d = buf[i] - st.origin;
            a1 += d;
            if constexpr (P >= 2) {
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
        }
        st.s[0] = a1;
        if constexpr (P >= 2) st.s[1] = a2;
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
// sums sx, sy, sxy, and, when enabled, sxx and syy.
template <bool kXX, bool kYY>
class ShiftedWindowCoSums {
public:
    struct State {
        double sx = 0.0, sy = 0.0, sxy = 0.0, sxx = 0.0, syy = 0.0;
        std::size_t index = 0;
        std::size_t count = 0;
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

    static inline void step(State& st, double* bx, double* by, std::size_t cap,
                            double x, double y) {
        if (st.count == cap) {
            const double dx = x - st.ox, dy = y - st.oy;
            const double qx = bx[st.index] - st.ox, qy = by[st.index] - st.oy;
            st.sx += dx - qx;
            st.sy += dy - qy;
            const double dxy = dx * dy, qxy = qx * qy;
            st.sxy += dxy - qxy;
            if constexpr (kXX) {
                const double dxx = dx * dx, qxx = qx * qx;
                st.sxx += dxx - qxx;
            }
            if constexpr (kYY) {
                const double dyy = dy * dy, qyy = qy * qy;
                st.syy += dyy - qyy;
            }
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
            if constexpr (kXX) {
                const double dxx = dx * dx;
                st.sxx += dxx;
            }
            if constexpr (kYY) {
                const double dyy = dy * dy;
                st.syy += dyy;
            }
            ++st.count;
        }
        bx[st.index] = x;
        by[st.index] = y;
        if (++st.index == cap) {
            st.index = 0;
            st = reanchored(st, bx, by, cap);
        }
    }

    struct Cursor {  // as ShiftedWindowSums::Cursor
        State st;
        double* bx;
        double* by;
        std::size_t cap;
        inline void push(double x, double y) { step(st, bx, by, cap, x, y); }
    };
    Cursor cursor() { return Cursor{st_, bx_.data(), by_.data(), capacity_}; }
    void set_state(const State& st) { st_ = st; }

private:
    static State reanchored(State st, const double* bx, const double* by, std::size_t cap) {
        const double n = static_cast<double>(st.count);
        st.ox += st.sx / n;
        st.oy += st.sy / n;
        double ax = 0.0, ay = 0.0, axx = 0.0, ayy = 0.0, axy = 0.0;
        for (std::size_t i = 0; i < cap; ++i) {
            const double dx = bx[i] - st.ox, dy = by[i] - st.oy;
            ax += dx;
            ay += dy;
            const double dxy = dx * dy;
            axy += dxy;
            if constexpr (kXX) {
                const double dxx = dx * dx;
                axx += dxx;
            }
            if constexpr (kYY) {
                const double dyy = dy * dy;
                ayy += dyy;
            }
        }
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
