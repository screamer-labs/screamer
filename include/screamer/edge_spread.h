#ifndef SCREAMER_EDGE_SPREAD_H
#define SCREAMER_EDGE_SPREAD_H

// EdgeSpread: the EDGE bid-ask spread estimator of Ardia, Guidotti and
// Kroencke (2024), 4 -> 1 over (open, high, low, close), on a trailing
// window of window_size bars. The output at bar t is the authors' estimator
// (github.com/eguidotti/bidask, pseudocode/README.md) applied to bars
// t - window_size + 1 .. t, with the pseudocode's handling of missing values.
//
// With o, h, l, c the log prices, m = (h + l) / 2, and the suffix 1 for the
// previous bar, each pair of consecutive bars gives the returns
//
//     r1 = m - o,  r2 = o - m1,  r3 = m - c1,  r4 = c1 - m1,  r5 = o - c1
//
// and the indicators tau = (h != l or l != c1), po1 = tau and o != h,
// po2 = tau and o != l, pc1 = tau and c1 != h1, pc2 = tau and c1 != l1.
// With means over the pairs of the window, pt = mean(tau),
// po = mean(po1) + mean(po2), pc = mean(pc1) + mean(pc2), and
//
//     d_k = r_k - mean(r_k) / pt * tau                        (k = 1, 3, 5)
//     x1  = -4/po d1 r2 - 4/pc d3 r4
//     x2  = -4/po d1 r5 - 4/pc d5 r4
//
// the squared spread is the variance-weighted mean of e1 = mean(x1) and
// e2 = mean(x2), s2 = (v2 e1 + v1 e2) / (v1 + v2) with v_k = var(x_k), or
// (e1 + e2) / 2 when v1 + v2 is not positive. The output is sqrt(|s2|),
// negative when s2 < 0 and sign is true. It is NaN when fewer than two
// pairs have tau = 1 or when po or pc is 0.
//
// Missing values (nan_policy nan-aware): a price that is NaN, or not positive
// and finite, is missing. Each indicator and return is left out of its mean
// where a price it uses is missing, as the pseudocode's nanmean does, and x1
// and x2 where any of o, h, l, h1, l1, c1 is. A bar still takes its place in
// the window whatever is missing, so the window is window_size bars.
//
// x1 and x2 depend on window means, so the window keeps the sums of the
// products their means and second moments expand into (33 sums, 7 counts).
// A step adds the newest pair and subtracts the pair that leaves, computed
// again from the ring of the last window_size bars' log prices; the sums are
// recomputed from the ring once every window_size steps, so rounding does not
// build up. O(1) amortized per bar.

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/common/float_info.h"
#include "screamer/common/functor_base.h"

namespace screamer {

class EdgeSpread : public FunctorBase<EdgeSpread, 4, 1> {
public:
    EdgeSpread(int window_size, bool sign = false, const std::string& start_policy = "strict")
        : sign_(sign),
          expanding_(parse_policy(start_policy)),
          cap_(checked_window(window_size)),
          ring_(4 * cap_, 0.0)
    {
        reset();
    }

    void reset() override {
        sum_.fill(0.0);
        cnt_.fill(0);
        pos_ = 0;
        bars_ = 0;
        since_ = 0;
    }

    double call(const InputArray& inputs) override {
        double bar[4];
        for (int k = 0; k < 4; ++k) {
            const double v = std::log(inputs[k]);
            // log(x) is finite exactly for a positive finite x.
            bar[k] = std::isfinite(v) ? v : std::numeric_limits<double>::quiet_NaN();
        }
        if (bars_ == cap_) {
            const std::size_t next = (pos_ + 1 == cap_) ? 0 : pos_ + 1;
            accumulate(&ring_[4 * pos_], &ring_[4 * next], -1);
        }
        if (bars_ > 0) {
            const std::size_t prev = (pos_ == 0) ? cap_ - 1 : pos_ - 1;
            accumulate(&ring_[4 * prev], bar, +1);
        }
        double* slot = &ring_[4 * pos_];
        for (int k = 0; k < 4; ++k) slot[k] = bar[k];
        if (++pos_ == cap_) pos_ = 0;
        if (bars_ < cap_) ++bars_;
        if (++since_ >= cap_) resum();
        if (bars_ < cap_ && !expanding_) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return estimate();
    }

private:
    // Sums over the pairs of the window. The first group has its own count
    // each; the rest are over the pairs with all of o, h, l, h1, l1, c1
    // (count kNV). u = r1 r2, ut = tau r2, w = r3 r4, wt = tau r4,
    // p = r1 r5, pt = tau r5, z = r4 r5.
    enum Sum {
        TAU, PO, PC1, PC2, R1, R3, R5,
        U, UT, W, WT, P, PT, Z,
        UU, UUT, UTUT, WW, WWT, WTWT, UW, UWT, UTW, UTWT,
        PP, PPT, PTPT, ZZ, ZWT, PZ, PWT, PTZ, PTWT,
        kSums
    };
    enum Count { NTAU, NPO, NPC1, NPC2, NR1, NR5, NV, kCounts };

    static bool parse_policy(const std::string& policy) {
        if (policy == "strict") return false;
        if (policy == "expanding") return true;
        throw std::invalid_argument("EdgeSpread: start_policy must be \"strict\" or \"expanding\", got \"" +
                                    policy + "\"");
    }

    static std::size_t checked_window(int window_size) {
        if (window_size < 3) {
            throw std::invalid_argument("Window size must be 3 or more.");
        }
        return static_cast<std::size_t>(window_size);
    }

    // Add (dir = +1) or subtract (dir = -1) the pair of bars a (earlier) and
    // b, each the log prices {o, h, l, c}.
    void accumulate(const double* a, const double* b, int dir) {
        const double h1 = a[1], l1 = a[2], c1 = a[3];
        const double o = b[0], h = b[1], l = b[2];
        const bool ho = !isnan2(o), hh = !isnan2(h), hl = !isnan2(l);
        const bool hh1 = !isnan2(h1), hl1 = !isnan2(l1), hc1 = !isnan2(c1);
        const double s = static_cast<double>(dir);
        const double m = 0.5 * (h + l);
        const bool vt = hh && hl && hc1;
        double tau = 0.0;
        if (vt) {
            tau = (h != l || l != c1) ? 1.0 : 0.0;
            cnt_[NTAU] += dir;
            sum_[TAU] += s * tau;
            sum_[R3] += s * (m - c1);
            if (ho) {
                cnt_[NPO] += dir;
                sum_[PO] += s * (tau * ((o != h) ? 1.0 : 0.0) + tau * ((o != l) ? 1.0 : 0.0));
            }
            if (hh1) {
                cnt_[NPC1] += dir;
                sum_[PC1] += s * (tau * ((c1 != h1) ? 1.0 : 0.0));
            }
            if (hl1) {
                cnt_[NPC2] += dir;
                sum_[PC2] += s * (tau * ((c1 != l1) ? 1.0 : 0.0));
            }
        }
        if (ho && hh && hl) {
            cnt_[NR1] += dir;
            sum_[R1] += s * (m - o);
        }
        if (ho && hc1) {
            cnt_[NR5] += dir;
            sum_[R5] += s * (o - c1);
        }
        if (vt && ho && hh1 && hl1) {
            const double m1 = 0.5 * (h1 + l1);
            const double r1 = m - o, r2 = o - m1, r3 = m - c1, r4 = c1 - m1, r5 = o - c1;
            const double u = r1 * r2, ut = tau * r2, w = r3 * r4, wt = tau * r4;
            const double p = r1 * r5, pt = tau * r5, z = r4 * r5;
            cnt_[NV] += dir;
            sum_[U] += s * u;
            sum_[UT] += s * ut;
            sum_[W] += s * w;
            sum_[WT] += s * wt;
            sum_[P] += s * p;
            sum_[PT] += s * pt;
            sum_[Z] += s * z;
            sum_[UU] += s * (u * u);
            sum_[UUT] += s * (u * ut);
            sum_[UTUT] += s * (ut * ut);
            sum_[WW] += s * (w * w);
            sum_[WWT] += s * (w * wt);
            sum_[WTWT] += s * (wt * wt);
            sum_[UW] += s * (u * w);
            sum_[UWT] += s * (u * wt);
            sum_[UTW] += s * (ut * w);
            sum_[UTWT] += s * (ut * wt);
            sum_[PP] += s * (p * p);
            sum_[PPT] += s * (p * pt);
            sum_[PTPT] += s * (pt * pt);
            sum_[ZZ] += s * (z * z);
            sum_[ZWT] += s * (z * wt);
            sum_[PZ] += s * (p * z);
            sum_[PWT] += s * (p * wt);
            sum_[PTZ] += s * (pt * z);
            sum_[PTWT] += s * (pt * wt);
        }
    }

    // The exact sums from the ring, oldest pair first. Out of line: it runs
    // once per window_size steps.
    SCREAMER_NOINLINE void resum() {
        sum_.fill(0.0);
        cnt_.fill(0);
        std::size_t i = (pos_ + cap_ - bars_) % cap_;
        for (std::size_t k = 1; k < bars_; ++k) {
            const std::size_t j = (i + 1 == cap_) ? 0 : i + 1;
            accumulate(&ring_[4 * i], &ring_[4 * j], +1);
            i = j;
        }
        since_ = 0;
    }

    // mean(x^2) - mean(x)^2 from the expanded sums. Where it is within
    // rounding of 0 (one pair, or equal x on every pair) it is 0, as the
    // pseudocode computes it from the x values themselves: rounding left in
    // v1 and v2 would otherwise set the weights of e1 and e2.
    double variance(double x2, double e) const {
        constexpr double kRel = 1e-10;
        const double v = x2 - e * e;
        return (cnt_[NV] > 1 && std::fabs(v) > kRel * x2) ? v : 0.0;
    }

    double estimate() const {
        constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
        for (long n : cnt_) {
            if (n <= 0) return kNaN;
        }
        // Fewer than two pairs with tau = 1. TAU is a sum of 0s and 1s, exact.
        if (sum_[TAU] < 2.0) return kNaN;
        const double pt = sum_[TAU] / static_cast<double>(cnt_[NTAU]);
        const double po = sum_[PO] / static_cast<double>(cnt_[NPO]);
        const double pc = sum_[PC1] / static_cast<double>(cnt_[NPC1]) +
                          sum_[PC2] / static_cast<double>(cnt_[NPC2]);
        if (po == 0.0 || pc == 0.0) return kNaN;

        const double a1 = -4.0 / po;
        const double a2 = -4.0 / pc;
        const double ma = sum_[R1] / static_cast<double>(cnt_[NR1]) / pt;
        const double mb = sum_[R3] / static_cast<double>(cnt_[NTAU]) / pt;
        const double mc = sum_[R5] / static_cast<double>(cnt_[NR5]) / pt;
        const double nv = static_cast<double>(cnt_[NV]);

        // x1 = k0 u + k1 ut + k2 w + k3 wt and x2 = q0 p + q1 pt + q2 z + q3 wt.
        const double k0 = a1, k1 = -a1 * ma, k2 = a2, k3 = -a2 * mb;
        const double q0 = a1, q1 = -a1 * ma, q2 = a2, q3 = -a2 * mc;
        const double e1 = (k0 * sum_[U] + k1 * sum_[UT] + k2 * sum_[W] + k3 * sum_[WT]) / nv;
        const double e2 = (q0 * sum_[P] + q1 * sum_[PT] + q2 * sum_[Z] + q3 * sum_[WT]) / nv;
        const double x11 = (k0 * k0 * sum_[UU] + k1 * k1 * sum_[UTUT] + k2 * k2 * sum_[WW] +
                            k3 * k3 * sum_[WTWT] +
                            2.0 * (k0 * k1 * sum_[UUT] + k0 * k2 * sum_[UW] + k0 * k3 * sum_[UWT] +
                                   k1 * k2 * sum_[UTW] + k1 * k3 * sum_[UTWT] + k2 * k3 * sum_[WWT])) / nv;
        const double x22 = (q0 * q0 * sum_[PP] + q1 * q1 * sum_[PTPT] + q2 * q2 * sum_[ZZ] +
                            q3 * q3 * sum_[WTWT] +
                            2.0 * (q0 * q1 * sum_[PPT] + q0 * q2 * sum_[PZ] + q0 * q3 * sum_[PWT] +
                                   q1 * q2 * sum_[PTZ] + q1 * q3 * sum_[PTWT] + q2 * q3 * sum_[ZWT])) / nv;
        const double v1 = variance(x11, e1);
        const double v2 = variance(x22, e2);
        const double vt = v1 + v2;
        const double s2 = (vt > 0.0) ? (v2 * e1 + v1 * e2) / vt : 0.5 * (e1 + e2);
        const double s = std::sqrt(std::fabs(s2));
        return (sign_ && s2 < 0.0) ? -s : s;
    }

    const bool sign_;
    const bool expanding_;
    const std::size_t cap_;
    std::vector<double> ring_;            // log {o, h, l, c} of the last cap_ bars
    std::array<double, kSums> sum_{};
    std::array<long, kCounts> cnt_{};
    std::size_t pos_ = 0;                 // next slot of the ring
    std::size_t bars_ = 0;                // bars in the ring
    std::size_t since_ = 0;               // steps since the last recompute
};

}  // namespace screamer

#endif  // SCREAMER_EDGE_SPREAD_H
