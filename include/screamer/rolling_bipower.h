#ifndef SCREAMER_ROLLING_BIPOWER_H
#define SCREAMER_ROLLING_BIPOWER_H

// RollingBipowerVar and RollingJumpShare (Barndorff-Nielsen and Shephard
// 2004). On a window of n returns (n = window_size once full),
//     BV = (pi / 2) * sum_{i=2..n} |r_i| |r_{i-1}| * n / (n - 1)
//     RV = sum_{i=1..n} r_i^2
// BV estimates the variance of the continuous part of the returns and RV the
// whole variance, jumps included; the factor n / (n - 1) scales the n - 1
// pairs to the n terms of RV. RollingJumpShare = max(0, 1 - BV / RV), the
// share of RV that comes from jumps, and 0 when RV is 0.
//
// start_policy: "strict" gives NaN until the window holds window_size
// returns, "expanding" uses the returns seen so far (two at least), "zero"
// starts from a window of zeros. nan_policy: ignore (a NaN is not stored;
// the pair after it joins the returns on either side). O(1) per step.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/rolling_sum.h"
#include "screamer/detail/start_policy.h"

namespace screamer {
namespace detail {

class BipowerWindow {
public:
    BipowerWindow(int window_size, const std::string& start_policy)
        : window_size_(window_size),
          policy_(parse_start_policy(start_policy)),
          pairs_(window_size >= 2 ? static_cast<std::size_t>(window_size - 1) : 1, start_policy),
          squares_(window_size >= 2 ? static_cast<std::size_t>(window_size) : 2, start_policy),
          square_nz_(window_size >= 2 ? static_cast<std::size_t>(window_size) : 2, 0),
          pair_nz_(window_size >= 2 ? static_cast<std::size_t>(window_size - 1) : 1, 0)
    {
        if (window_size < 2) throw std::invalid_argument("Window size must be 2 or more.");
        reset();
    }

    void reset() {
        pairs_.reset();
        squares_.reset();
        const bool zero = policy_ == StartPolicy::Zero;
        n_ = zero ? static_cast<std::size_t>(window_size_) : 0;
        has_prev_ = zero;               // the zero window's last return is 0
        prev_abs_ = 0.0;
        std::fill(square_nz_.begin(), square_nz_.end(), static_cast<unsigned char>(0));
        std::fill(pair_nz_.begin(), pair_nz_.end(), static_cast<unsigned char>(0));
        square_pos_ = pair_pos_ = 0;
        square_count_ = pair_count_ = 0;
    }

    // Adds a finite return; bv and rv are NaN where the policy gives none.
    void push(double r, double& bv, double& rv) {
        constexpr double kHalfPi = 1.5707963267948966;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (n_ < static_cast<std::size_t>(window_size_)) ++n_;
        rv = squares_.append(r * r);
        track(square_nz_, square_pos_, square_count_, r * r != 0.0);
        // The sliding sums keep a rounding residue after nonzero terms leave;
        // an exact count of nonzero terms makes an all-zero window exactly 0.
        if (!std::isnan(rv)) rv = square_count_ == 0 ? 0.0 : rv;
        const double a = std::fabs(r);
        double pair_sum = nan;
        if (has_prev_) {
            const double prod = prev_abs_ * a;
            // inf * 0 is NaN: RollingSum skips it, and so does the flag ring,
            // so the two hold the same terms.
            if (!std::isnan(prod)) {
                pair_sum = pairs_.append(prod);
                track(pair_nz_, pair_pos_, pair_count_, prod != 0.0);
                if (!std::isnan(pair_sum)) pair_sum = pair_count_ == 0 ? 0.0 : pair_sum;
            }
        }
        prev_abs_ = a;
        has_prev_ = true;
        if (n_ < 2) {
            bv = nan;
            return;
        }
        const double n = static_cast<double>(n_);
        const double scale = kHalfPi * n / (n - 1.0);
        bv = pair_sum * scale;
        if (bv < 0.0) bv = 0.0;     // a residue of the sliding sum
    }

    int window_size() const { return window_size_; }
    StartPolicy start_policy() const { return policy_; }

    // The two sliding sums, the rings of nonzero flags with their slots and
    // counts, the samples seen (up to window_size) and the last |r|.
    template <class Ar>
    void state(Ar& ar) {
        ar.object("pairs", pairs_);
        ar.object("squares", squares_);
        ar.fixed("square_nonzero", square_nz_, square_nz_.size());
        ar.fixed("pair_nonzero", pair_nz_, pair_nz_.size());
        ar("square_pos", square_pos_);
        ar("pair_pos", pair_pos_);
        ar("square_count", square_count_);
        ar("pair_count", pair_count_);
        ar("n", n_);
        ar("has_prev", has_prev_);
        ar("prev_abs", prev_abs_);
        ar.check(square_pos_ < square_nz_.size(), "square_pos must be below window_size");
        ar.check(pair_pos_ < pair_nz_.size(), "pair_pos must be below window_size - 1");
        ar.check(square_count_ == count_set(square_nz_),
                 "square_count must equal the nonzero flags of square_nonzero");
        ar.check(pair_count_ == count_set(pair_nz_),
                 "pair_count must equal the nonzero flags of pair_nonzero");
        ar.check(n_ <= static_cast<std::size_t>(window_size_), "n must be at most window_size");
    }

private:
    // The flags set in a ring; a flag other than 0 or 1 does not count.
    static std::size_t count_set(const std::vector<unsigned char>& ring) {
        std::size_t n = 0;
        for (unsigned char f : ring) {
            if (f > 1) return static_cast<std::size_t>(-1);
            n += f;
        }
        return n;
    }

    // Ring of "term was nonzero" flags with a running count.
    static void track(std::vector<unsigned char>& ring, std::size_t& pos,
                      std::size_t& count, bool nonzero) {
        count -= ring[pos];
        ring[pos] = nonzero ? 1 : 0;
        count += ring[pos];
        if (++pos == ring.size()) pos = 0;
    }

    const int window_size_;
    const StartPolicy policy_;
    RollingSum pairs_, squares_;
    std::vector<unsigned char> square_nz_, pair_nz_;
    std::size_t square_pos_ = 0, pair_pos_ = 0, square_count_ = 0, pair_count_ = 0;
    std::size_t n_ = 0;
    bool has_prev_ = false;
    double prev_abs_ = 0.0;
};

}  // namespace detail

class RollingBipowerVar : public serial::Saveable<RollingBipowerVar, ScreamerBase> {
public:
    static constexpr const char* kTypeName = "RollingBipowerVar";
    static constexpr int kStateVersion = 1;

    template <class Ar>
    void settings(Ar& ar) {
        ar("window_size", window_.window_size());
        ar("start_policy", std::string(detail::start_policy_name(window_.start_policy())));
    }

    template <class Ar>
    void state(Ar& ar) {
        window_.state(ar);
    }

    RollingBipowerVar(int window_size = 20, const std::string& start_policy = "strict")
        : window_(window_size, start_policy) {}

    void reset() override { window_.reset(); }

    double process_scalar(double x) override {
        if (isnan2(x)) return std::numeric_limits<double>::quiet_NaN();
        double bv = 0.0, rv = 0.0;
        window_.push(x, bv, rv);
        return bv;
    }

private:
    detail::BipowerWindow window_;
};

class RollingJumpShare : public serial::Saveable<RollingJumpShare, ScreamerBase> {
public:
    static constexpr const char* kTypeName = "RollingJumpShare";
    static constexpr int kStateVersion = 1;

    template <class Ar>
    void settings(Ar& ar) {
        ar("window_size", window_.window_size());
        ar("start_policy", std::string(detail::start_policy_name(window_.start_policy())));
    }

    template <class Ar>
    void state(Ar& ar) {
        window_.state(ar);
    }

    RollingJumpShare(int window_size = 20, const std::string& start_policy = "strict")
        : window_(window_size, start_policy) {}

    void reset() override { window_.reset(); }

    double process_scalar(double x) override {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (isnan2(x)) return nan;
        double bv = 0.0, rv = 0.0;
        window_.push(x, bv, rv);
        if (isnan2(bv) || isnan2(rv)) return nan;
        if (rv == 0.0) return 0.0;
        const double share = 1.0 - bv / rv;
        if (share < 0.0) return 0.0;
        return share > 1.0 ? 1.0 : share;
    }

private:
    detail::BipowerWindow window_;
};

}  // namespace screamer

#endif  // SCREAMER_ROLLING_BIPOWER_H
