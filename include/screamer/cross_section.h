#ifndef SCREAMER_CROSS_SECTION_H
#define SCREAMER_CROSS_SECTION_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/common/float_info.h"
#include "screamer/common/row_op.h"

namespace screamer {

    // Cross-sectional operators: each transforms one row of K values (one per
    // asset) at a time, using only that row, so they are causal and stream
    // the same way they run on a (T, K) array. NaN entries are assets that are
    // not available on the row: they are left out of the cross-section and
    // their output is NaN.

    namespace detail {

        inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

        // Indices of the non-NaN entries of a row, reusing `idx`.
        inline void valid_indices(const double* in, std::size_t k, std::vector<std::size_t>& idx) {
            idx.clear();
            for (std::size_t i = 0; i < k; ++i)
                if (!isnan2(in[i])) idx.push_back(i);
        }

        // Sort `idx` by value; equal values keep column order.
        inline void sort_by_value(const double* in, std::vector<std::size_t>& idx) {
            std::stable_sort(idx.begin(), idx.end(),
                             [in](std::size_t a, std::size_t b) { return in[a] < in[b]; });
        }

        // Mean and sample standard deviation (ddof = 1) of the valid entries.
        inline void mean_std(const double* in, const std::vector<std::size_t>& idx,
                             double& mean, double& sd) {
            const double n = static_cast<double>(idx.size());
            double s = 0.0;
            for (std::size_t i : idx) s += in[i];
            mean = s / n;
            double ss = 0.0;
            for (std::size_t i : idx) { const double d = in[i] - mean; ss += d * d; }
            sd = idx.size() > 1 ? std::sqrt(ss / (n - 1.0)) : kNaN;
        }

        inline void fill_nan(double* out, std::size_t k) {
            std::fill(out, out + k, kNaN);
        }

    } // namespace detail

    // CrossRank: the rank of each asset within the row, ties given their
    // average rank, scaled to [0, 1] ("unit": lowest 0, highest 1) or to
    // [-1, 1] ("signed"). A row with one valid asset gives the midpoint.
    class CrossRank : public serial::Saveable<CrossRank, RowOp> {
    public:
        static constexpr const char* kTypeName = "CrossRank";
        static constexpr int kStateVersion = 1;

        template <class Ar>
        void settings(Ar& ar) {
            ar("scale", std::string(signed_ ? "signed" : "unit"));
        }

        // A row's result depends on that row alone; the state is the width.
        template <class Ar>
        void state(Ar& ar) { width_state(ar); }

        explicit CrossRank(const std::string& scale = "signed") : signed_(parse(scale)) {}

    protected:
        void map_row(const double* in, double* out, std::size_t k) override {
            detail::fill_nan(out, k);
            detail::valid_indices(in, k, idx_);
            const std::size_t n = idx_.size();
            if (n == 0) return;
            detail::sort_by_value(in, idx_);
            for (std::size_t a = 0; a < n;) {
                std::size_t b = a + 1;
                while (b < n && in[idx_[b]] == in[idx_[a]]) ++b;
                const double avg = 0.5 * static_cast<double>(a + b - 1);   // 0-based
                const double u = n > 1 ? avg / static_cast<double>(n - 1) : 0.5;
                for (std::size_t j = a; j < b; ++j)
                    out[idx_[j]] = signed_ ? 2.0 * u - 1.0 : u;
                a = b;
            }
        }

    private:
        static bool parse(const std::string& s) {
            if (s == "signed") return true;
            if (s == "unit") return false;
            throw std::invalid_argument("scale must be \"signed\" or \"unit\".");
        }
        bool signed_;
        std::vector<std::size_t> idx_;
    };

    // CrossZscore: (x - mean) / std across the row, with the sample standard
    // deviation (ddof = 1). A row whose valid values are all equal gives 0; a
    // row with one valid asset gives NaN.
    class CrossZscore : public serial::Saveable<CrossZscore, RowOp> {
    public:
        static constexpr const char* kTypeName = "CrossZscore";
        static constexpr int kStateVersion = 1;

        template <class Ar>
        void settings(Ar&) {}

        // A row's result depends on that row alone; the state is the width.
        template <class Ar>
        void state(Ar& ar) { width_state(ar); }

    protected:
        void map_row(const double* in, double* out, std::size_t k) override {
            detail::fill_nan(out, k);
            detail::valid_indices(in, k, idx_);
            if (idx_.size() < 2) return;
            double mean, sd;
            detail::mean_std(in, idx_, mean, sd);
            for (std::size_t i : idx_) out[i] = sd > 0.0 ? (in[i] - mean) / sd : 0.0;
        }

    private:
        std::vector<std::size_t> idx_;
    };

    // CrossStd: the standard deviation of the row's valid values, one value
    // per row (a row reduction): sqrt(sum (x - mean)^2 / (n - ddof)) over the
    // n non-NaN assets, as pandas DataFrame.std(axis=1, ddof=ddof) and numpy
    // nanstd(axis=1, ddof=ddof). A row with n <= ddof gives NaN.
    class CrossStd : public serial::Saveable<CrossStd, RowOp> {
    public:
        static constexpr const char* kTypeName = "CrossStd";
        static constexpr int kStateVersion = 1;

        template <class Ar>
        void settings(Ar& ar) {
            ar("ddof", ddof_);
        }

        // A row's result depends on that row alone; the state is the width.
        template <class Ar>
        void state(Ar& ar) { width_state(ar); }

        explicit CrossStd(int ddof = 1) : serial::Saveable<CrossStd, RowOp>(1, 1, /*per_asset_out=*/false), ddof_(ddof) {
            if (ddof_ < 0) throw std::invalid_argument("ddof must be non-negative.");
        }

    protected:
        void map_row(const double* in, double* out, std::size_t k) override {
            double s = 0.0;
            std::size_t n = 0;
            for (std::size_t i = 0; i < k; ++i)
                if (!isnan2(in[i])) { s += in[i]; ++n; }
            if (n <= static_cast<std::size_t>(ddof_)) { out[0] = detail::kNaN; return; }
            const double mean = s / static_cast<double>(n);
            double ss = 0.0;
            for (std::size_t i = 0; i < k; ++i)
                if (!isnan2(in[i])) { const double d = in[i] - mean; ss += d * d; }
            out[0] = std::sqrt(ss / static_cast<double>(n - static_cast<std::size_t>(ddof_)));
        }

    private:
        int ddof_;
    };

    // CrossDemean: x - mean across the row.
    class CrossDemean : public serial::Saveable<CrossDemean, RowOp> {
    public:
        static constexpr const char* kTypeName = "CrossDemean";
        static constexpr int kStateVersion = 1;

        template <class Ar>
        void settings(Ar&) {}

        // A row's result depends on that row alone; the state is the width.
        template <class Ar>
        void state(Ar& ar) { width_state(ar); }

    protected:
        void map_row(const double* in, double* out, std::size_t k) override {
            double s = 0.0;
            std::size_t n = 0;
            for (std::size_t i = 0; i < k; ++i)
                if (!isnan2(in[i])) { s += in[i]; ++n; }
            const double mean = n ? s / static_cast<double>(n) : detail::kNaN;
            for (std::size_t i = 0; i < k; ++i)
                out[i] = isnan2(in[i]) ? detail::kNaN : in[i] - mean;
        }
    };

    // CrossWinsorize: clip each value to the row's `lower` and `upper`
    // quantiles (linear interpolation between order statistics, as numpy's
    // default quantile).
    class CrossWinsorize : public serial::Saveable<CrossWinsorize, RowOp> {
    public:
        static constexpr const char* kTypeName = "CrossWinsorize";
        static constexpr int kStateVersion = 1;

        template <class Ar>
        void settings(Ar& ar) {
            ar("lower", lo_);
            ar("upper", hi_);
        }

        // A row's result depends on that row alone; the state is the width.
        template <class Ar>
        void state(Ar& ar) { width_state(ar); }

        CrossWinsorize(double lower = 0.05, double upper = 0.95) : lo_(lower), hi_(upper) {
            if (!(lo_ >= 0.0 && lo_ <= hi_ && hi_ <= 1.0))
                throw std::invalid_argument("need 0 <= lower <= upper <= 1.");
        }

    protected:
        void map_row(const double* in, double* out, std::size_t k) override {
            detail::fill_nan(out, k);
            detail::valid_indices(in, k, idx_);
            if (idx_.empty()) return;
            sorted_.clear();
            for (std::size_t i : idx_) sorted_.push_back(in[i]);
            std::sort(sorted_.begin(), sorted_.end());
            const double qlo = quantile(lo_), qhi = quantile(hi_);
            for (std::size_t i : idx_) out[i] = std::clamp(in[i], qlo, qhi);
        }

    private:
        double quantile(double p) const {
            const double pos = p * static_cast<double>(sorted_.size() - 1);
            const std::size_t i = static_cast<std::size_t>(std::floor(pos));
            if (i + 1 >= sorted_.size()) return sorted_.back();
            const double f = pos - static_cast<double>(i);
            return sorted_[i] + f * (sorted_[i + 1] - sorted_[i]);
        }
        double lo_, hi_;
        std::vector<std::size_t> idx_;
        std::vector<double> sorted_;
    };

    // CrossNormalize: scale a row of scores into portfolio weights.
    //   net unset (default): w = x * gross / sum|x|, keeping each sign.
    //   net given: demean the row, then scale the long side to sum to
    //     (gross + net) / 2 and the short side to (gross - net) / 2, so the
    //     weights have gross exposure `gross` and net exposure `net`
    //     (net = 0 is dollar-neutral).
    // A row with no exposure to scale (all zero, or all equal with a net
    // target) gives zero weights.
    class CrossNormalize : public serial::Saveable<CrossNormalize, RowOp> {
    public:
        static constexpr const char* kTypeName = "CrossNormalize";
        static constexpr int kStateVersion = 1;

        template <class Ar>
        void settings(Ar& ar) {
            ar("gross", gross_);
            ar("net", isnan2(net_) ? std::optional<double>() : std::optional<double>(net_));
        }

        // A row's result depends on that row alone; the state is the width.
        template <class Ar>
        void state(Ar& ar) { width_state(ar); }

        CrossNormalize(double gross = 1.0, std::optional<double> net = std::nullopt)
            : gross_(gross), net_(net ? *net : detail::kNaN) {
            if (!(std::isfinite(gross_) && gross_ >= 0.0))
                throw std::invalid_argument("gross must be finite and non-negative.");
            if (!isnan2(net_) && !(std::isfinite(net_) && std::abs(net_) <= gross_))
                throw std::invalid_argument("net must be in [-gross, gross].");
        }

    protected:
        void map_row(const double* in, double* out, std::size_t k) override {
            double s = 0.0, n = 0.0;
            for (std::size_t i = 0; i < k; ++i)
                if (!isnan2(in[i])) { s += in[i]; n += 1.0; }
            if (isnan2(net_)) {
                double abs_sum = 0.0;
                for (std::size_t i = 0; i < k; ++i)
                    if (!isnan2(in[i])) abs_sum += std::abs(in[i]);
                const double f = abs_sum > 0.0 ? gross_ / abs_sum : 0.0;
                for (std::size_t i = 0; i < k; ++i)
                    out[i] = isnan2(in[i]) ? detail::kNaN : in[i] * f;
                return;
            }
            const double mean = n > 0.0 ? s / n : 0.0;
            double longs = 0.0, shorts = 0.0;
            for (std::size_t i = 0; i < k; ++i) {
                if (isnan2(in[i])) continue;
                const double d = in[i] - mean;
                if (d > 0.0) longs += d; else shorts -= d;
            }
            const bool both = longs > 0.0 && shorts > 0.0;
            const double fl = both ? 0.5 * (gross_ + net_) / longs : 0.0;
            const double fs = both ? 0.5 * (gross_ - net_) / shorts : 0.0;
            for (std::size_t i = 0; i < k; ++i) {
                if (isnan2(in[i])) { out[i] = detail::kNaN; continue; }
                const double d = in[i] - mean;
                out[i] = d > 0.0 ? d * fl : d * fs;
            }
        }

    private:
        double gross_, net_;
    };

    // CrossTopBottom: +1 for the `count` highest values of the row, -1 for
    // the `count` lowest, 0 for the rest. Ties are broken by column order
    // (the earlier column ranks lower). With fewer than 2 * count valid
    // assets, count is reduced to half of them so no asset is on both sides.
    // A row whose valid values are all equal gives 0 for every asset.
    class CrossTopBottom : public serial::Saveable<CrossTopBottom, RowOp> {
    public:
        static constexpr const char* kTypeName = "CrossTopBottom";
        static constexpr int kStateVersion = 1;

        template <class Ar>
        void settings(Ar& ar) {
            ar("count", count_);
        }

        // A row's result depends on that row alone; the state is the width.
        template <class Ar>
        void state(Ar& ar) { width_state(ar); }

        explicit CrossTopBottom(int count = 1) : count_(count) {
            if (count_ < 1) throw std::invalid_argument("count must be at least 1.");
        }

    protected:
        void map_row(const double* in, double* out, std::size_t k) override {
            detail::fill_nan(out, k);
            detail::valid_indices(in, k, idx_);
            const std::size_t n = idx_.size();
            for (std::size_t i : idx_) out[i] = 0.0;
            const std::size_t c = std::min<std::size_t>(static_cast<std::size_t>(count_), n / 2);
            if (c == 0) return;
            detail::sort_by_value(in, idx_);
            if (in[idx_.front()] == in[idx_.back()]) return;   // all equal: no ranking
            for (std::size_t j = 0; j < c; ++j) {
                out[idx_[j]] = -1.0;
                out[idx_[n - 1 - j]] = 1.0;
            }
        }

    private:
        int count_;
        std::vector<std::size_t> idx_;
    };

} // namespace screamer

#endif // SCREAMER_CROSS_SECTION_H
