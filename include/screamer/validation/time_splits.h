#ifndef SCREAMER_VALIDATION_TIME_SPLITS_H
#define SCREAMER_VALIDATION_TIME_SPLITS_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace screamer { namespace validation {

    // time_splits: scikit-learn TimeSeriesSplit folds with label purging.
    //
    // The n_splits test blocks of test_size rows end at row n; fold f tests on
    // [s_f, s_f + test_size) with s_f = n - (n_splits - f) * test_size. Its
    // training rows come before s_f, and a row t is dropped when its label
    // reaches into the test block: t + label_length[t] >= s_f. max_train_size
    // keeps the most recent rows that remain. A constant label_length is
    // scikit-learn's gap, so the folds then equal TimeSeriesSplit's.
    //
    // Arguments arrive as doubles, empty when not given, so Python and
    // JavaScript share every check.

    template <typename Index>
    struct TimeSplits {
        // Fold f trains on index[offsets[2f], offsets[2f+1]) and tests on
        // index[offsets[2f+1], offsets[2f+2]).
        std::vector<Index> index;
        std::vector<std::size_t> offsets;
    };

    // The function's and its arguments' names in messages: Python's, or
    // JavaScript's camelCase ones.
    struct TimeSplitsNames {
        const char* who;
        const char* n_splits;
        const char* test_size;
        const char* max_train_size;
        const char* label_length;
        const char* drop_unlabeled;   // the call that drops rows without a label
    };
    inline constexpr TimeSplitsNames kTimeSplitsPython{
        "time_splits", "n_splits", "test_size", "max_train_size", "label_length",
        "forecast_pairs(..., dropna=True)"};
    inline constexpr TimeSplitsNames kTimeSplitsJs{
        "timeSplits", "nSplits", "testSize", "maxTrainSize", "labelLength",
        "forecastPairs(..., { dropna: true })"};

    namespace detail {
        // An integer count of at least `least`; one above the size_t range
        // (32 bits in WASM) is clamped to its top, which no later check passes
        // except max_train_size, where it means "all rows".
        inline std::size_t split_count(double x, const char* who, const char* name, double least) {
            if (!(x >= least) || x != std::floor(x))
                throw std::invalid_argument(std::string(who) + ": " + name + " must be an integer of at least " +
                                            std::to_string(static_cast<long long>(least)));
            constexpr std::size_t top = std::numeric_limits<std::size_t>::max();
            return x >= static_cast<double>(top) ? top : static_cast<std::size_t>(x);
        }
    }  // namespace detail

    // `label` is n values, or nullptr for the constant `label_scalar`.
    template <typename Index>
    TimeSplits<Index> time_splits(double n_arg, double n_splits_arg, std::optional<double> test_size_arg,
                                  std::optional<double> max_train_arg, const double* label, std::size_t n_label,
                                  double label_scalar, const TimeSplitsNames& names) {
        const auto num = [](std::size_t x) { return std::to_string(x); };
        const std::string who = std::string(names.who) + ": ";
        const std::string ll = names.label_length;
        const std::size_t n = detail::split_count(n_arg, names.who, "n", 1);
        const std::size_t splits = detail::split_count(n_splits_arg, names.who, names.n_splits, 2);
        if (splits >= n)
            throw std::invalid_argument(who + names.n_splits + " = " + num(splits) + " needs at least " +
                                        num(splits + 1) + " rows, got n = " + num(n));
        const std::size_t test_size = test_size_arg
            ? detail::split_count(*test_size_arg, names.who, names.test_size, 1) : n / (splits + 1);
        const std::size_t cap = max_train_arg
            ? detail::split_count(*max_train_arg, names.who, names.max_train_size, 1) : n;
        if (label) {
            if (n_label != n)
                throw std::invalid_argument(who + ll + " has " + num(n_label) +
                                            " values, expected n = " + num(n));
            for (std::size_t t = 0; t < n; ++t) {
                if (std::isnan(label[t]))
                    throw std::invalid_argument(who + ll + "[" + num(t) +
                                                "] is NaN; drop the rows without a label first, for example with " +
                                                names.drop_unlabeled);
                if (label[t] < 0.0)
                    throw std::invalid_argument(who + ll + "[" + num(t) + "] must not be negative");
            }
        } else {
            if (std::isnan(label_scalar))
                throw std::invalid_argument(who + ll + " is NaN");
            if (label_scalar < 0.0)
                throw std::invalid_argument(who + ll + " must not be negative");
        }
        if (test_size > (n - 1) / splits)    // splits * test_size >= n, without overflow
            throw std::invalid_argument(who + names.n_splits + " = " + num(splits) + " test blocks of " +
                                        num(test_size) + " rows need more than n = " + num(n) + " rows");
        const std::size_t first = n - splits * test_size;

        TimeSplits<Index> out;
        // The folds' total row count, checked against what a vector can hold
        // (splits * test_size < n, so only the training rows can overflow).
        const auto too_large = [&] {
            return std::invalid_argument(who + "n = " + num(n) + " is too large");
        };
        const std::size_t max_size = out.index.max_size();
        std::size_t size = splits * test_size;
        for (std::size_t f = 0; f < splits; ++f) {
            const std::size_t train = std::min(first + f * test_size, cap);
            if (size > max_size || train > max_size - size) throw too_large();
            size += train;
        }
        try {
            out.index.reserve(size);
        } catch (const std::bad_alloc&) {
            throw too_large();
        }
        out.offsets.reserve(2 * splits + 1);
        out.offsets.push_back(0);
        for (std::size_t f = 0; f < splits; ++f) {
            const std::size_t start = first + f * test_size;
            const double s = static_cast<double>(start);
            // Walk back from the test block and keep up to `cap` rows whose
            // label ends before it; the kept rows land in reverse order.
            const std::size_t lo = out.index.size();
            std::size_t kept = 0;
            for (std::size_t t = start; t-- > 0 && kept < cap;) {
                const double len = label ? label[t] : label_scalar;
                if (static_cast<double>(t) + len < s) {
                    out.index.push_back(static_cast<Index>(t));
                    ++kept;
                }
            }
            if (kept == 0)
                throw std::invalid_argument(
                    who + "fold " + num(f) + " has no training rows: every row before its test block at row " +
                    num(start) + " has a label that reaches into the block");
            std::reverse(out.index.begin() + static_cast<std::ptrdiff_t>(lo), out.index.end());
            out.offsets.push_back(out.index.size());
            for (std::size_t t = start; t < start + test_size; ++t) out.index.push_back(static_cast<Index>(t));
            out.offsets.push_back(out.index.size());
        }
        return out;
    }

}}  // namespace screamer::validation

#endif  // SCREAMER_VALIDATION_TIME_SPLITS_H
