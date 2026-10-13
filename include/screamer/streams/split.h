#ifndef SCREAMER_STREAMS_SPLIT_H
#define SCREAMER_STREAMS_SPLIT_H

// split: partition a tagged stream (values, sources, index) into one
// (values, index) stream per source, the inverse of merge. The rules and the
// partition live here, once; the Python binding (bindings/bindings_streams.cpp)
// and the WebAssembly binding (wasm/split_bindings.cpp) only move arrays in and
// out.
//
// The rules:
// - values, sources and the index (when given) are 1-D and of one length;
//   otherwise std::invalid_argument (ValueError in Python, RangeError in JS);
// - a source is a whole number from 0 to 2**32 - 1, the index-style rule
//   (checked_index.h): a fractional, NaN, negative or too large source is an
//   IndexTypeError (TypeError in both bindings);
// - n, the number of outputs, defaults to max(sources) + 1 (0 with no events);
//   a given n is a whole number (else SettingTypeError), not negative, and
//   must exceed every source, so no event is dropped (else
//   std::invalid_argument). There are at most 2**31 - 1 outputs.
//
// One pass reads and checks the tags, one counts the events per output, and
// one scatters them, O(N) in all. Each output keeps the event order (stable).

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "screamer/dag/checked_index.h"
#include "screamer/dag/max_pending.h"

namespace screamer { namespace streams {

constexpr double kMaxSplitSource = 4294967295.0;      // 2**32 - 1
constexpr std::uint32_t kMaxSplitOutputs = 2147483647u;   // 2**31 - 1, as 2.8.0's int n

[[noreturn]] inline void throw_bad_split_source(const std::string& text) {
    throw dag::IndexTypeError("split: a source must be a whole number from 0 to 4294967295, got " +
                              text);
}

// One source tag: a whole number from 0 to 2**32 - 1.
template <class S>
inline std::uint32_t split_tag(S s) {
    if constexpr (std::is_same_v<S, bool>) {
        return s ? 1u : 0u;
    } else if constexpr (std::is_floating_point_v<S>) {
        const double d = static_cast<double>(s);
        if (d >= 0.0 && d <= kMaxSplitSource && d == std::floor(d))   // NaN fails
            return static_cast<std::uint32_t>(d);
        throw_bad_split_source(dag::number_text(d));
    } else if constexpr (std::is_signed_v<S>) {
        if (s >= 0 && static_cast<std::uint64_t>(s) <= 0xFFFFFFFFull)
            return static_cast<std::uint32_t>(s);
        throw_bad_split_source(std::to_string(s));
    } else {
        if (static_cast<std::uint64_t>(s) <= 0xFFFFFFFFull) return static_cast<std::uint32_t>(s);
        throw_bad_split_source(std::to_string(s));
    }
}

// The dimensions of a call: values and sources 1-D, and one length for values,
// sources and the index (when there is one).
inline void check_split_inputs(std::size_t values_ndim, std::size_t values_len,
                               std::size_t sources_ndim, std::size_t sources_len,
                               bool has_index, std::size_t index_ndim, std::size_t index_len) {
    if (values_ndim != 1)
        throw std::invalid_argument("split: values must be 1-D (one value per event), got " +
                                    std::to_string(values_ndim) + " dimensions");
    if (sources_ndim != 1)
        throw std::invalid_argument("split: sources must be 1-D (one source per event), got " +
                                    std::to_string(sources_ndim) + " dimensions");
    if (has_index && index_ndim != 1)
        throw std::invalid_argument("split: index must be 1-D (one index per event), got " +
                                    std::to_string(index_ndim) + " dimensions");
    if (sources_len != values_len || (has_index && index_len != values_len))
        throw std::invalid_argument("split: values/sources/index length differ");
}

// The checked tags of one call, the number of outputs, and the events per
// output. Reads `src` in place when it already holds uint32 tags; the caller
// keeps `src` alive while the plan is used.
class SplitPlan {
public:
    template <class S>
    SplitPlan(const S* src, std::size_t total, bool has_n, double n) : total_(total) {
        std::uint32_t top = 0;
        if constexpr (std::is_same_v<S, std::uint32_t>) {
            tags_ = src;
            for (std::size_t j = 0; j < total; ++j) top = src[j] > top ? src[j] : top;
        } else {
            owned_.resize(total);
            for (std::size_t j = 0; j < total; ++j) {
                const std::uint32_t t = split_tag(src[j]);
                owned_[j] = t;
                top = t > top ? t : top;
            }
            tags_ = owned_.data();
        }
        counts_.assign(outputs(top, has_n, n), 0);
        for (std::size_t j = 0; j < total; ++j) counts_[tags_[j]]++;
    }

    std::size_t outputs() const { return counts_.size(); }
    const std::vector<std::size_t>& counts() const { return counts_; }

    // Writes each event's value, and its index when `k` is not null, to the
    // end of its output so far: vout[i] and kout[i] hold counts()[i] items.
    template <class Index>
    void scatter(const double* v, const Index* k, double* const* vout, Index* const* kout) const {
        std::vector<std::size_t> pos(counts_.size(), 0);
        for (std::size_t j = 0; j < total_; ++j) {
            const std::uint32_t s = tags_[j];
            const std::size_t p = pos[s]++;
            vout[s][p] = v[j];
            if (k) kout[s][p] = k[j];
        }
    }

private:
    // The number of outputs: n when given, else the largest tag + 1.
    std::size_t outputs(std::uint32_t top, bool has_n, double n) const {
        if (!has_n) {
            if (!total_) return 0;
            if (top >= kMaxSplitOutputs)
                throw std::invalid_argument("split: sources up to " + std::to_string(top) +
                                            " need more than 2**31 - 1 outputs");
            return static_cast<std::size_t>(top) + 1;
        }
        if (!std::isfinite(n) || n != std::floor(n))
            throw dag::SettingTypeError("split: n must be an integer, got " + dag::number_text(n));
        if (total_ && n <= static_cast<double>(top))
            throw std::invalid_argument("split: n=" + dag::number_text(n) +
                                        " is too small for sources up to " + std::to_string(top) +
                                        "; events would be dropped");
        if (n < 0) throw std::invalid_argument("split: n must be non-negative");
        if (n > static_cast<double>(kMaxSplitOutputs))
            throw std::invalid_argument("split: n must be at most 2**31 - 1, got " + dag::number_text(n));
        return static_cast<std::size_t>(n);
    }

    std::size_t total_;
    std::vector<std::uint32_t> owned_;
    const std::uint32_t* tags_ = nullptr;
    std::vector<std::size_t> counts_;
};

}}  // namespace screamer::streams
#endif
