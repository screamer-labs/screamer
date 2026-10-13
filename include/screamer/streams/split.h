#ifndef SCREAMER_STREAMS_SPLIT_H
#define SCREAMER_STREAMS_SPLIT_H

// split: partition a tagged stream (values, sources, index) into one
// (values, index) stream per source, the inverse of merge. The rules and the
// partition live here, once; the Python binding (bindings/bindings_streams.cpp)
// and the WebAssembly binding (wasm/split_bindings.cpp) only move arrays in and
// out.
//
// The rules:
// - values have a time axis: shape (T, ...), each event a row of K values
//   (K = 1 for 1-D values); sources and the index (when given) are 1-D of
//   length T; otherwise std::invalid_argument (ValueError in Python, RangeError
//   in JavaScript);
// - a source is a whole number: a bool, a fractional or a non-finite source is
//   an IndexTypeError (TypeError in both bindings), as an index of the wrong
//   kind is; a whole source outside 0 to 2**31 - 2 is std::invalid_argument;
// - n, the number of outputs, defaults to max(sources) + 1 (0 with no events);
//   a given n is a whole number (else SettingTypeError), not negative, at most
//   2**31 - 1, and must exceed every source, so no event is dropped (else
//   std::invalid_argument).
//
// One pass reads, checks and counts the tags, and one scatters the events,
// O(N) in all. Each output keeps the event order (stable).

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "screamer/dag/checked_index.h"
#include "screamer/dag/max_pending.h"

namespace screamer { namespace streams {

constexpr std::uint32_t kMaxSplitOutputs = 2147483647u;            // 2**31 - 1
constexpr std::uint32_t kMaxSplitSource = kMaxSplitOutputs - 1;    // 2**31 - 2

// A source that is not a whole number; `text` is the value or its kind.
[[noreturn]] inline void throw_split_source_kind(const std::string& text) {
    throw dag::IndexTypeError("split: a source must be a whole number, got " + text);
}

// A whole source outside the range of outputs.
[[noreturn]] inline void throw_split_source_range(const std::string& text) {
    throw std::invalid_argument("split: a source must be from 0 to 2147483646, got " + text);
}

// An n that is not a whole number; `text` is the value or its kind.
[[noreturn]] inline void throw_split_n_kind(const std::string& text) {
    throw dag::SettingTypeError("split: n must be an integer, got " + text);
}

// More outputs than memory holds (a large source id makes one output per id
// up to it).
[[noreturn]] inline void throw_split_memory(std::size_t outputs) {
    throw std::invalid_argument(
        "split: " + std::to_string(outputs) + " outputs need more memory than is "
        "available; pass smaller source ids");
}

// Runs `f`, which allocates the storage of `outputs` outputs; an allocation
// that fails raises throw_split_memory.
template <class F>
inline auto split_alloc(std::size_t outputs, F&& f) -> decltype(f()) {
    try { return f(); }
    catch (const std::bad_alloc&) { throw_split_memory(outputs); }
    catch (const std::length_error&) { throw_split_memory(outputs); }
}

// One source tag.
template <class S>
inline std::uint32_t split_tag(S s) {
    if constexpr (std::is_same_v<S, bool>) {
        throw_split_source_kind("a bool");
    } else if constexpr (std::is_floating_point_v<S>) {
        const double d = static_cast<double>(s);
        if (!std::isfinite(d) || d != std::floor(d)) throw_split_source_kind(dag::number_text(d));
        if (d < 0.0 || d > kMaxSplitSource) throw_split_source_range(dag::number_text(d));
        return static_cast<std::uint32_t>(d);
    } else if constexpr (std::is_signed_v<S>) {
        if (s < 0 || static_cast<std::uint64_t>(s) > kMaxSplitSource)
            throw_split_source_range(std::to_string(s));
        return static_cast<std::uint32_t>(s);
    } else {
        if (static_cast<std::uint64_t>(s) > kMaxSplitSource) throw_split_source_range(std::to_string(s));
        return static_cast<std::uint32_t>(s);
    }
}

// The dimensions of a call: values with a time axis of T rows, sources 1-D of
// T, and the index (when there is one) 1-D of T.
inline void check_split_inputs(std::size_t values_ndim, std::size_t values_rows,
                               std::size_t sources_ndim, std::size_t sources_len,
                               bool has_index, std::size_t index_ndim, std::size_t index_len) {
    if (values_ndim < 1)
        throw std::invalid_argument("split: values must have a time axis, got 0 dimensions");
    if (sources_ndim != 1)
        throw std::invalid_argument("split: sources must be 1-D (one source per event), got " +
                                    std::to_string(sources_ndim) + " dimensions");
    if (has_index && index_ndim != 1)
        throw std::invalid_argument("split: index must be 1-D (one index per event), got " +
                                    std::to_string(index_ndim) + " dimensions");
    if (sources_len != values_rows)
        throw std::invalid_argument("split: the stream has " + std::to_string(values_rows) +
                                    " rows of values but " + std::to_string(sources_len) +
                                    " sources");
    if (has_index) dag::check_feed_length("split", -1, values_rows, index_len);
}

// The checked tags of one call, the number of outputs, and the events per
// output. Reads `src` in place when it holds uint32 tags already in range; the
// caller keeps `src` alive while the plan is used.
class SplitPlan {
public:
    template <class S>
    SplitPlan(const S* src, std::size_t total, bool has_n, double n)
        : total_(total), limit_(count_limit(has_n, n)) {
        if constexpr (std::is_same_v<S, std::uint32_t>) {
            std::size_t j = 0;
            for (; j < total && src[j] <= kMaxSplitSource; ++j) count(src[j]);
            if (j == total) {
                tags_ = src;
                finish(has_n, n);
                return;
            }
            counts_.clear();               // a tag out of range: the checked path reports it
            top_ = 0;
        }
        owned_.resize(total);
        for (std::size_t j = 0; j < total; ++j) {
            const std::uint32_t t = split_tag(src[j]);
            owned_[j] = t;
            count(t);
        }
        tags_ = owned_.data();
        finish(has_n, n);
    }

    // From tags a binding has checked with split_tag already.
    SplitPlan(std::vector<std::uint32_t>&& tags, bool has_n, double n)
        : total_(tags.size()), limit_(count_limit(has_n, n)), owned_(std::move(tags)) {
        for (std::uint32_t t : owned_) count(t);
        tags_ = owned_.data();
        finish(has_n, n);
    }

    SplitPlan(const SplitPlan&) = delete;
    SplitPlan& operator=(const SplitPlan&) = delete;

    std::size_t outputs() const { return counts_.size(); }
    const std::vector<std::size_t>& counts() const { return counts_; }

    // Writes each event's row of `width` values, and its index when `k` is not
    // null, to the end of its output so far: vout[i] holds counts()[i] rows
    // and kout[i] counts()[i] items.
    template <class Index>
    void scatter(const double* v, std::size_t width, const Index* k,
                 double* const* vout, Index* const* kout) const {
        std::vector<std::size_t> pos(counts_.size(), 0);
        // One loop per case, so the per-event loop has no branch.
        if (width == 1 && k) {
            Index prev = total_ ? k[0] : Index{};
            for (std::size_t j = 0; j < total_; ++j) {
                const Index kj = k[j];
                if (kj < prev) index_decreases(j, k);
                prev = kj;
                const std::uint32_t s = tags_[j];
                const std::size_t p = pos[s]++;
                vout[s][p] = v[j];
                kout[s][p] = kj;
            }
        } else if (width == 1) {
            for (std::size_t j = 0; j < total_; ++j) {
                const std::uint32_t s = tags_[j];
                vout[s][pos[s]++] = v[j];
            }
        } else {
            for (std::size_t j = 0; j < total_; ++j) {
                const std::uint32_t s = tags_[j];
                const std::size_t p = pos[s]++;
                if (width) std::memcpy(vout[s] + p * width, v + j * width, width * sizeof(double));
                if (k) {
                    if (j != 0 && k[j] < k[j - 1]) index_decreases(j, k);
                    kout[s][p] = k[j];
                }
            }
        }
    }

private:
    // The index of the stream must not decrease (checked_index.h), checked in
    // the scatter loop; the caller's output buffers own their storage.
    template <class Index>
    [[noreturn]] static void index_decreases(std::size_t j, const Index* k) {
        dag::throw_index_decreases(std::string("the split stream"), j, k[j - 1], k[j]);
    }

    // The outputs a given n allows (n itself is checked in finish): a tag at or
    // beyond it is not counted, so a small n never sizes counts_ by a large tag.
    static std::size_t count_limit(bool has_n, double n) {
        if (!has_n || !std::isfinite(n) || n != std::floor(n)) return SIZE_MAX;
        if (n <= 0) return 0;
        return n >= kMaxSplitOutputs ? kMaxSplitOutputs : static_cast<std::size_t>(n);
    }

    // Counts one event of a checked tag; counts_ grows to the largest tag + 1
    // as the tags are read, so no separate counting pass is needed.
    void count(std::uint32_t t) {
        top_ = t > top_ ? t : top_;
        if (t >= limit_) return;                       // finish reports n too small
        if (t >= counts_.size()) grow(static_cast<std::size_t>(t) + 1);
        ++counts_[t];
    }

    // The number of outputs: n when given, else the largest tag + 1.
    void finish(bool has_n, double n) {
        if (!has_n) return;
        if (!std::isfinite(n) || n != std::floor(n)) throw_split_n_kind(dag::number_text(n));
        if (total_ && n <= static_cast<double>(top_))
            throw std::invalid_argument("split: n=" + dag::number_text(n) +
                                        " is too small for sources up to " + std::to_string(top_) +
                                        "; events would be dropped");
        if (n < 0) throw std::invalid_argument("split: n must be non-negative");
        if (n > static_cast<double>(kMaxSplitOutputs))
            throw std::invalid_argument("split: n must be at most 2**31 - 1, got " +
                                        dag::number_text(n));
        grow(static_cast<std::size_t>(n));
    }

    void grow(std::size_t size) {
        split_alloc(size, [&] { counts_.resize(size, 0); });
    }

    std::size_t total_;
    std::size_t limit_;
    std::uint32_t top_ = 0;
    std::vector<std::uint32_t> owned_;
    const std::uint32_t* tags_ = nullptr;
    std::vector<std::size_t> counts_;
};

}}  // namespace screamer::streams
#endif
