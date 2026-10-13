#ifndef SCREAMER_DAG_OUTPUT_JOIN_H
#define SCREAMER_DAG_OUTPUT_JOIN_H

// Output alignment of a multi-output Pipeline (align_outputs). The M outputs of
// a compiled graph each carry their own index. Aligning them puts every output
// on one index: the sorted union of the output indexes, one row per distinct
// index, holding each output's latest value at that index (an as-of join, as
// CombineLatest with emit="when_all"). Rows start once every output has a value.
//
// One rule serves every regime and both bindings:
//   * align_outputs(buffers): batch, and a live session's result(), over the
//     buffers that a run or a drain returned.
//   * OutputJoin: the join across drains, for a lazy call and for a live
//     session's result() (CompiledGraph::drain_aligned). It carries each
//     output's latest value from one drain to the next and settles an index
//     once every output is past it (it emitted a row, or a watermark, at a
//     higher index), so the rows of all drains, concatenated, are the rows of
//     align_outputs over the whole run, whatever the drain timing.
//
// The outputs are merged by index; at one index, the last row of each output
// wins. Each output of a graph is sorted by index.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "screamer/dag/max_pending.h"
#include "screamer/dag/output_buffer.h"

namespace screamer { namespace dag {

// Aligned outputs: one index, and each output's values on it in a row-major
// block of widths[j] columns.
struct AlignedOutputs {
    std::vector<std::int64_t> index;
    std::vector<std::vector<double>> values;   // values[j]: rows * widths[j]
    std::vector<std::size_t> widths;

    std::size_t rows() const { return index.size(); }
    void clear_rows() {
        index.clear();
        for (auto& v : values) v.clear();
    }
};

namespace detail {

// The join state: each output's latest value, and which outputs have one. A
// value is held as a pointer to the row it came from; persist() copies the
// values in, for rows whose storage is about to be reused.
class AsOfRow {
public:
    void layout(const std::vector<std::size_t>& widths) {
        widths_ = widths;
        offset_.resize(widths.size());
        std::size_t total = 0;
        for (std::size_t j = 0; j < widths.size(); ++j) { offset_[j] = total; total += widths[j]; }
        own_.assign(total, 0.0);
        ptr_.assign(widths.size(), nullptr);
        seen_count_ = 0;
    }
    // Output j's latest value is the row at `v`.
    void point(std::size_t j, const double* v) {
        if (!ptr_[j]) ++seen_count_;
        ptr_[j] = v;
    }
    // Counting only: output j has a value.
    void mark(std::size_t j) {
        static const double none = 0.0;
        point(j, &none);
    }
    // Changes the width of output j, which has no value yet; the others keep
    // theirs.
    void set_width(std::size_t j, std::size_t w) {
        if (widths_[j] == w) return;
        persist();
        const std::vector<double> old = own_;
        const std::vector<std::size_t> old_offset = offset_, old_width = widths_;
        widths_[j] = w;
        std::size_t total = 0;
        for (std::size_t k = 0; k < widths_.size(); ++k) { offset_[k] = total; total += widths_[k]; }
        own_.assign(total, 0.0);
        for (std::size_t k = 0; k < widths_.size(); ++k) {
            if (!ptr_[k]) continue;
            std::copy(old.begin() + static_cast<std::ptrdiff_t>(old_offset[k]),
                      old.begin() + static_cast<std::ptrdiff_t>(old_offset[k] + old_width[k]),
                      own_.begin() + static_cast<std::ptrdiff_t>(offset_[k]));
            ptr_[k] = own_.data() + offset_[k];
        }
    }
    bool complete() const { return seen_count_ == widths_.size(); }
    void persist() {
        for (std::size_t j = 0; j < widths_.size(); ++j) {
            double* dst = own_.data() + offset_[j];
            if (!ptr_[j] || ptr_[j] == dst) continue;
            std::copy(ptr_[j], ptr_[j] + widths_[j], dst);
            ptr_[j] = dst;
        }
    }
    void append_to(AlignedOutputs& out) const {
        for (std::size_t j = 0; j < widths_.size(); ++j) {
            const double* src = ptr_[j];
            if (widths_[j] == 1) out.values[j].push_back(*src);
            else out.values[j].insert(out.values[j].end(), src, src + widths_[j]);
        }
    }

private:
    std::vector<std::size_t> widths_, offset_;
    std::vector<double> own_;              // persisted values
    std::vector<const double*> ptr_;       // each output's latest row (null = none yet)
    std::size_t seen_count_ = 0;
};

// Merges the rows of `src` by index, one distinct index at a time: every row of
// every output at that index is applied (the last one of an output wins), then
// one aligned row goes out if every output has a value. With `has_bound` it
// stops at the first index that is not below `bound`; every row of a lower
// index must then already be in `src`. With Write false it only counts rows.
// Src: n(), empty(j), head(j), head_values(j), pop(j).
template <bool Write, class Src>
std::size_t merge_rows(Src& src, bool has_bound, std::int64_t bound, AsOfRow& row,
                       AlignedOutputs* out) {
    const std::size_t n = src.n();
    std::size_t emitted = 0;
    for (;;) {
        bool any = false;
        std::int64_t k = 0;
        for (std::size_t j = 0; j < n; ++j) {
            if (src.empty(j)) continue;
            const std::int64_t h = src.head(j);
            if (!any || h < k) { k = h; any = true; }
        }
        if (!any || (has_bound && k >= bound)) break;
        for (std::size_t j = 0; j < n; ++j) {
            if (src.empty(j) || src.head(j) != k) continue;
            const double* last = nullptr;
            do {
                if constexpr (Write) last = src.head_values(j);
                src.pop(j);
            } while (!src.empty(j) && src.head(j) == k);
            if constexpr (Write) row.point(j, last);
            else row.mark(j);
        }
        if (row.complete()) {
            if constexpr (Write) { out->index.push_back(k); row.append_to(*out); }
            ++emitted;
        }
    }
    return emitted;
}

// Reads drained buffers in place.
struct BufferCursor {
    const std::vector<OutputBuffer>& b;
    std::vector<std::size_t> pos;
    explicit BufferCursor(const std::vector<OutputBuffer>& bufs) : b(bufs), pos(bufs.size(), 0) {}
    std::size_t n() const { return b.size(); }
    bool empty(std::size_t j) const { return pos[j] >= b[j].indices.size(); }
    std::int64_t head(std::size_t j) const { return b[j].indices[pos[j]]; }
    const double* head_values(std::size_t j) const { return b[j].values.data() + pos[j] * b[j].width; }
    void pop(std::size_t j) { ++pos[j]; }
};

}  // namespace detail

namespace detail {

// Whether every buffer has the same strictly increasing index. The aligned
// outputs are then the buffers themselves.
inline bool one_strict_index(const std::vector<OutputBuffer>& bufs) {
    const std::vector<std::int64_t>& k = bufs[0].indices;
    for (std::size_t r = 1; r < k.size(); ++r)
        if (k[r] <= k[r - 1]) return false;
    for (std::size_t j = 1; j < bufs.size(); ++j)
        if (bufs[j].indices != k) return false;
    return true;
}

}  // namespace detail

// Aligns the buffers of one run (or one drain) of a graph. A first pass over
// the indexes counts the rows, so every array is allocated once at its size.
inline AlignedOutputs align_outputs(const std::vector<OutputBuffer>& bufs) {
    AlignedOutputs out;
    out.widths.resize(bufs.size());
    for (std::size_t j = 0; j < bufs.size(); ++j) out.widths[j] = bufs[j].width;
    out.values.resize(bufs.size());
    detail::AsOfRow row;
    row.layout(out.widths);
    std::size_t rows;
    {
        detail::BufferCursor c(bufs);
        rows = detail::merge_rows<false>(c, false, 0, row, nullptr);
    }
    if (rows == 0) return out;
    out.index.reserve(rows);
    for (std::size_t j = 0; j < bufs.size(); ++j) out.values[j].reserve(rows * out.widths[j]);
    row.layout(out.widths);
    detail::BufferCursor c(bufs);
    detail::merge_rows<true>(c, false, 0, row, &out);
    return out;
}

// align_outputs() of buffers the caller gives up: when every output has the
// same strictly increasing index (outputs of one stream, say), the buffers
// move into the result without a copy.
inline AlignedOutputs align_outputs(std::vector<OutputBuffer>&& bufs) {
    if (bufs.empty() || !detail::one_strict_index(bufs))
        return align_outputs(static_cast<const std::vector<OutputBuffer>&>(bufs));
    AlignedOutputs out;
    out.widths.resize(bufs.size());
    out.values.resize(bufs.size());
    for (std::size_t j = 0; j < bufs.size(); ++j) {
        out.widths[j] = bufs[j].width;
        out.values[j] = std::move(bufs[j].values);
    }
    out.index = std::move(bufs[0].indices);
    return out;
}

// The join across drains, for a lazy call and a live session's result():
// take() each drain, with `end` once the input has ended (a flush). Each output
// is bounded by the higher of its last emitted index and its watermark (no row
// will come below either); an index is settled once it is below every output's
// bound, and at the end every row is settled. Each output's latest value
// carries over to the next drain. The rows that wait, held as doubles, are the
// lag between the outputs: an output that emits rarely (a long resample
// window) holds the others' rows of its open window, as a batch run does. An
// output whose rows are all dropped (a Filter, Dropna) still advances by its
// watermarks, so it holds nothing. In a live session (take's `capped`), more
// than `max_pending` waiting rows after a drain raise, as a join's reorder
// buffer does: an output has stalled. A lazy call holds them as batch does.
class OutputJoin {
public:
    explicit OutputJoin(std::size_t n_out = 0, std::size_t max_pending = kDefaultMaxPending)
        : q_(n_out), max_pending_(max_pending) {
        out_.values.resize(n_out);
        out_.widths.assign(n_out, 1);
    }

    void reset() {
        for (Queue& q : q_) q = Queue{};
        std::fill(out_.widths.begin(), out_.widths.end(), std::size_t(1));
        out_.clear_rows();
        laid_out_ = false;
        pending_ = 0;
    }

    // The rows one drain settles: one buffer per output, each sorted by index.
    // With `end`, every row that waits is settled too. The first drain of a
    // run that ends with it (a batch run, or one live result() after a flush)
    // is align_outputs of the buffers, which take their storage.
    AlignedOutputs take(std::vector<OutputBuffer>& bufs, bool end, bool capped = true) {
        check_count(bufs);
        if (end && fresh()) {
            seed(bufs);
            return align_outputs(std::move(bufs));
        }
        push(bufs);
        if (end && pending_) settle(false, 0);
        if (capped && pending_ > max_pending_)
            throw std::runtime_error(
                "output alignment overflow: " + std::to_string(pending_) +
                " rows wait for an output that has not advanced; a stream is stalled. "
                "Call advance(now) or check the feed.");
        AlignedOutputs rows = std::move(out_);
        out_ = AlignedOutputs{};
        out_.values.resize(q_.size());
        out_.widths = rows.widths;
        return rows;
    }

    std::size_t pending() const { return pending_; }

private:
    struct Queue {
        std::vector<std::int64_t> idx;
        std::vector<double> vals;
        std::size_t head = 0;      // rows before head are consumed
        std::size_t width = 1;
        std::int64_t wm = 0;       // highest index emitted
        bool emitted = false;      // whether the output has emitted a row
        std::int64_t bound = 0;    // max(highest index emitted, watermark)
        bool bounded = false;      // whether `bound` is set
        // Drops the consumed rows once they are at least half the queue, so
        // the storage is reused and bounded by the rows that wait.
        void compact() {
            if (head == 0) return;
            if (head == idx.size()) { idx.clear(); vals.clear(); head = 0; return; }
            if (head * 2 < idx.size()) return;
            idx.erase(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(head));
            vals.erase(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(head * width));
            head = 0;
        }
    };

    struct QueueCursor {
        std::vector<Queue>& q;
        std::size_t popped = 0;
        std::size_t n() const { return q.size(); }
        bool empty(std::size_t j) const { return q[j].head >= q[j].idx.size(); }
        std::int64_t head(std::size_t j) const { return q[j].idx[q[j].head]; }
        const double* head_values(std::size_t j) const {
            return q[j].vals.data() + q[j].head * q[j].width;
        }
        void pop(std::size_t j) { ++q[j].head; ++popped; }
    };

    void check_count(const std::vector<OutputBuffer>& bufs) const {
        if (bufs.size() != q_.size())
            throw std::invalid_argument(
                "output alignment: expected " + std::to_string(q_.size()) +
                " outputs, got " + std::to_string(bufs.size()));
    }

    // Nothing taken since the reset: no row waits and no output has a value.
    bool fresh() const {
        if (laid_out_ || pending_) return false;
        for (const Queue& q : q_) if (q.emitted) return false;
        return true;
    }

    // Output j's first row fixes its width.
    void learn_width(std::size_t j, std::size_t w) {
        Queue& q = q_[j];
        if (q.emitted) {
            if (w != q.width)
                throw std::invalid_argument(
                    "output " + std::to_string(j) + " changed from " +
                    std::to_string(q.width) + " to " + std::to_string(w) + " columns");
            return;
        }
        q.width = w;
        out_.widths[j] = w;
        if (laid_out_) row_.set_width(j, w);
    }

    // The state after align_outputs(bufs) as one drain that ends: each
    // output's last row is its latest value, and nothing waits.
    void seed(const std::vector<OutputBuffer>& bufs) {
        for (std::size_t j = 0; j < bufs.size(); ++j) {
            const OutputBuffer& b = bufs[j];
            out_.widths[j] = b.width;
            if (b.indices.empty()) continue;
            learn_width(j, b.width);
            q_[j].wm = b.indices.back();
            q_[j].emitted = true;
            raise_bound(q_[j], b.indices.back());
        }
        for (std::size_t j = 0; j < bufs.size(); ++j) {
            if (bufs[j].has_watermark) raise_bound(q_[j], bufs[j].watermark);
        }
        row_.layout(out_.widths);
        laid_out_ = true;
        for (std::size_t j = 0; j < bufs.size(); ++j) {
            const OutputBuffer& b = bufs[j];
            if (!b.indices.empty())
                row_.point(j, b.values.data() + (b.indices.size() - 1) * b.width);
        }
        row_.persist();                   // before the buffers move out
    }

    void push(const std::vector<OutputBuffer>& bufs) {
        bool all_bounded = true;
        for (std::size_t j = 0; j < bufs.size(); ++j) {
            const OutputBuffer& b = bufs[j];
            Queue& q = q_[j];
            if (!b.indices.empty()) {
                learn_width(j, b.width);
                q.idx.insert(q.idx.end(), b.indices.begin(), b.indices.end());
                q.vals.insert(q.vals.end(), b.values.begin(), b.values.end());
                pending_ += b.indices.size();
                const std::int64_t last = b.indices.back();
                if (!q.emitted || last > q.wm) q.wm = last;
                q.emitted = true;
                raise_bound(q, last);
            } else if (!q.emitted && q.width != b.width) {
                // The graph's width for an output without a row yet: it can
                // grow once an input's width is known, before any row.
                q.width = b.width;
                out_.widths[j] = b.width;
                if (laid_out_) row_.set_width(j, b.width);
            }
            if (b.has_watermark) raise_bound(q, b.watermark);
            all_bounded &= q.bounded;
        }
        if (all_bounded) {
            std::int64_t bound = q_[0].bound;
            for (const Queue& q : q_) bound = std::min(bound, q.bound);
            settle(true, bound);
        }
    }

    static void raise_bound(Queue& q, std::int64_t b) {
        if (!q.bounded || b > q.bound) { q.bound = b; q.bounded = true; }
    }

    void settle(bool has_bound, std::int64_t bound) {
        if (!laid_out_) {
            row_.layout(out_.widths);
            laid_out_ = true;
        }
        QueueCursor c{q_};
        detail::merge_rows<true>(c, has_bound, bound, row_, &out_);
        pending_ -= c.popped;
        row_.persist();                   // before compact() moves the rows
        for (Queue& q : q_) q.compact();
    }

    std::vector<Queue> q_;
    std::size_t max_pending_;        // the most rows that may wait
    detail::AsOfRow row_;
    bool laid_out_ = false;
    std::size_t pending_ = 0;        // rows held in the queues
    AlignedOutputs out_;             // settled rows of the current take()
};

}}  // namespace screamer::dag
#endif
