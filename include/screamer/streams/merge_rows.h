#ifndef SCREAMER_STREAMS_MERGE_ROWS_H
#define SCREAMER_STREAMS_MERGE_ROWS_H

// The merge of N streams into one in index order, shared by the Python and
// JavaScript bindings. Each stream holds rows of one width (a 1-D stream has
// width 1); every stream of a merge must have the same width. Equal indexes
// keep the input order (a lower input first), as streams::MergeSource does.
//
// merge_rows() is the batch form over arrays; MergeOrder is the iterator
// form, which picks the next source while the binding pulls events.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/dag/checked_index.h"
#include "screamer/streams/event.h"
#include "screamer/streams/merge_source.h"
#include "screamer/streams/vector_source.h"

namespace screamer { namespace streams {

// The width rule of a merge: every input has rows of one width. `first` is
// the input that set the width; the message names both widths.
// `event` >= 0 names the event of an iterator source whose row differs.
[[noreturn]] inline void throw_merge_width(const std::string& who, std::size_t first,
                                           std::size_t first_width, std::size_t input,
                                           std::size_t width, std::ptrdiff_t event = -1) {
    const std::string at = event >= 0
        ? "event " + std::to_string(event) + " of input " + std::to_string(input) + " has width "
        : "input " + std::to_string(input) + " has rows of width ";
    throw std::invalid_argument(who + ": " + at + std::to_string(width) + " but input " +
                                std::to_string(first) + " has rows of width " +
                                std::to_string(first_width) +
                                "; every input of a merge has the same width");
}

// The row shape of an input in a message: "()" for a 1-D stream, "(2, 3)".
inline std::string row_shape_text(const std::vector<std::size_t>& shape) {
    std::string t = "(";
    for (std::size_t d = 0; d < shape.size(); ++d)
        t += (d ? ", " : "") + std::to_string(shape[d]);
    return t + (shape.size() == 1 ? ",)" : ")");
}

// Records the row shape of the first input that has one and checks the
// others: the width (values per row) first, then the shape.
class MergeWidth {
public:
    explicit MergeWidth(std::string who = "merge") : who_(std::move(who)) {}
    void check(std::size_t input, const std::vector<std::size_t>& row_shape,
               std::ptrdiff_t event = -1) {
        std::size_t w = 1;
        for (std::size_t d : row_shape) w *= d;
        if (!set_) {
            set_ = true;
            first_ = input;
            width_ = w;
            shape_ = row_shape;
            return;
        }
        if (w != width_) throw_merge_width(who_, first_, width_, input, w, event);
        if (row_shape != shape_)
            throw std::invalid_argument(who_ + ": " +
                                        (event >= 0 ? "event " + std::to_string(event) + " of " : std::string()) +
                                        "input " + std::to_string(input) +
                                        (event >= 0 ? " has shape " : " has rows of shape ") +
                                        row_shape_text(row_shape) +
                                        " but input " + std::to_string(first_) +
                                        " has rows of shape " + row_shape_text(shape_) +
                                        "; every input of a merge has the same row shape");
    }
    bool known() const { return set_; }
    std::size_t width() const { return set_ ? width_ : 1; }
    const std::vector<std::size_t>& shape() const { return shape_; }

private:
    std::string who_;
    bool set_ = false;
    std::size_t first_ = 0;
    std::size_t width_ = 1;
    std::vector<std::size_t> shape_;
};

namespace detail {

// The events of one input of a wide merge: its index, with the row number as
// the event value, so the merge orders rows without reading them.
template <class Index>
class RowSource : public Source<Index> {
public:
    RowSource(const Index* indices, std::size_t size, std::size_t input)
        : indices_(indices), size_(size), input_(input) {}

    std::optional<Event<Index>> next() override {
        if (i_ >= size_) return std::nullopt;
        if (i_ != 0 && indices_[i_] < indices_[i_ - 1])
            dag::throw_index_decreases(input_, i_, indices_[i_ - 1], indices_[i_]);
        Event<Index> e{indices_[i_], static_cast<double>(i_), 0};
        ++i_;
        return e;
    }

private:
    const Index* indices_;
    std::size_t size_;
    std::size_t i_ = 0;
    std::size_t input_;
};

}  // namespace detail

// Merges `index.size()` streams. Stream i has rows[i] rows of `width` values
// at values[i] (row-major) and rows[i] index values at index[i]. Writes the
// merged index, rows and source of each row to the outputs, which hold
// sum(rows) entries (times `width` for out_values). A decreasing index raises
// the index-order error (checked_index.h).
template <class Index>
void merge_rows(const std::vector<const Index*>& index, const std::vector<const double*>& values,
                const std::vector<std::size_t>& rows, std::size_t width, Index* out_index,
                double* out_values, std::uint32_t* out_source) {
    const std::size_t n = index.size();
    std::vector<Source<Index>*> children(n);
    std::size_t k = 0;
    if (width == 1) {
        // One value per row: the value rides in the event.
        std::vector<VectorSource<Index>> src;
        src.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            src.emplace_back(index[i], values[i], rows[i], i);
            children[i] = &src[i];
        }
        MergeSource<Index> merge(children);
        while (auto e = merge.next()) {
            out_index[k] = e->index;
            out_values[k] = e->value;
            out_source[k] = e->source;
            ++k;
        }
        return;
    }
    std::vector<detail::RowSource<Index>> src;
    src.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        src.emplace_back(index[i], rows[i], i);
        children[i] = &src[i];
    }
    MergeSource<Index> merge(children);
    while (auto e = merge.next()) {
        out_index[k] = e->index;
        out_source[k] = e->source;
        const std::size_t row = static_cast<std::size_t>(e->value);
        if (width)
            std::memcpy(out_values + k * width, values[e->source] + row * width,
                        width * sizeof(double));
        ++k;
    }
}

// The iterator merge: the binding pulls one event at a time from the source
// that want() names, offers its index (or ends the source), and once want()
// is -1, pop() gives the source whose pending event comes next. That event's
// values stay with the binding; MergeOrder holds only the indexes.
template <class Index>
class MergeOrder {
public:
    explicit MergeOrder(std::size_t n, bool check_order = true)
        : owes_(n, 1), ended_(n, 0), events_(n, 0), last_(n), check_order_(check_order) {}

    // A source that owes its next event (pulled lazily, one per pop), or -1.
    std::ptrdiff_t want() const {
        for (std::size_t i = 0; i < owes_.size(); ++i)
            if (owes_[i] && !ended_[i]) return static_cast<std::ptrdiff_t>(i);
        return -1;
    }

    // The next event of source i has index `index`.
    void offer(std::size_t i, Index index) {
        const std::size_t ev = events_[i]++;
        if (check_order_ && ev != 0 && index < last_[i])
            dag::throw_index_decreases(i, ev, last_[i], index);
        last_[i] = index;
        owes_[i] = 0;
        heap_.push(Node{index, static_cast<std::uint32_t>(i)});
    }

    // Source i has no more events.
    void end(std::size_t i) {
        ended_[i] = 1;
        owes_[i] = 0;
    }

    // The source of the next event in index order (it then owes its next
    // event), or -1 when every source has ended. Call once want() is -1.
    std::ptrdiff_t pop() {
        if (heap_.empty()) return -1;
        const Node top = heap_.top();
        heap_.pop();
        owes_[top.source] = 1;
        return static_cast<std::ptrdiff_t>(top.source);
    }

    // The index of the event pop() gave last.
    Index index_of(std::size_t i) const { return last_[i]; }

private:
    struct Node {
        Index index;
        std::uint32_t source;
    };
    struct Greater {
        bool operator()(const Node& a, const Node& b) const {
            if (a.index != b.index) return a.index > b.index;
            return a.source > b.source;
        }
    };
    std::vector<std::uint8_t> owes_;
    std::vector<std::uint8_t> ended_;
    std::vector<std::size_t> events_;
    std::vector<Index> last_;
    bool check_order_;
    std::priority_queue<Node, std::vector<Node>, Greater> heap_;
};

}}  // namespace screamer::streams
#endif
