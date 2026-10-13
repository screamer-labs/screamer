#ifndef SCREAMER_DAG_COMBINE_STREAM_H
#define SCREAMER_DAG_COMBINE_STREAM_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>
#include "screamer/dag/compiled_graph.h"
#include "screamer/dag/graph.h"
#include "screamer/dag/max_pending.h"

namespace screamer { namespace dag {

// combine_latest over indexed sources that hand over one event at a time when
// asked, such as JavaScript iterators and async iterators. The engine is the
// graph of a Python iterator combine_latest: one CombineLatest node over the
// inputs. Each source's next event waits as its head; step() feeds the head
// with the smallest index (ties by source order, as MergeSource) into the
// graph, so the graph sees the events of a merge by index and buffers no more
// than a pulled merge would.
//
// The caller loops: while want() names a source, offer() its next event or
// end() it; then step(), and read the rows that step settled with rows().
class CombineStream {
public:
    CombineStream(std::vector<std::size_t> widths, bool when_all,
                  std::size_t max_pending = kDefaultMaxPending)
        : widths_(std::move(widths)), heads_(widths_.size()),
          head_index_(widths_.size(), 0), state_(widths_.size(), kWanted) {
        if (widths_.empty()) throw std::invalid_argument("combine_latest: needs at least one stream");
        for (std::size_t i = 0; i < widths_.size(); ++i) heads_[i].resize(widths_[i]);
        GraphBuilder gb;
        std::vector<std::size_t> ins;
        for (std::size_t i = 0; i < widths_.size(); ++i) ins.push_back(gb.add_input());
        gb.set_outputs({gb.add_combine_latest(std::move(ins), when_all, max_pending)});
        graph_ = std::make_unique<CompiledGraph>(gb.take());
        graph_->reset();
        graph_->set_merge_order(true);   // step() feeds the smallest head first
    }

    // The source whose next event is due, or -1 when every source has a head
    // or has ended.
    std::ptrdiff_t want() const {
        for (std::size_t i = 0; i < state_.size(); ++i)
            if (state_[i] == kWanted) return static_cast<std::ptrdiff_t>(i);
        return -1;
    }

    // The next event of source i: its index and width(i) values.
    void offer(std::size_t i, std::int64_t index, const double* values) {
        check_wanted(i);
        std::copy(values, values + widths_[i], heads_[i].begin());
        head_index_[i] = index;
        state_[i] = kHead;
    }

    // Source i has no more events.
    void end(std::size_t i) {
        check_wanted(i);
        state_[i] = kEnded;
    }

    // Feeds the smallest head into the graph, or flushes the graph once every
    // source has ended. Returns false when the stream is done; rows() then
    // holds the last rows.
    bool step() {
        if (done_) return false;
        if (want() >= 0) throw std::logic_error("CombineStream: a source owes an event");
        std::ptrdiff_t pick = -1;
        for (std::size_t i = 0; i < state_.size(); ++i)
            if (state_[i] == kHead && (pick < 0 || head_index_[i] < head_index_[static_cast<std::size_t>(pick)]))
                pick = static_cast<std::ptrdiff_t>(i);
        if (pick < 0) {
            graph_->flush();
            graph_->drain_into(rows_);
            done_ = true;
            return false;
        }
        const std::size_t i = static_cast<std::size_t>(pick);
        state_[i] = kWanted;
        graph_->push_event_wide(i, head_index_[i], heads_[i].data(), widths_[i]);
        graph_->drain_into(rows_);
        return true;
    }

    std::size_t width(std::size_t i) const { return widths_[i]; }

    // The rows the last step settled: indexes, and flat rows of the inputs'
    // widths summed.
    const OutputBuffer& rows() const { return rows_[0]; }

private:
    enum State : char { kWanted, kHead, kEnded };

    void check_wanted(std::size_t i) const {
        if (i >= state_.size() || state_[i] != kWanted)
            throw std::logic_error("CombineStream: source " + std::to_string(i) + " owes no event");
    }

    std::vector<std::size_t> widths_;
    std::vector<std::vector<double>> heads_;
    std::vector<std::int64_t> head_index_;
    std::vector<State> state_;
    std::unique_ptr<CompiledGraph> graph_;
    std::vector<OutputBuffer> rows_ = std::vector<OutputBuffer>(1);
    bool done_ = false;
};

}}  // namespace screamer::dag
#endif
