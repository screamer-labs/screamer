#ifndef SCREAMER_STREAMS_MERGE_SOURCE_H
#define SCREAMER_STREAMS_MERGE_SOURCE_H

#include <cstdint>
#include <optional>
#include <queue>
#include <vector>
#include "screamer/streams/event.h"

namespace screamer { namespace streams {

// K-way merge of N individually index-sorted child sources into one globally
// index-sorted stream. Event::source is set to the child index. Ties (equal
// indices across children) break by ascending child index, deterministically.
template <class Index>
class MergeSource : public Source<Index> {
public:
    explicit MergeSource(std::vector<Source<Index>*> children)
        : children_(std::move(children)), dirty_(children_.size(), 1) {
        // No priming here: children are pulled lazily on the first next(). This
        // keeps the merge from reading further ahead than the caller has asked
        // for, which matters for live and infinite sources.
    }

    std::optional<Event<Index>> next() override {
        // Refill every slot that owes a pull (all of them on the first call,
        // then just the slot emitted by the previous call). A child that
        // returns nullopt is exhausted and stays out of the heap for good.
        for (std::uint32_t i = 0; i < children_.size(); ++i) {
            if (!dirty_[i]) continue;
            dirty_[i] = 0;
            if (auto e = children_[i]->next()) {
                heap_.push(Node{e->index, i, e->value});
            }
        }
        if (heap_.empty()) return std::nullopt;
        Node top = heap_.top();
        heap_.pop();
        // Defer this child's refill to the next call, so we never pull its
        // successor before the caller requests another event.
        dirty_[top.source] = 1;
        return Event<Index>{top.index, top.value, top.source};
    }

private:
    struct Node {
        Index index;
        std::uint32_t source;
        double value;
    };
    // Min-heap on (index, source): smaller index first; equal indices -> smaller
    // source index first (deterministic tie-break by child order).
    struct Greater {
        bool operator()(const Node& a, const Node& b) const {
            if (a.index != b.index) return a.index > b.index;
            return a.source > b.source;
        }
    };

    std::vector<Source<Index>*> children_;
    // Per-child "owes a pull" flags (uint8 to avoid vector<bool> proxies).
    std::vector<std::uint8_t> dirty_;
    std::priority_queue<Node, std::vector<Node>, Greater> heap_;
};

}} // namespace screamer::streams
#endif
