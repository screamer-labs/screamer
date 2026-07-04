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
        : children_(std::move(children)) {
        // Prime the heap with the first event from each child.
        for (std::uint32_t i = 0; i < children_.size(); ++i) {
            if (auto e = children_[i]->next()) {
                heap_.push(Node{e->index, i, e->value});
            }
        }
    }

    std::optional<Event<Index>> next() override {
        if (heap_.empty()) return std::nullopt;
        Node top = heap_.top();
        heap_.pop();
        // Pull the child's next event to keep the heap primed.
        if (auto e = children_[top.source]->next()) {
            heap_.push(Node{e->index, top.source, e->value});
        }
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
    std::priority_queue<Node, std::vector<Node>, Greater> heap_;
};

}} // namespace screamer::streams
#endif
