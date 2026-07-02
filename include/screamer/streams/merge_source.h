#ifndef SCREAMER_STREAMS_MERGE_SOURCE_H
#define SCREAMER_STREAMS_MERGE_SOURCE_H

#include <cstdint>
#include <queue>
#include <vector>
#include "screamer/streams/event.h"

namespace screamer { namespace streams {

// K-way merge of N individually key-sorted child sources into one globally
// key-sorted stream. Event::source is set to the child index. Ties (equal
// keys across children) break by ascending child index, deterministically.
template <class Key>
class MergeSource : public Source<Key> {
public:
    explicit MergeSource(std::vector<Source<Key>*> children)
        : children_(std::move(children)) {
        // Prime the heap with the first event from each child.
        for (std::uint32_t i = 0; i < children_.size(); ++i) {
            if (auto e = children_[i]->next()) {
                heap_.push(Node{e->key, i, e->value});
            }
        }
    }

    std::optional<Event<Key>> next() override {
        if (heap_.empty()) return std::nullopt;
        Node top = heap_.top();
        heap_.pop();
        // Pull the child's next event to keep the heap primed.
        if (auto e = children_[top.source]->next()) {
            heap_.push(Node{e->key, top.source, e->value});
        }
        return Event<Key>{top.key, top.value, top.source};
    }

private:
    struct Node {
        Key key;
        std::uint32_t source;
        double value;
    };
    // Min-heap on (key, source): smaller key first; equal keys -> smaller
    // source index first (deterministic tie-break by child order).
    struct Greater {
        bool operator()(const Node& a, const Node& b) const {
            if (a.key != b.key) return a.key > b.key;
            return a.source > b.source;
        }
    };

    std::vector<Source<Key>*> children_;
    std::priority_queue<Node, std::vector<Node>, Greater> heap_;
};

}} // namespace screamer::streams
#endif
