#ifndef SCREAMER_DAG_REORDER_HEAP_H
#define SCREAMER_DAG_REORDER_HEAP_H

#include <algorithm>
#include <cstddef>
#include <vector>

namespace screamer { namespace dag {

// The min-heap of a join's reorder buffer: std::priority_queue's order (the
// same push_heap/pop_heap with `Greater`), with a clear() that keeps the
// storage, so a reset allocates nothing and costs no pop per entry.
template <class T, class Greater>
class ReorderHeap {
public:
    bool empty() const { return v_.empty(); }
    std::size_t size() const { return v_.size(); }
    const T& top() const { return v_.front(); }
    void push(const T& x) {
        v_.push_back(x);
        std::push_heap(v_.begin(), v_.end(), Greater{});
    }
    void pop() {
        std::pop_heap(v_.begin(), v_.end(), Greater{});
        v_.pop_back();
    }
    void clear() { v_.clear(); }

    // The entries in the order pop() takes them (a copy), and a heap made of
    // given entries: a saved join stores its waiting events in release order.
    std::vector<T> in_order() const {
        std::vector<T> v = v_;
        std::sort(v.begin(), v.end(), [](const T& a, const T& b) { return Greater{}(b, a); });
        return v;
    }
    void assign(std::vector<T> v) {
        v_ = std::move(v);
        std::make_heap(v_.begin(), v_.end(), Greater{});
    }

private:
    std::vector<T> v_;
};

}} // namespace screamer::dag
#endif
