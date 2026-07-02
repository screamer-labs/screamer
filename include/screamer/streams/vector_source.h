#ifndef SCREAMER_STREAMS_VECTOR_SOURCE_H
#define SCREAMER_STREAMS_VECTOR_SOURCE_H

#include <cstddef>
#include "screamer/streams/event.h"

namespace screamer { namespace streams {

template <class Key>
class VectorSource : public Source<Key> {
public:
    VectorSource(const Key* keys, const double* values, std::size_t size)
        : keys_(keys), values_(values), size_(size), i_(0) {}

    std::optional<Event<Key>> next() override {
        if (i_ >= size_) return std::nullopt;
        Event<Key> e{keys_[i_], values_[i_], 0};
        ++i_;
        return e;
    }

private:
    const Key* keys_;
    const double* values_;
    std::size_t size_;
    std::size_t i_;
};

}} // namespace screamer::streams
#endif
