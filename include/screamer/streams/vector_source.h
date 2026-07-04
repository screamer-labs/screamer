#ifndef SCREAMER_STREAMS_VECTOR_SOURCE_H
#define SCREAMER_STREAMS_VECTOR_SOURCE_H

#include <cstddef>
#include "screamer/streams/event.h"

namespace screamer { namespace streams {

template <class Index>
class VectorSource : public Source<Index> {
public:
    VectorSource(const Index* indices, const double* values, std::size_t size)
        : indices_(indices), values_(values), size_(size), i_(0) {}

    std::optional<Event<Index>> next() override {
        if (i_ >= size_) return std::nullopt;
        Event<Index> e{indices_[i_], values_[i_], 0};
        ++i_;
        return e;
    }

private:
    const Index* indices_;
    const double* values_;
    std::size_t size_;
    std::size_t i_;
};

}} // namespace screamer::streams
#endif
