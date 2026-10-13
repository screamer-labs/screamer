#ifndef SCREAMER_STREAMS_VECTOR_SOURCE_H
#define SCREAMER_STREAMS_VECTOR_SOURCE_H

#include <cstddef>
#include "screamer/dag/checked_index.h"
#include "screamer/streams/event.h"

namespace screamer { namespace streams {

// The events of input `input` from arrays. An index below the one before it
// raises the index-order error (checked_index.h) when that event is read;
// `Checked = false` is for a caller that checked the order in a loop of its
// own over the index.
template <class Index, bool Checked = true>
class VectorSource : public Source<Index> {
public:
    VectorSource(const Index* indices, const double* values, std::size_t size,
                 std::size_t input = 0)
        : indices_(indices), values_(values), size_(size), i_(0), input_(input) {}

    std::optional<Event<Index>> next() override {
        if (i_ >= size_) return std::nullopt;
        if (Checked && i_ != 0 && indices_[i_] < indices_[i_ - 1])
            dag::throw_index_decreases(input_, i_, indices_[i_ - 1], indices_[i_]);
        if (Checked) dag::check_index_value(input_, i_, indices_[i_]);
        Event<Index> e{indices_[i_], values_[i_], 0};
        ++i_;
        return e;
    }

private:
    const Index* indices_;
    const double* values_;
    std::size_t size_;
    std::size_t i_;
    std::size_t input_;
};

}} // namespace screamer::streams
#endif
