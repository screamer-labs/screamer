#ifndef SCREAMER_DAG_FRAME_H
#define SCREAMER_DAG_FRAME_H

#include <cstddef>

namespace screamer { namespace dag {

// One event on a graph edge. `values` points at the EMITTER's reused buffer and
// is valid only for the duration of the synchronous push() call — a consumer
// reads it immediately and does not retain the pointer. `width` is the number
// of doubles (1 for a normal functor, N for an aligned combine_latest, M for a
// multi-output functor).
template <class Index>
struct Frame {
    Index index;
    const double* values;
    std::size_t width;
};

// Receives frames. One method, one job.
template <class Index>
struct Sink {
    virtual ~Sink() = default;
    virtual void push(const Frame<Index>& f) = 0;
    virtual void flush() {}
};

}} // namespace screamer::dag
#endif
