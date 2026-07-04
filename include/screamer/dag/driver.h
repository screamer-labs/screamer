#ifndef SCREAMER_DAG_DRIVER_H
#define SCREAMER_DAG_DRIVER_H

#include <cstddef>
#include "screamer/dag/frame.h"

namespace screamer { namespace dag {

// Batch driver: replay a row-major (T, width) value buffer as T frames (row i is
// values + i*width, pointing directly into the caller's buffer), pushing each
// into the sink. No copies; no per-event allocation.
template <class Index>
void replay_batch(const Index* indices, const double* values,
                  std::size_t T, std::size_t width, Sink<Index>& sink) {
    for (std::size_t i = 0; i < T; ++i) {
        sink.push(Frame<Index>{indices[i], values + i * width, width});
    }
    sink.flush();
}

}} // namespace screamer::dag
#endif
