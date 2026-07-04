#ifndef SCREAMER_STREAMS_DRIVER_H
#define SCREAMER_STREAMS_DRIVER_H

#include "screamer/streams/event.h"

namespace screamer { namespace streams {

// Batch driver: pull every event from the source and push it into the graph.
template <class Index>
void run_batch(Source<Index>& src, Sink<Index>& sink) {
    while (auto e = src.next()) {
        sink.push(*e);
    }
    sink.flush();
}

}} // namespace screamer::streams
#endif
