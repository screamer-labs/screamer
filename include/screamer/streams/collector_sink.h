#ifndef SCREAMER_STREAMS_COLLECTOR_SINK_H
#define SCREAMER_STREAMS_COLLECTOR_SINK_H

#include <cstddef>
#include "screamer/streams/event.h"

namespace screamer { namespace streams {

// Terminal sink: writes each emitted event to preallocated output buffers.
template <class Index>
class CollectorSink : public Sink<Index> {
public:
    CollectorSink(Index* out_index, double* out_values)
        : ok_(out_index), ov_(out_values), n_(0) {}

    void push(const Event<Index>& e) override {
        ok_[n_] = e.index;
        ov_[n_] = e.value;
        ++n_;
    }

    std::size_t count() const { return n_; }

private:
    Index* ok_;
    double* ov_;
    std::size_t n_;
};

// Terminal sink that keeps only values (used when the caller discards index).
template <class Index>
class ValueCollectorSink : public Sink<Index> {
public:
    explicit ValueCollectorSink(double* out_values) : ov_(out_values), n_(0) {}
    void push(const Event<Index>& e) override { ov_[n_] = e.value; ++n_; }
    std::size_t count() const { return n_; }
private:
    double* ov_;
    std::size_t n_;
};

}} // namespace screamer::streams
#endif
