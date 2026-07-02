#ifndef SCREAMER_STREAMS_COLLECTOR_SINK_H
#define SCREAMER_STREAMS_COLLECTOR_SINK_H

#include <cstddef>
#include "screamer/streams/event.h"

namespace screamer { namespace streams {

// Terminal sink: writes each emitted event to preallocated output buffers.
template <class Key>
class CollectorSink : public Sink<Key> {
public:
    CollectorSink(Key* out_keys, double* out_values)
        : ok_(out_keys), ov_(out_values), n_(0) {}

    void push(const Event<Key>& e) override {
        ok_[n_] = e.key;
        ov_[n_] = e.value;
        ++n_;
    }

    std::size_t count() const { return n_; }

private:
    Key* ok_;
    double* ov_;
    std::size_t n_;
};

// Terminal sink that keeps only values (used when the caller discards keys).
template <class Key>
class ValueCollectorSink : public Sink<Key> {
public:
    explicit ValueCollectorSink(double* out_values) : ov_(out_values), n_(0) {}
    void push(const Event<Key>& e) override { ov_[n_] = e.value; ++n_; }
    std::size_t count() const { return n_; }
private:
    double* ov_;
    std::size_t n_;
};

}} // namespace screamer::streams
#endif
