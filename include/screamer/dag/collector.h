#ifndef SCREAMER_DAG_COLLECTOR_H
#define SCREAMER_DAG_COLLECTOR_H

#include <cassert>
#include <cstddef>
#include "screamer/dag/frame.h"

namespace screamer { namespace dag {

// Terminal sink: writes each frame's `width` values into a row-major (T, width)
// output buffer.
template <class Key>
class Collector : public Sink<Key> {
public:
    Collector(double* out, std::size_t width) : out_(out), width_(width), n_(0) {}

    void push(const Frame<Key>& f) override {
        assert(f.width == width_);
        for (std::size_t j = 0; j < f.width; ++j) out_[n_ * width_ + j] = f.values[j];
        ++n_;
    }

    std::size_t count() const { return n_; }

private:
    double* out_;
    std::size_t width_;
    std::size_t n_;
};

}} // namespace screamer::dag
#endif
