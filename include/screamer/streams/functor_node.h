#ifndef SCREAMER_STREAMS_FUNCTOR_NODE_H
#define SCREAMER_STREAMS_FUNCTOR_NODE_H

#include "screamer/streams/event.h"
#include "screamer/common/base.h"

namespace screamer { namespace streams {

// Push node wrapping an existing 1->1 ScreamerBase functor. Shape-preserving:
// one input event -> one output event, key and source tag passed through.

// INVARIANT: the graph path drives process_scalar() per event, bypassing any
// overridden process_array_no_stride/process_array_stride fast paths. Bit-identity
// with the existing array path therefore holds only while every functor's array
// override stays element-wise-equal to its process_scalar. A future functor whose
// array override changes rounding/associativity (e.g. SIMD reassociation) would
// diverge from the graph path here while both remain independently "correct".
template <class Key>
class FunctorNode : public Sink<Key> {
public:
    FunctorNode(ScreamerBase& fn, Sink<Key>& downstream)
        : fn_(fn), downstream_(downstream) {}

    void push(const Event<Key>& e) override {
        downstream_.push({e.key, fn_.process_scalar(e.value), e.source});
    }

    void flush() override { downstream_.flush(); }

private:
    ScreamerBase& fn_;
    Sink<Key>& downstream_;
};

}} // namespace screamer::streams
#endif
