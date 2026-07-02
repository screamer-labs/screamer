#ifndef SCREAMER_STREAMS_FUNCTOR_NODE_H
#define SCREAMER_STREAMS_FUNCTOR_NODE_H

#include "screamer/streams/event.h"
#include "screamer/common/base.h"

namespace screamer { namespace streams {

// Push node wrapping an existing 1->1 ScreamerBase functor. Shape-preserving:
// one input event -> one output event, key and source tag passed through.
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
