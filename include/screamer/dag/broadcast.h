#ifndef SCREAMER_DAG_BROADCAST_H
#define SCREAMER_DAG_BROADCAST_H

#include <vector>
#include "screamer/dag/frame.h"

namespace screamer { namespace dag {

// Fan-out: forwards each frame to every registered downstream sink. One job.
template <class Key>
class Broadcast : public Sink<Key> {
public:
    void add(Sink<Key>& s) { sinks_.push_back(&s); }
    void push(const Frame<Key>& f) override { for (auto* s : sinks_) s->push(f); }
    void flush() override { for (auto* s : sinks_) s->flush(); }
private:
    std::vector<Sink<Key>*> sinks_;
};

}} // namespace screamer::dag
#endif
