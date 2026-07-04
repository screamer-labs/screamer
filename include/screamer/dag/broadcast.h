#ifndef SCREAMER_DAG_BROADCAST_H
#define SCREAMER_DAG_BROADCAST_H

#include <vector>
#include "screamer/dag/frame.h"

namespace screamer { namespace dag {

// Fan-out: forwards each frame to every registered downstream sink. One job.
template <class Index>
class Broadcast : public Sink<Index> {
public:
    void add(Sink<Index>& s) { sinks_.push_back(&s); }
    void push(const Frame<Index>& f) override { for (auto* s : sinks_) s->push(f); }
    void flush() override { for (auto* s : sinks_) s->flush(); }
private:
    std::vector<Sink<Index>*> sinks_;
};

}} // namespace screamer::dag
#endif
