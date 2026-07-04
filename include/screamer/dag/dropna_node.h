#ifndef SCREAMER_DAG_DROPNA_NODE_H
#define SCREAMER_DAG_DROPNA_NODE_H

#include <cstddef>
#include "screamer/common/float_info.h"
#include "screamer/dag/frame.h"

namespace screamer { namespace dag {

// Drops events whose values are NaN. how_all=false ("any"): drop if any value
// is NaN. how_all=true ("all"): drop only if every value is NaN (an empty frame
// is never dropped). Cardinality-reducing; forwards the surviving frame pointer
// unchanged (zero per-event allocation).
template <class Key>
class DropNaNode : public Sink<Key> {
public:
    DropNaNode(bool how_all, Sink<Key>& downstream)
        : how_all_(how_all), downstream_(downstream) {}

    void push(const Frame<Key>& f) override {
        bool any_nan = false;
        bool all_nan = f.width > 0;
        for (std::size_t i = 0; i < f.width; ++i) {
            if (screamer::isnan2(f.values[i])) any_nan = true;
            else                               all_nan = false;
        }
        bool drop = how_all_ ? all_nan : any_nan;
        if (!drop) downstream_.push(f);
    }

    void flush() override { downstream_.flush(); }

private:
    bool how_all_;
    Sink<Key>& downstream_;
};

}} // namespace screamer::dag
#endif
