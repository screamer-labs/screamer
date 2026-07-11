#ifndef SCREAMER_DAG_FRAME_H
#define SCREAMER_DAG_FRAME_H

#include <cstddef>
#include "screamer/dag/resettable.h"

namespace screamer { namespace dag {

// One event on a graph edge. `values` points at the EMITTER's reused buffer and
// is valid only for the duration of the synchronous push() call — a consumer
// reads it immediately and does not retain the pointer. `width` is the number
// of doubles (1 for a normal functor, N for an aligned combine_latest, M for a
// multi-output functor).
template <class Index>
struct Frame {
    Index index;
    const double* values;
    std::size_t width;
};

// Receives frames. Derives from Resettable so every Sink subtype can be reset
// via a single polymorphic pointer without knowing the concrete type.
template <class Index>
struct Sink : public Resettable {
    virtual ~Sink() = default;
    virtual void push(const Frame<Index>& f) = 0;
    virtual void flush() {}
    // Reset all internal state to initial conditions. Default is a no-op so
    // stateless nodes inherit it without modification.
    virtual void reset() {}
    // Arity metadata: expected input frame width and emitted frame width.
    // Returns 0 when the width is context-dependent (passthrough nodes, terminal
    // sinks) or unknown at construction time. Operator nodes override both.
    virtual std::size_t n_in()  const { return 0; }
    virtual std::size_t n_out() const { return 0; }
};

}} // namespace screamer::dag
#endif
