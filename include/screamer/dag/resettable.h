#ifndef SCREAMER_DAG_RESETTABLE_H
#define SCREAMER_DAG_RESETTABLE_H

#include <string>
#include "screamer/common/json.h"
#include "screamer/serial/errors.h"

namespace screamer {

namespace serial { namespace detail { class Loader; } }

namespace dag {

// Minimal polymorphic reset interface. Stateful graph nodes that CompiledGraph
// must reset between runs derive from this. Sink<Index> derives from Resettable,
// so functor and resample nodes get the interface for free. Orchestrators such as
// CombineLatestNode (which are not Sinks) also derive from Resettable and
// override reset() with their own state-clearing logic.
//
// A node that a live session saves also gives its state as an operator does
// (serial/saveable.h): the tree {"version": N, "values": {...}} of everything
// reset() touches. load_into loads that tree as part of the session's load
// `ar` (serial/archive.h ar.op), so a failed load undoes it with the rest.
// dag/node_state.h implements the three save calls for a node that lists its
// members in a state(Ar&) template; the defaults say the node cannot be
// saved.
struct Resettable {
    virtual ~Resettable() = default;
    virtual void reset() {}

    virtual bool saveable() const { return false; }
    virtual void write_state(json::Value&) const {
        throw serial::NotSaveable("node", "the node has no save and load support");
    }
    virtual void load_into(const json::Value&, const std::string& path, serial::detail::Loader&) {
        throw serial::NotSaveable(path, "the node has no save and load support");
    }
};

}} // namespace screamer::dag
#endif
