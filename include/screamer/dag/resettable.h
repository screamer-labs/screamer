#ifndef SCREAMER_DAG_RESETTABLE_H
#define SCREAMER_DAG_RESETTABLE_H

namespace screamer { namespace dag {

// Minimal polymorphic reset interface. Stateful graph nodes that CompiledGraph
// must reset between runs derive from this. Sink<Index> derives from Resettable,
// so functor and resample nodes get the interface for free. Orchestrators such as
// CombineLatestNode and MultiResampleNode (which are not Sinks) also derive from
// Resettable and override reset() with their own state-clearing logic.
struct Resettable {
    virtual ~Resettable() = default;
    virtual void reset() {}
};

}} // namespace screamer::dag
#endif
