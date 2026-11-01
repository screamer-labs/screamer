#ifndef SCREAMER_DAG_NODE_STATE_H
#define SCREAMER_DAG_NODE_STATE_H

// The save calls of a stream node (dag/resettable.h). A node lists what
// reset() touches in a member template state(Ar&), as an operator does
// (serial/archive.h), declares kStateVersion, and forwards the virtuals:
//
//   bool saveable() const override { return true; }
//   void write_state(json::Value& out) const override { write_node_state(*this, out); }
//   void read_state(const json::Value& in, const std::string& path) override {
//       read_node_state(*this, in, path);
//   }
//
// Its tree is {"version": N, "values": {...}}, the form of an operator's. A
// node saves its logical content, not its storage: a ring oldest first, the
// waiting events of a join in the order it releases them. A load rebuilds the
// storage from it, so saving a loaded node writes the same tree.

#include <cstddef>
#include <optional>
#include <string>
#include <vector>
#include "screamer/common/json.h"
#include "screamer/serial/archive.h"
#include "screamer/serial/errors.h"
#include "screamer/serial/saveable.h"

namespace screamer { namespace dag {

template <class N>
void write_node_state(const N& node, json::Value& out) {
    out = serial::detail::state_node(N::kStateVersion, serial::save_state(node));
}

// Loads a node's tree: a Validator pass, then a Reader pass, so a tree that
// does not fit leaves the node unchanged.
template <class N>
void read_node_state(N& node, const json::Value& in, const std::string& path) {
    bool columns = false;
    const json::Value& body = serial::detail::state_body(in, N::kStateVersion, path, columns);
    if (columns) throw serial::StateMismatch(path + ".columns", "unknown field");
    serial::load_state(node, body, path);
}

// A member of a node that is set or not (`has`), saved as the value or null.
template <class Ar, class T>
void optional_index(Ar& ar, const char* name, bool& has, T& value) {
    std::optional<T> v;
    if (has) v = value;
    ar.local(name, v);
    if constexpr (Ar::kLoading) {
        ar.set(has, v.has_value());
        ar.set(value, v.value_or(T{}));
    }
}

// The events a join's reorder buffer holds, in the order it releases them:
// their indexes, ports, and values (the port's width each, row after row).
template <class Index>
struct Waiting {
    std::vector<Index> index;
    std::vector<std::size_t> port;
    std::vector<double> values;

    template <class Ar>
    void fields(Ar& ar) {
        ar.local("pending_index", index);
        ar.local("pending_port", port, index.size());
        ar.local("pending_values", values);
    }
};

}} // namespace screamer::dag
#endif
