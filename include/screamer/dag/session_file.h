#ifndef SCREAMER_DAG_SESSION_FILE_H
#define SCREAMER_DAG_SESSION_FILE_H

// The save files of a Pipeline and of a live session (serial/file.h), shared
// by the Python and JavaScript bindings. The settings of both are the
// Pipeline description, as Pipeline.to_json writes it (graph_json.h
// check_graph); a Pipeline has no state, a live session has its engine's
// (CompiledGraph::write_state).

#include <string>
#include "screamer/common/json.h"
#include "screamer/dag/compiled_graph.h"
#include "screamer/dag/graph_json.h"
#include "screamer/serial/errors.h"
#include "screamer/serial/file.h"

namespace screamer { namespace dag {

inline json::Value pipeline_node(const json::Value& description) {
    return serial::make_node("Pipeline", check_graph(description), json::Value::null());
}

inline json::Value session_node(const CompiledGraph& g, const json::Value& description) {
    json::Value state;
    g.write_state(state, "LiveSession");
    return serial::make_node("LiveSession", check_graph(description), std::move(state));
}

// Loads the state of a session file (JSON text, as the bindings pass it)
// into the engine of a session built from the file's settings.
inline void load_session_state(CompiledGraph& g, const std::string& text) {
    json::Value tree;
    try {
        tree = ::screamer::json::parse(text, "state JSON");
    } catch (const std::invalid_argument& e) {
        throw serial::SaveError(e.what());
    }
    g.read_state(tree, "LiveSession");
}

}} // namespace screamer::dag
#endif
