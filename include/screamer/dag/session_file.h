#ifndef SCREAMER_DAG_SESSION_FILE_H
#define SCREAMER_DAG_SESSION_FILE_H

// The save files of a Pipeline and of a live session (serial/file.h), shared
// by the Python and JavaScript bindings. The settings of both are the
// Pipeline description, as Pipeline.to_json writes it (graph_json.h
// check_graph), with each number written in its setting's kind
// (save_settings); a Pipeline has no state, a live session has its engine's
// (CompiledGraph::write_state), which records the description's structure.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "screamer/common/json.h"
#include "screamer/dag/compiled_graph.h"
#include "screamer/dag/graph_json.h"
#include "screamer/serial/errors.h"
#include "screamer/serial/file.h"
#include "screamer/serial/op_registry.h"

namespace screamer { namespace dag {

namespace detail {

// The kind of a stream operator's setting (the settings of graph_json.h's
// operator schemas): 'i' an integer, 'f' a float, 'o' a functor reducer
// ({"functor", "params"}), 0 for any other.
inline char operator_param_kind(const std::string& op, const std::string& name) {
    if (op == "CombineLatest") return name == "max_pending" ? 'i' : 0;
    if (op == "Delay") return name == "duration" ? 'i' : 0;
    if (op == "forecast_pairs") return name == "count" || name == "duration" ? 'i' : 0;
    if (op == "Resample") {
        if (name == "threshold") return 'f';
        if (name == "agg") return 'o';
        if (name == "every" || name == "freq" || name == "count" || name == "origin" ||
            name == "max_age" || name == "max_fill")
            return 'i';
    }
    return 0;
}

// `v` written in kind `k`: a whole float as an integer for 'i', an integer
// as a float for 'f' (each number of an array too).
inline void number_kind(json::Value& v, char k) {
    using K = json::Value::Kind;
    if (k == 'f') {
        if (v.kind == K::Int) v = json::Value::number(static_cast<double>(v.i));
        else if (v.kind == K::Array) for (json::Value& e : v.a) number_kind(e, k);
    } else if (k == 'i' && v.kind == K::Float && !v.whole && std::isfinite(v.d) &&
               v.d == std::floor(v.d) && std::fabs(v.d) < 9.2e18) {
        v = json::Value::integer(static_cast<std::int64_t>(v.d));
    }
}

void functor_params(const std::string& cls, json::Value& params);

// A {"functor", "params"} setting (Segment's op, a Resample reducer).
inline void nested_functor(json::Value& v) {
    if (v.kind != json::Value::Kind::Object) return;
    const json::Value* cls = v.get("functor");
    if (!cls || cls->kind != json::Value::Kind::String) return;
    for (auto& kv : v.o)
        if (kv.first == "params") functor_params(cls->s, kv.second);
}

inline void functor_params(const std::string& cls, json::Value& params) {
    if (params.kind != json::Value::Kind::Object) return;
    for (auto& kv : params.o) {
        const char k = serial::param_kind(cls, kv.first);
        if (k == 'o') nested_functor(kv.second);
        else number_kind(kv.second, k);
    }
}

}  // namespace detail

// The settings of a save file: the description in its normal form, each
// number in its setting's kind, so Python's 3 and 3.0 and JavaScript's 3
// write the same bytes. Pipeline.to_json keeps the numbers as given.
inline json::Value save_settings(const json::Value& description) {
    json::Value d = check_graph(description);
    for (auto& kv : d.o) {
        if (kv.first == "max_pending") detail::number_kind(kv.second, 'i');
        if (kv.first != "nodes") continue;
        for (json::Value& node : kv.second.a) {
            const json::Value* kind = node.get("kind");
            const json::Value* what = node.get(kind->s == "operator" ? "op" : "cls");
            if (!what) continue;
            const std::string name = what->s;
            for (auto& p : node.o) {
                if (p.first != "params") continue;
                if (kind->s == "functor") { detail::functor_params(name, p.second); continue; }
                for (auto& s : p.second.o) {
                    const char k = detail::operator_param_kind(name, s.first);
                    if (k == 'o') detail::nested_functor(s.second);
                    else detail::number_kind(s.second, k);
                }
            }
        }
    }
    return d;
}

// Gives `g` the structure of the description in save form (`settings`): the
// graph's inputs, outputs and alignment, and per node its kind, its operator
// with a stream operator's settings, and its inputs.
inline void set_structure(CompiledGraph& g, const json::Value& settings) {
    json::Value graph = json::Value::object();
    for (const char* k : {"inputs", "outputs", "align_outputs"})
        if (const json::Value* v = settings.get(k)) graph.set(k, *v);
    std::vector<json::Value> nodes;
    for (const json::Value& n : settings.get("nodes")->a) {
        const std::size_t id = static_cast<std::size_t>(n.get("id")->i);
        if (id >= nodes.size()) nodes.resize(id + 1);
        json::Value s = json::Value::object();
        const std::string& kind = n.get("kind")->s;
        s.set("kind", *n.get("kind"));
        if (kind == "input") {
            s.set("name", *n.get("name"));
        } else if (kind == "functor") {
            s.set("cls", *n.get("cls"));
            s.set("in", *n.get("in"));
        } else {
            s.set("op", *n.get("op"));
            s.set("params", *n.get("params"));
            s.set("in", *n.get("in"));
        }
        nodes[id] = std::move(s);
    }
    g.set_structure(std::move(graph), std::move(nodes));
}

inline json::Value pipeline_node(const json::Value& description) {
    return serial::make_node("Pipeline", save_settings(description), json::Value::null());
}

inline json::Value session_node(CompiledGraph& g, const json::Value& description) {
    json::Value settings = save_settings(description);
    set_structure(g, settings);
    json::Value state;
    g.write_state(state, "LiveSession");
    return serial::make_node("LiveSession", std::move(settings), std::move(state));
}

// Loads the state of a session file (JSON text, as the bindings pass it)
// into the engine of a session built from the file's settings;
// `description` is that session's Pipeline description, whose structure the
// state must record.
inline void load_session_state(CompiledGraph& g, const std::string& text,
                               const json::Value& description) {
    json::Value tree;
    try {
        tree = ::screamer::json::parse(text, "state JSON");
    } catch (const std::invalid_argument& e) {
        throw serial::SaveError(e.what());
    }
    set_structure(g, save_settings(description));
    g.read_state(tree, "LiveSession");
}

}} // namespace screamer::dag
#endif
