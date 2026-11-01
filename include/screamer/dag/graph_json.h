#ifndef SCREAMER_DAG_GRAPH_JSON_H
#define SCREAMER_DAG_GRAPH_JSON_H

// The saved form of a Pipeline, shared by the Python and JavaScript bindings:
// the JSON text (parse and write) and the schema of the description.
//
// The description is the one Python's Pipeline.to_dict() has always written:
//
//   { "screamer_dag": 1,
//     "inputs": ["x", "y"],                      the call signature, in order
//     "align_outputs": true, "max_pending": 1000000,
//     "nodes": [ {"id": 0, "kind": "input", "name": "x"},
//                {"id": 2, "kind": "functor", "cls": "RollingMean",
//                 "params": {"window_size": 5}, "in": [0]},
//                {"id": 3, "kind": "operator", "op": "Resample",
//                 "params": {"every": 5, "agg": "last", ...}, "in": [2]} ],
//     "outputs": [3] }
//
// A node's "in" names earlier nodes. A functor's params are its settings by
// their Python names; a Resample's functor reducer is {"functor": name,
// "params": {...}}. check_graph() validates a description and gives it in its
// normal form (canonical operator names, defaults filled in); each binding
// then builds its own nodes from it with its own constructors, which check
// the settings. The text is JSON with Python's spellings of the non-finite
// numbers (NaN, Infinity, -Infinity), which a setting such as an infinite
// position cap needs; dump() writes it as Python's json.dumps does, so both
// languages write the same text for the same description.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "screamer/common/json.h"

// The JSON tree and text live in common/json.h, shared with save files;
// dag::json names the same Value, dump and helpers, and its parse() names a
// Pipeline in its messages ("invalid Pipeline JSON: ...").
namespace screamer { namespace dag { namespace json {
using namespace ::screamer::json;
inline Value parse(const std::string& text) { return ::screamer::json::parse(text, "Pipeline JSON"); }
}}}  // namespace screamer::dag::json

namespace screamer { namespace dag {

// ---- The schema -------------------------------------------------------------

inline constexpr int kGraphSchemaVersion = 1;

namespace detail {

// A value in a message: a string in single quotes, as the names of classes
// and operators read; anything else as its JSON text.
inline std::string graph_text(const json::Value& v) {
    if (v.kind == json::Value::Kind::String) return "'" + v.s + "'";
    return json::dump(v, -1);
}

inline std::string json_kind(const json::Value& v) {
    using K = json::Value::Kind;
    switch (v.kind) {
    case K::Null: return "null";
    case K::Bool: return "a boolean";
    case K::Int: case K::Float: return "a number";
    case K::String: return "a string";
    case K::Array: return "an array";
    case K::Object: return "an object";
    }
    return "a value";
}

[[noreturn]] inline void graph_error(const std::string& what) {
    throw std::invalid_argument("Pipeline description: " + what);
}

// A setting of a stream operator and the kind a save file writes its number
// in (dag/session_file.h): 'i' an integer, 'f' a float, 'o' a functor
// reducer, 0 for any other value.
struct OperatorParam {
    const char* name;
    char kind;
};

struct OperatorSchema {
    const char* name;        // the canonical name
    const char* alias;       // the name of graphs saved before 3.0.0, or ""
    std::size_t min_in, max_in;
    std::vector<OperatorParam> params;
};

inline const std::vector<OperatorSchema>& operator_schemas() {
    static const std::vector<OperatorSchema> s = {
        {"CombineLatest", "combine_latest", 1, SIZE_MAX, {{"emit", 0}, {"max_pending", 'i'}}},
        {"Dropna", "dropna", 1, SIZE_MAX, {{"how", 0}}},
        {"Select", "select", 1, 1, {{"columns", 'i'}}},
        {"Filter", "", 2, 2, {}},
        {"Delay", "", 1, 1, {{"duration", 'i'}}},
        {"Resample", "resample", 1, 2,
         {{"every", 'i'}, {"freq", 'i'}, {"count", 'i'}, {"threshold", 'f'}, {"clock", 0},
          {"agg", 'o'}, {"origin", 'i'}, {"label", 0}, {"fill", 0}, {"max_age", 'i'},
          {"max_fill", 'i'}}},
        {"forecast_pairs", "", 2, 2,
         {{"count", 'i'}, {"duration", 'i'}, {"index", 0}, {"dropna", 0}}},
    };
    return s;
}

inline const json::Value& field(const json::Value& node, const std::string& key,
                                const std::string& where) {
    const json::Value* v = node.get(key);
    if (!v) graph_error(where + " has no \"" + key + "\"");
    return *v;
}

inline std::int64_t node_id(const json::Value& v, const std::string& where) {
    if (v.kind == json::Value::Kind::Int) return v.i;
    if (v.kind == json::Value::Kind::Float && v.d == std::floor(v.d) && std::fabs(v.d) < 9.0e15)
        return static_cast<std::int64_t>(v.d);
    graph_error(where + " must be an integer node id, got " + graph_text(v));
}

}  // namespace detail

// Checks a Pipeline description and returns it in its normal form: each
// operator under its canonical name, "params" present on every operator and
// functor node, "align_outputs" present. Raises std::invalid_argument
// (ValueError in Python, RangeError in JavaScript) naming the problem.
inline json::Value check_graph(const json::Value& doc) {
    using K = json::Value::Kind;
    using detail::graph_error;
    using detail::graph_text;
    if (doc.kind != K::Object)
        graph_error("a Pipeline description is a JSON object, got " + detail::json_kind(doc));
    const json::Value* version = doc.get("screamer_dag");
    if (!version || !version->is_number() || version->as_double() != kGraphSchemaVersion)
        throw std::invalid_argument("unsupported screamer_dag version: " +
                                    (version ? graph_text(*version) : std::string("none")) +
                                    " (expected " + std::to_string(kGraphSchemaVersion) + ")");
    json::Value out = json::Value::object();
    out.set("screamer_dag", json::Value::integer(kGraphSchemaVersion));

    const json::Value& inputs = detail::field(doc, "inputs", "the description");
    if (inputs.kind != K::Array) graph_error("\"inputs\" must be an array of input names");
    for (const json::Value& n : inputs.a)
        if (n.kind != K::String) graph_error("\"inputs\" must be an array of input names, got " + graph_text(n));
    out.set("inputs", inputs);

    const json::Value* align = doc.get("align_outputs");
    if (align && align->kind != K::Bool)
        graph_error("\"align_outputs\" must be true or false, got " + graph_text(*align));
    out.set("align_outputs", align ? *align : json::Value::boolean(true));
    if (const json::Value* mp = doc.get("max_pending")) out.set("max_pending", *mp);

    const json::Value& nodes = detail::field(doc, "nodes", "the description");
    if (nodes.kind != K::Array) graph_error("\"nodes\" must be an array");
    // The ids and input names seen so far, as hash sets: the check is linear
    // in the size of the description.
    std::unordered_set<std::int64_t> ids;
    std::unordered_set<std::string> names;   // the input nodes' names
    std::vector<const std::string*> name_order;   // the same, in node order
    json::Value out_nodes = json::Value::array();
    for (std::size_t k = 0; k < nodes.a.size(); ++k) {
        const json::Value& n = nodes.a[k];
        const std::string where = "node " + std::to_string(k);
        if (n.kind != K::Object) graph_error(where + " must be an object, got " + detail::json_kind(n));
        const std::int64_t id = detail::node_id(detail::field(n, "id", where), where + ": \"id\"");
        if (ids.count(id)) graph_error(where + ": the id " + std::to_string(id) + " is used twice");
        const json::Value& kind = detail::field(n, "kind", where);
        json::Value node = json::Value::object();
        node.set("id", json::Value::integer(id));
        if (kind.kind == K::String && kind.s == "input") {
            const json::Value& name = detail::field(n, "name", where);
            if (name.kind != K::String) graph_error(where + ": an input's \"name\" must be a string");
            if (!names.insert(name.s).second)
                graph_error("the input name '" + name.s + "' is used twice");
            name_order.push_back(&name.s);
            node.set("kind", kind);
            node.set("name", name);
        } else if (kind.kind == K::String && (kind.s == "operator" || kind.s == "functor")) {
            const bool op = kind.s == "operator";
            const json::Value& what = detail::field(n, op ? "op" : "cls", where);
            if (what.kind != K::String)
                graph_error(where + ": \"" + (op ? "op" : "cls") + "\" must be a string, got " + graph_text(what));
            const json::Value* params = n.get("params");
            if (params && params->kind != K::Object)
                graph_error(where + ": \"params\" must be an object, got " + detail::json_kind(*params));
            const json::Value& in = detail::field(n, "in", where);
            if (in.kind != K::Array) graph_error(where + ": \"in\" must be an array of node ids");
            json::Value in_ids = json::Value::array();
            for (const json::Value& i : in.a) {
                const std::int64_t src = detail::node_id(i, where + ": an entry of \"in\"");
                if (!ids.count(src))
                    graph_error(where + ": input " + std::to_string(src) + " is not an earlier node");
                in_ids.a.push_back(json::Value::integer(src));
            }
            node.set("kind", kind);
            if (op) {
                const detail::OperatorSchema* schema = nullptr;
                for (const auto& sc : detail::operator_schemas())
                    if (what.s == sc.name || (*sc.alias && what.s == sc.alias)) schema = &sc;
                if (!schema) throw std::invalid_argument("unknown DAG operator: " + graph_text(what));
                const std::string who = where + " (" + schema->name + ")";
                if (in_ids.a.size() < schema->min_in || in_ids.a.size() > schema->max_in)
                    graph_error(who + " takes " + std::to_string(schema->min_in) +
                                (schema->max_in == schema->min_in ? std::string()
                                 : schema->max_in == SIZE_MAX ? " or more"
                                 : " or " + std::to_string(schema->max_in)) +
                                " inputs, got " + std::to_string(in_ids.a.size()));
                if (params)
                    for (const auto& kv : params->o) {
                        bool known = false;
                        for (const detail::OperatorParam& p : schema->params) known = known || kv.first == p.name;
                        if (!known) graph_error(who + ": unknown parameter '" + kv.first + "'");
                    }
                node.set("op", json::Value::string(schema->name));
            } else {
                node.set("cls", what);
            }
            json::Value ps = params ? *params : json::Value::object();
            // A forecast_pairs node saved before `index` existed stamped its
            // rows at the target: it keeps that layout.
            if (op && node.get("op")->s == "forecast_pairs" && !ps.get("index")) {
                // In the place the bindings write it: before "dropna".
                auto at = ps.o.begin();
                while (at != ps.o.end() && at->first != "dropna") ++at;
                ps.o.emplace(at, "index", json::Value::string("target"));
            }
            node.set("params", std::move(ps));
            node.set("in", std::move(in_ids));
        } else {
            throw std::invalid_argument("unknown node kind: " + graph_text(kind));
        }
        ids.insert(id);
        out_nodes.a.push_back(std::move(node));
    }
    // Every declared input is an input node, and the reverse.
    std::unordered_set<std::string> declared;
    for (const json::Value& n : inputs.a) {
        if (!names.count(n.s)) graph_error("\"inputs\" names '" + n.s + "', which is not an input node");
        declared.insert(n.s);
    }
    for (const std::string* seen : name_order)
        if (!declared.count(*seen)) graph_error("the input node '" + *seen + "' is not in \"inputs\"");
    out.set("nodes", std::move(out_nodes));

    const json::Value& outputs = detail::field(doc, "outputs", "the description");
    if (outputs.kind != K::Array) graph_error("\"outputs\" must be an array of node ids");
    json::Value out_ids = json::Value::array();
    for (const json::Value& o : outputs.a) {
        const std::int64_t id = detail::node_id(o, "an entry of \"outputs\"");
        if (!ids.count(id)) graph_error("output " + std::to_string(id) + " is not a node");
        out_ids.a.push_back(json::Value::integer(id));
    }
    out.set("outputs", std::move(out_ids));
    return out;
}

// A functor node of `n_in` inputs is fed its inputs one node each, or one
// node that carries them as columns. Each binding checks the count once it
// has built the functor (the schema does not know an operator's inputs).
inline void check_functor_inputs(std::size_t node, const std::string& cls, std::size_t n_in,
                                 std::size_t got) {
    if (got == n_in || got == 1) return;
    detail::graph_error("node " + std::to_string(node) + " (" + cls + ") takes " +
                        std::to_string(n_in) + (n_in == 1 ? " input" : " inputs") +
                        (n_in > 1 ? " (or one input with " + std::to_string(n_in) + " columns)" : "") +
                        ", got " + std::to_string(got));
}

// The reducer-class error of a binding that does not know a functor's name.
[[noreturn]] inline void throw_unknown_functor(const std::string& name) {
    throw std::invalid_argument("unknown functor class: '" + name + "'");
}

}}  // namespace screamer::dag
#endif
