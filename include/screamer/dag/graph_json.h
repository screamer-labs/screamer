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

#include <cerrno>
#include <cfloat>
#include <iomanip>
#include <locale>
#include <sstream>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace screamer { namespace dag { namespace json {

// The JSON text never depends on the locale, and nothing here reads global
// locale state (LC_NUMERIC, localeconv): numbers go through streams imbued
// with std::locale::classic(), whose num_put and num_get use the "C" rules
// on every thread whatever the process locale. (std::to_chars and
// std::from_chars for double would do the same, but libc++ provides
// from_chars for double only in recent releases, and to_chars needs macOS
// 13.3, newer than the wheels' deployment target.)
namespace detail {

// A JSON number's text (checked by the parser) as the nearest double. A value
// beyond the double range is infinite and one below it zero or subnormal, as
// Python's float() gives; the streams flag both (libc++ and libstdc++ set
// failbit, and libstdc++ gives DBL_MAX for an overflow), so the flag is read
// against the value.
inline double read_double(const std::string& text) {
    std::istringstream is(text);
    is.imbue(std::locale::classic());
    double d = 0.0;
    is >> d;
    if (is.fail() && (std::isinf(d) || std::fabs(d) == DBL_MAX))
        d = (text[0] == '-') ? -HUGE_VAL : HUGE_VAL;
    return d;
}

// Decimal text of an integer, without the C library's (locale-aware) printf.
inline std::string int_text(std::int64_t v) {
    char buf[24];
    char* p = buf + sizeof buf;
    const bool neg = v < 0;
    // Work in unsigned: -INT64_MIN does not fit int64.
    std::uint64_t u = neg ? 0 - static_cast<std::uint64_t>(v) : static_cast<std::uint64_t>(v);
    do { *--p = static_cast<char>('0' + u % 10); u /= 10; } while (u);
    if (neg) *--p = '-';
    return std::string(p, buf + sizeof buf);
}

// The shortest scientific text of `v` ("d.ddde±xx", the "C" rules) that
// reads back to `v`: its digits and decimal exponent.
inline void shortest_digits(double v, std::string& digits, int& exp10) {
    std::ostringstream os;
    os.imbue(std::locale::classic());
    os << std::scientific;
    std::string t;
    for (int prec = 1; prec <= 17; ++prec) {
        os.str(std::string());
        os << std::setprecision(prec - 1) << v;
        t = os.str();
        if (read_double(t) == v) break;
    }
    const std::size_t e = t.find('e');
    digits.clear();
    for (std::size_t k = 0; k < e; ++k)
        if (t[k] >= '0' && t[k] <= '9') digits += t[k];
    int x = 0;
    std::size_t k = e + 1;
    const bool neg = t[k] == '-';
    if (t[k] == '-' || t[k] == '+') ++k;
    for (; k < t.size(); ++k) x = x * 10 + (t[k] - '0');
    exp10 = neg ? -x : x;
}

}  // namespace detail

// ---- Values ----------------------------------------------------------------

struct Value {
    enum class Kind { Null, Bool, Int, Float, String, Array, Object };
    Kind kind = Kind::Null;
    bool b = false;
    std::int64_t i = 0;
    double d = 0.0;
    std::string s;          // a String; for a whole Float past int64, its digits
    bool whole = false;     // a Float written as a whole number past int64
    std::vector<Value> a;
    std::vector<std::pair<std::string, Value>> o;   // in insertion order

    static Value null() { return Value(); }
    static Value boolean(bool v) { Value x; x.kind = Kind::Bool; x.b = v; return x; }
    static Value integer(std::int64_t v) { Value x; x.kind = Kind::Int; x.i = v; return x; }
    static Value number(double v) { Value x; x.kind = Kind::Float; x.d = v; return x; }
    // A whole number past the int64 range: a Float that keeps its digits, so
    // a binding reads it as an integer (a setting then says "must be at
    // most") and the writer writes the digits back.
    static Value big_integer(double v, std::string digits) {
        Value x = number(v);
        x.whole = true;
        x.s = std::move(digits);
        return x;
    }
    static Value string(std::string v) { Value x; x.kind = Kind::String; x.s = std::move(v); return x; }
    static Value array() { Value x; x.kind = Kind::Array; return x; }
    static Value object() { Value x; x.kind = Kind::Object; return x; }

    bool is_number() const { return kind == Kind::Int || kind == Kind::Float; }
    double as_double() const { return kind == Kind::Int ? static_cast<double>(i) : d; }

    const Value* get(const std::string& key) const {
        for (const auto& kv : o)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }
    // Sets `key`, keeping its place when it is already there (as a later
    // duplicate key of a JSON object replaces the earlier one in Python).
    void set(const std::string& key, Value v) {
        for (auto& kv : o)
            if (kv.first == key) { kv.second = std::move(v); return; }
        o.emplace_back(key, std::move(v));
    }
};

// ---- Parse -------------------------------------------------------------------

namespace detail {

class Parser {
public:
    explicit Parser(const std::string& text) : t_(text) {}

    Value document() {
        ws();
        Value v = value(0);
        ws();
        if (p_ != t_.size()) fail("extra data after the JSON value");
        return v;
    }

private:
    static constexpr int kMaxDepth = 512;
    const std::string& t_;
    std::size_t p_ = 0;

    [[noreturn]] void fail(const std::string& what) const {
        std::size_t line = 1, col = 1;
        for (std::size_t k = 0; k < p_ && k < t_.size(); ++k) {
            if (t_[k] == '\n') { ++line; col = 1; } else { ++col; }
        }
        throw std::invalid_argument("invalid Pipeline JSON: " + what + " at line " +
                                    std::to_string(line) + ", column " + std::to_string(col));
    }

    void ws() {
        while (p_ < t_.size() && (t_[p_] == ' ' || t_[p_] == '\t' || t_[p_] == '\n' || t_[p_] == '\r'))
            ++p_;
    }

    bool word(const char* w) {
        std::size_t n = 0;
        while (w[n]) ++n;
        if (t_.compare(p_, n, w) != 0) return false;
        p_ += n;
        return true;
    }

    Value value(int depth) {
        if (depth > kMaxDepth) fail("values nested too deeply");
        if (p_ >= t_.size()) fail("expected a value");
        const char c = t_[p_];
        if (c == '{') return object(depth);
        if (c == '[') return array(depth);
        if (c == '"') return Value::string(string());
        if (word("true")) return Value::boolean(true);
        if (word("false")) return Value::boolean(false);
        if (word("null")) return Value::null();
        if (word("NaN")) return Value::number(std::nan(""));
        if (word("Infinity")) return Value::number(HUGE_VAL);
        if (word("-Infinity")) return Value::number(-HUGE_VAL);
        if (c == '-' || (c >= '0' && c <= '9')) return number();
        fail("expected a value");
    }

    Value object(int depth) {
        ++p_;   // {
        Value v = Value::object();
        ws();
        if (p_ < t_.size() && t_[p_] == '}') { ++p_; return v; }
        for (;;) {
            ws();
            if (p_ >= t_.size() || t_[p_] != '"') fail("expected a property name in double quotes");
            std::string key = string();
            ws();
            if (p_ >= t_.size() || t_[p_] != ':') fail("expected ':' after a property name");
            ++p_;
            ws();
            v.set(key, value(depth + 1));
            ws();
            if (p_ < t_.size() && t_[p_] == ',') { ++p_; continue; }
            if (p_ < t_.size() && t_[p_] == '}') { ++p_; return v; }
            fail("expected ',' or '}' in an object");
        }
    }

    Value array(int depth) {
        ++p_;   // [
        Value v = Value::array();
        ws();
        if (p_ < t_.size() && t_[p_] == ']') { ++p_; return v; }
        for (;;) {
            ws();
            v.a.push_back(value(depth + 1));
            ws();
            if (p_ < t_.size() && t_[p_] == ',') { ++p_; continue; }
            if (p_ < t_.size() && t_[p_] == ']') { ++p_; return v; }
            fail("expected ',' or ']' in an array");
        }
    }

    unsigned hex4() {
        if (p_ + 4 > t_.size()) fail("expected four hex digits after \\u");
        unsigned u = 0;
        for (int k = 0; k < 4; ++k) {
            const char h = t_[p_++];
            u <<= 4;
            if (h >= '0' && h <= '9') u |= static_cast<unsigned>(h - '0');
            else if (h >= 'a' && h <= 'f') u |= static_cast<unsigned>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') u |= static_cast<unsigned>(h - 'A' + 10);
            else { --p_; fail("expected four hex digits after \\u"); }
        }
        return u;
    }

    static void utf8(std::string& out, unsigned cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    std::string string() {
        ++p_;   // "
        std::string out;
        for (;;) {
            if (p_ >= t_.size()) fail("unterminated string");
            const unsigned char c = static_cast<unsigned char>(t_[p_]);
            if (c == '"') { ++p_; return out; }
            if (c < 0x20) fail("a control character in a string");
            if (c != '\\') { out += static_cast<char>(c); ++p_; continue; }
            ++p_;
            if (p_ >= t_.size()) fail("unterminated string");
            const char e = t_[p_++];
            switch (e) {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'u': {
                const std::size_t at = p_ - 2;
                unsigned cp = hex4();
                // A surrogate pair is one code point; a lone surrogate is no
                // character, and both languages refuse it.
                if (cp >= 0xD800 && cp < 0xDC00 && p_ + 1 < t_.size() && t_[p_] == '\\' &&
                    t_[p_ + 1] == 'u') {
                    const std::size_t back = p_;
                    p_ += 2;
                    const unsigned lo = hex4();
                    if (lo >= 0xDC00 && lo < 0xE000) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    else p_ = back;
                }
                if (cp >= 0xD800 && cp < 0xE000) {
                    p_ = at;
                    fail("a lone surrogate in a string");
                }
                utf8(out, cp);
                break;
            }
            default:
                --p_;
                fail("an invalid escape in a string");
            }
        }
    }

    Value number() {
        const std::size_t start = p_;
        if (t_[p_] == '-') ++p_;
        if (p_ >= t_.size() || !(t_[p_] >= '0' && t_[p_] <= '9')) fail("expected a digit");
        if (t_[p_] == '0') {
            ++p_;
        } else {
            while (p_ < t_.size() && t_[p_] >= '0' && t_[p_] <= '9') ++p_;
        }
        bool whole = true;
        if (p_ < t_.size() && t_[p_] == '.') {
            whole = false;
            ++p_;
            if (p_ >= t_.size() || !(t_[p_] >= '0' && t_[p_] <= '9')) fail("expected a digit after '.'");
            while (p_ < t_.size() && t_[p_] >= '0' && t_[p_] <= '9') ++p_;
        }
        if (p_ < t_.size() && (t_[p_] == 'e' || t_[p_] == 'E')) {
            whole = false;
            ++p_;
            if (p_ < t_.size() && (t_[p_] == '+' || t_[p_] == '-')) ++p_;
            if (p_ >= t_.size() || !(t_[p_] >= '0' && t_[p_] <= '9')) fail("expected a digit in an exponent");
            while (p_ < t_.size() && t_[p_] >= '0' && t_[p_] <= '9') ++p_;
        }
        const std::string text = t_.substr(start, p_ - start);
        if (whole) {
            // An integer that fits int64 stays an integer; a larger one is a float.
            const bool neg = text[0] == '-';
            const std::uint64_t limit = neg ? 9223372036854775808ULL : 9223372036854775807ULL;
            std::uint64_t u = 0;
            bool fits = true;
            for (std::size_t k = neg ? 1 : 0; k < text.size() && fits; ++k) {
                const std::uint64_t d = static_cast<std::uint64_t>(text[k] - '0');
                if (u > (limit - d) / 10) fits = false;
                else u = u * 10 + d;
            }
            if (fits)
                return Value::integer(neg ? static_cast<std::int64_t>(0 - u) : static_cast<std::int64_t>(u));
            return Value::big_integer(json::detail::read_double(text), text);
        }
        return Value::number(json::detail::read_double(text));
    }
};

}  // namespace detail

inline Value parse(const std::string& text) { return detail::Parser(text).document(); }

// ---- Write -------------------------------------------------------------------

// A double as Python's repr writes it: the shortest digits that read back to
// the same double, in fixed notation for a decimal point position from -3
// to 16 and in exponent notation (1e-05, 1e+16) otherwise; a whole number
// keeps ".0". JSON spells the non-finite values NaN, Infinity and -Infinity.
inline std::string float_text(double v) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
    if (v == 0.0) return std::signbit(v) ? "-0.0" : "0.0";
    const bool neg = std::signbit(v);
    std::string digits;
    int exp10 = 0;
    json::detail::shortest_digits(std::fabs(v), digits, exp10);
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    const int decpt = exp10 + 1;   // value = 0.digits * 10^decpt
    const int n = static_cast<int>(digits.size());
    std::string out = neg ? "-" : "";
    if (decpt <= -4 || decpt > 16) {
        out += digits[0];
        if (n > 1) out += "." + digits.substr(1);
        const int x = decpt - 1;
        const int ax = x < 0 ? -x : x;
        out += 'e';
        out += x < 0 ? '-' : '+';
        if (ax < 10) out += '0';
        out += json::detail::int_text(ax);
    } else if (decpt <= 0) {
        out += "0." + std::string(static_cast<std::size_t>(-decpt), '0') + digits;
    } else if (decpt >= n) {
        out += digits + std::string(static_cast<std::size_t>(decpt - n), '0') + ".0";
    } else {
        out += digits.substr(0, static_cast<std::size_t>(decpt)) + "." +
               digits.substr(static_cast<std::size_t>(decpt));
    }
    return out;
}

namespace detail {

// A string as json.dumps writes it (ensure_ascii): ASCII only, a code point
// above 0x7F as \uXXXX (a surrogate pair above 0xFFFF), lower-case hex.
inline void write_string(std::string& out, const std::string& s) {
    static const char* hex = "0123456789abcdef";
    auto u4 = [&](unsigned u) {
        out += "\\u";
        for (int k = 3; k >= 0; --k) out += hex[(u >> (4 * k)) & 0xF];
    };
    out += '"';
    for (std::size_t k = 0; k < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[k]);
        if (c < 0x80) {
            switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) u4(c);
                else out += static_cast<char>(c);
            }
            ++k;
            continue;
        }
        // A UTF-8 sequence (a malformed byte is written as its own code point).
        unsigned cp = c;
        std::size_t len = 1;
        if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
        if (k + len > s.size()) len = 1, cp = c;
        for (std::size_t j = 1; j < len; ++j) cp = (cp << 6) | (static_cast<unsigned char>(s[k + j]) & 0x3F);
        k += len;
        if (cp >= 0x10000) {
            cp -= 0x10000;
            u4(0xD800 + (cp >> 10));
            u4(0xDC00 + (cp & 0x3FF));
        } else {
            u4(cp);
        }
    }
    out += '"';
}

inline void write(std::string& out, const Value& v, int indent, int level) {
    using K = Value::Kind;
    const auto newline = [&](int lvl) {
        out += '\n';
        out.append(static_cast<std::size_t>(indent) * static_cast<std::size_t>(lvl), ' ');
    };
    switch (v.kind) {
    case K::Null: out += "null"; return;
    case K::Bool: out += v.b ? "true" : "false"; return;
    case K::Int: out += json::detail::int_text(v.i); return;
    case K::Float: out += v.whole ? v.s : float_text(v.d); return;
    case K::String: write_string(out, v.s); return;
    case K::Array:
        if (v.a.empty()) { out += "[]"; return; }
        out += '[';
        for (std::size_t k = 0; k < v.a.size(); ++k) {
            if (k) out += indent >= 0 ? "," : ", ";
            if (indent >= 0) newline(level + 1);
            write(out, v.a[k], indent, level + 1);
        }
        if (indent >= 0) newline(level);
        out += ']';
        return;
    case K::Object:
        if (v.o.empty()) { out += "{}"; return; }
        out += '{';
        for (std::size_t k = 0; k < v.o.size(); ++k) {
            if (k) out += indent >= 0 ? "," : ", ";
            if (indent >= 0) newline(level + 1);
            write_string(out, v.o[k].first);
            out += ": ";
            write(out, v.o[k].second, indent, level + 1);
        }
        if (indent >= 0) newline(level);
        out += '}';
        return;
    }
}

}  // namespace detail

// The text of `v` as Python's json.dumps(v, indent=indent) writes it; a
// negative indent is json.dumps' default (one line, ", " and ": ").
inline std::string dump(const Value& v, int indent) {
    std::string out;
    detail::write(out, v, indent, 0);
    return out;
}

}  // namespace json

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

struct OperatorSchema {
    const char* name;        // the canonical name
    const char* alias;       // the name of graphs saved before 3.0.0, or ""
    std::size_t min_in, max_in;
    std::vector<const char*> params;
};

inline const std::vector<OperatorSchema>& operator_schemas() {
    static const std::vector<OperatorSchema> s = {
        {"CombineLatest", "combine_latest", 1, SIZE_MAX, {"emit", "max_pending"}},
        {"Dropna", "dropna", 1, SIZE_MAX, {"how"}},
        {"Select", "select", 1, 1, {"columns"}},
        {"Filter", "", 2, 2, {}},
        {"Delay", "", 1, 1, {"duration"}},
        {"Resample", "resample", 1, 2,
         {"every", "freq", "count", "threshold", "clock", "agg", "origin", "label", "fill",
          "max_age", "max_fill"}},
        {"forecast_pairs", "", 2, 2, {"count", "duration", "dropna"}},
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
    std::vector<std::int64_t> ids;
    std::vector<std::string> names;          // the input nodes' names
    json::Value out_nodes = json::Value::array();
    for (std::size_t k = 0; k < nodes.a.size(); ++k) {
        const json::Value& n = nodes.a[k];
        const std::string where = "node " + std::to_string(k);
        if (n.kind != K::Object) graph_error(where + " must be an object, got " + detail::json_kind(n));
        const std::int64_t id = detail::node_id(detail::field(n, "id", where), where + ": \"id\"");
        for (std::int64_t seen : ids)
            if (seen == id) graph_error(where + ": the id " + std::to_string(id) + " is used twice");
        const json::Value& kind = detail::field(n, "kind", where);
        json::Value node = json::Value::object();
        node.set("id", json::Value::integer(id));
        if (kind.kind == K::String && kind.s == "input") {
            const json::Value& name = detail::field(n, "name", where);
            if (name.kind != K::String) graph_error(where + ": an input's \"name\" must be a string");
            for (const std::string& seen : names)
                if (seen == name.s) graph_error("the input name '" + name.s + "' is used twice");
            names.push_back(name.s);
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
                bool earlier = false;
                for (std::int64_t seen : ids) earlier = earlier || seen == src;
                if (!earlier)
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
                        for (const char* p : schema->params) known = known || kv.first == p;
                        if (!known) graph_error(who + ": unknown parameter '" + kv.first + "'");
                    }
                node.set("op", json::Value::string(schema->name));
            } else {
                node.set("cls", what);
            }
            node.set("params", params ? *params : json::Value::object());
            node.set("in", std::move(in_ids));
        } else {
            throw std::invalid_argument("unknown node kind: " + graph_text(kind));
        }
        ids.push_back(id);
        out_nodes.a.push_back(std::move(node));
    }
    // Every declared input is an input node, and the reverse.
    for (const json::Value& n : inputs.a) {
        bool found = false;
        for (const std::string& seen : names) found = found || seen == n.s;
        if (!found) graph_error("\"inputs\" names '" + n.s + "', which is not an input node");
    }
    for (const std::string& seen : names) {
        bool found = false;
        for (const json::Value& n : inputs.a) found = found || seen == n.s;
        if (!found) graph_error("the input node '" + seen + "' is not in \"inputs\"");
    }
    out.set("nodes", std::move(out_nodes));

    const json::Value& outputs = detail::field(doc, "outputs", "the description");
    if (outputs.kind != K::Array) graph_error("\"outputs\" must be an array of node ids");
    json::Value out_ids = json::Value::array();
    for (const json::Value& o : outputs.a) {
        const std::int64_t id = detail::node_id(o, "an entry of \"outputs\"");
        bool known = false;
        for (std::int64_t seen : ids) known = known || seen == id;
        if (!known) graph_error("output " + std::to_string(id) + " is not a node");
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
