#ifndef SCREAMER_COMMON_SHAPE_H
#define SCREAMER_COMMON_SHAPE_H

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace screamer { namespace shape {

// The shape contract (docs/polymorphic_api.md): every array is (T, *A, F?).
// T is time (axis 0); A are zero or more asset axes, each asset an independent
// series with its own operator state; F is an optional last axis of fields.
// This header holds the rules that read shapes, once, for the Python and the
// JavaScript bindings.

using Shape = std::vector<std::size_t>;

inline std::string to_string(const Shape& s) {
    std::ostringstream o;
    o << '(';
    for (std::size_t i = 0; i < s.size(); ++i) o << (i ? ", " : "") << s[i];
    if (s.size() == 1) o << ',';
    o << ')';
    return o.str();
}

// ---- layout strings ---------------------------------------------------------
//
// A layout names the axes of an array with T (time), A (asset) and F (field),
// optionally followed by "->" and the wanted result: "TAF->TA" drops a size-1
// field axis, "TF->TAF" inserts a size-1 asset axis, "AT->TA" reorders. Without
// "->" the result is the canonical (T, *A, F), with F inserted when missing.

struct LayoutPlan {
    std::string out;                   // result axis letters, e.g. "TAF"
    std::vector<int> source;           // per result axis: input axis, or -1 (inserted, size 1)
    std::vector<std::size_t> dropped;  // input axes removed; each has size 1
    Shape shape;                       // result shape
};

namespace detail {

inline std::invalid_argument layout_error(const std::string& layout, const std::string& what) {
    return std::invalid_argument("layout '" + layout + "': " + what);
}

inline void check_side(const std::string& side, const std::string& layout, const char* name) {
    std::size_t t = 0, f = 0;
    for (char c : side) {
        if (c == 'T') ++t;
        else if (c == 'F') ++f;
        else if (c != 'A')
            throw layout_error(layout, std::string("unknown axis letter '") + c +
                                       "'; use T (time), A (asset) and F (field)");
    }
    if (t != 1) throw layout_error(layout, std::string("the ") + name + " must name exactly one T axis");
    if (f > 1) throw layout_error(layout, std::string("the ") + name + " can name at most one F axis");
}

}  // namespace detail

inline LayoutPlan plan_layout(const std::string& raw, const Shape& in) {
    std::string layout;
    for (char c : raw) if (c != ' ') layout += c;
    std::string left = layout, right;
    const std::size_t arrow = layout.find("->");
    if (arrow != std::string::npos) {
        left = layout.substr(0, arrow);
        right = layout.substr(arrow + 2);
        if (right.find("->") != std::string::npos)
            throw detail::layout_error(raw, "use at most one '->'");
    }
    detail::check_side(left, raw, "left side");
    if (left.size() != in.size())
        throw detail::layout_error(raw, "names " + std::to_string(left.size()) +
            " axes, but the input has shape " + to_string(in));
    if (arrow == std::string::npos) {
        right = "T";
        for (char c : left) if (c == 'A') right += 'A';
        right += 'F';
    }
    detail::check_side(right, raw, "result");
    if (right[0] != 'T')
        throw detail::layout_error(raw, "the result must start with T, because time is axis 0");

    int t_in = -1, f_in = -1;
    std::vector<int> a_in;
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (left[i] == 'T') t_in = static_cast<int>(i);
        else if (left[i] == 'F') f_in = static_cast<int>(i);
        else a_in.push_back(static_cast<int>(i));
    }
    LayoutPlan p;
    p.out = right;
    std::vector<bool> used(left.size(), false);
    std::size_t a_next = 0;
    for (char c : right) {
        int src;
        if (c == 'T') src = t_in;
        else if (c == 'F') src = f_in;
        else src = a_next < a_in.size() ? a_in[a_next++] : -1;
        p.source.push_back(src);
        if (src >= 0) used[static_cast<std::size_t>(src)] = true;
        p.shape.push_back(src >= 0 ? in[static_cast<std::size_t>(src)] : 1);
    }
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (used[i]) continue;
        if (in[i] != 1)
            throw detail::layout_error(raw, std::string("drops axis ") + left[i] + " (axis " +
                std::to_string(i) + "), which has size " + std::to_string(in[i]) + " in shape " +
                to_string(in) + "; only a size-1 axis can be dropped");
        p.dropped.push_back(i);
    }
    return p;
}

// ---- operator inputs ----------------------------------------------------------

// One input field of shape (T, *A) in a C-contiguous buffer: either a whole
// argument, or one field of a packed (T, *A, N) argument.
struct View {
    Shape shape;                        // (T, *A)
    std::vector<std::size_t> strides;   // element strides, one per axis of shape
    std::size_t offset = 0;             // element offset of the first value
};

inline View contiguous_view(const Shape& shape) {
    View v;
    v.shape = shape;
    v.strides.assign(shape.size(), 1);
    for (std::size_t d = shape.size(); d-- > 1; ) v.strides[d - 1] = v.strides[d] * shape[d];
    return v;
}

// Rule 1: a trailing size-1 field axis is consumed, so (T, K, 1) reads as
// (T, K). Only an array with an asset axis has a field axis to consume:
// (T, 1) stays one asset.
inline Shape consume_unit_field(Shape s) {
    if (s.size() >= 3 && s.back() == 1) s.pop_back();
    return s;
}

// Rule 4: one packed (T, *A, n) array holds the n input fields along its last axis.
inline std::vector<View> split_fields(const Shape& packed, std::size_t n) {
    const View whole = contiguous_view(packed);
    std::vector<View> out(n);
    for (std::size_t j = 0; j < n; ++j) {
        out[j].shape.assign(packed.begin(), packed.end() - 1);
        out[j].strides.assign(whole.strides.begin(), whole.strides.end() - 1);
        out[j].offset = j;
    }
    return out;
}

// The input fields of an n_in-input call, given the argument shapes: one packed
// argument is split along its last axis; separate arguments have a trailing
// size-1 field axis consumed.
inline std::vector<View> input_views(const std::vector<Shape>& args, std::size_t n_in) {
    const std::string n = std::to_string(n_in);
    if (args.size() == 1 && n_in > 1) {
        const Shape& s = args[0];
        if (s.size() < 2 || s.back() != n_in)
            throw std::invalid_argument("This functor expects " + n + " inputs; got one array of shape " +
                to_string(s) + ". Pass a (T, ..., " + n + ") array or " + n + " separate arrays.");
        return split_fields(s, n_in);
    }
    if (args.size() != n_in)
        throw std::invalid_argument("This functor expects " + n + " inputs; got " +
                                    std::to_string(args.size()) + ".");
    std::vector<View> out;
    out.reserve(args.size());
    for (const Shape& s : args) out.push_back(contiguous_view(consume_unit_field(s)));
    return out;
}

// Rule 4: the inputs' asset shapes broadcast with NumPy rules (right-aligned,
// each axis equal or 1) and T must match. The plan lists, per input and per
// result column, where that column starts, so a broadcast input is read in
// place, never copied once per asset.
struct Plan {
    Shape shape;                              // (T, *A): the result without output fields
    std::size_t T = 0, columns = 1;           // columns = product of A
    std::vector<std::size_t> time_stride;     // per input
    std::vector<std::size_t> offset;          // [input * columns + column]
};

inline Plan broadcast(const std::vector<View>& v) {
    if (v.empty()) throw std::invalid_argument("broadcast: no inputs");
    std::string all;
    for (std::size_t i = 0; i < v.size(); ++i) all += (i ? ", " : "") + to_string(v[i].shape);
    Plan p;
    std::size_t na = 0;
    for (const View& x : v) {
        if (x.shape.empty()) throw std::invalid_argument("every input needs a time axis; got shapes " + all);
        if (x.shape[0] != v[0].shape[0])
            throw std::invalid_argument("inputs must have the same length along time (axis 0); got shapes " + all);
        na = std::max(na, x.shape.size() - 1);
    }
    p.T = v[0].shape[0];
    Shape assets(na, 1);
    for (const View& x : v) {
        const std::size_t k = x.shape.size() - 1;
        for (std::size_t j = 0; j < k; ++j) {
            const std::size_t d = x.shape[1 + j];
            std::size_t& a = assets[na - k + j];
            if (d == a || d == 1) continue;
            if (a == 1) { a = d; continue; }
            throw std::invalid_argument("input shapes " + all +
                " do not broadcast: each asset axis must be equal or 1 (time is axis 0)");
        }
    }
    p.shape.push_back(p.T);
    p.shape.insert(p.shape.end(), assets.begin(), assets.end());
    for (std::size_t a : assets) p.columns *= a;
    p.time_stride.resize(v.size());
    p.offset.assign(v.size() * p.columns, 0);
    for (std::size_t i = 0; i < v.size(); ++i) {
        const View& x = v[i];
        const std::size_t k = x.shape.size() - 1;
        p.time_stride[i] = x.strides[0];
        std::vector<std::size_t> astride(na, 0);   // stays 0 on broadcast axes
        for (std::size_t j = 0; j < k; ++j)
            if (x.shape[1 + j] != 1) astride[na - k + j] = x.strides[1 + j];
        std::vector<std::size_t> idx(na, 0);
        for (std::size_t c = 0; c < p.columns; ++c) {
            std::size_t off = x.offset;
            for (std::size_t d = 0; d < na; ++d) off += idx[d] * astride[d];
            p.offset[i * p.columns + c] = off;
            for (std::size_t d = na; d-- > 0; ) {      // odometer, last axis fastest
                if (++idx[d] < assets[d]) break;
                idx[d] = 0;
            }
        }
    }
    return p;
}

}}  // namespace screamer::shape
#endif
