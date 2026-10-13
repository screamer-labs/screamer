#ifndef SCREAMER_STREAMS_JOIN_PLAN_H
#define SCREAMER_STREAMS_JOIN_PLAN_H

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "screamer/common/shape.h"
#include "screamer/streams/combine_latest.h"
#include "screamer/streams/merge_source.h"
#include "screamer/streams/vector_source.h"

namespace screamer { namespace streams {

// The layout join of combine_latest, shared by the Python and JavaScript
// bindings. Each input reads as (T, *A_i, F_i): time, its asset axes and its
// fields. The joined row broadcasts the asset axes of every input (NumPy rules,
// as an operator call) and concatenates the fields in input order, so the
// result is (T, *A, F_0 + F_1 + ...).
//
// The join itself runs on flat rows: input i contributes width_i =
// prod(A_i) * F_i values per event, and a flat row holds the inputs' events
// one after another. assemble() turns flat rows into (*A, F) rows.

using shape::Shape;

// `who` names the function in messages: combine_latest in Python,
// combineLatest in JavaScript.

// The layouts of an n-input join: none, one string for every input, or one
// per input. `given` is the length of a per-input list.
inline void check_layout_count(std::size_t given, std::size_t n, const std::string& who = "combine_latest") {
    if (given != n)
        throw std::invalid_argument(who + ": pass one layout per input (" + std::to_string(n) +
                                    "), or one string for all");
}

// Positional inputs (no index) are paired by position, so they must have the
// same length.
inline void check_aligned_lengths(const std::vector<std::size_t>& lengths,
                                  const std::string& who = "combine_latest") {
    if (lengths.empty() ||
        std::any_of(lengths.begin(), lengths.end(), [&](std::size_t n) { return n != lengths.front(); }))
        throw std::invalid_argument(who +
            ": streams have no index, so they are assumed aligned - "
            "lengths must match, or provide an index to align different clocks");
}

// One input of a join.
struct JoinInput {
    std::size_t length = 0;             // T: the events in the input's data
    Shape fields;                       // (*A_i, F_i): the shape of one event
    std::size_t width = 1;              // values per event: prod(fields)
    std::size_t time_stride = 1;        // values between two events in the input's data
    std::vector<std::size_t> gather;    // per event value: its offset in the input's event
    bool in_order = true;               // gather[j] == j: an event is read as it is
};

struct JoinPlan {
    std::vector<JoinInput> inputs;
    Shape assets;                       // the broadcast asset shape A
    std::size_t columns = 1;            // prod(assets)
    std::size_t fields = 0;             // F_0 + F_1 + ...
    std::size_t width = 0;              // values in a flat row: width_0 + width_1 + ...
    bool wide = false;                  // an input has assets, or more than one field
    std::vector<std::size_t> start;     // per input: its first value in a flat row
    std::vector<std::size_t> field_start;   // per input: its first field in a joined row
    std::vector<std::size_t> offset;    // [i * columns + c]: input i's fields of asset c in a flat row

    std::vector<std::size_t> widths() const {
        std::vector<std::size_t> w;
        for (const JoinInput& in : inputs) w.push_back(in.width);
        return w;
    }

    // Values in one joined row: prod(assets) * fields, which is `width`
    // without assets.
    std::size_t row_size() const { return columns * fields; }

    // The shape of one joined row: (*A, F) with asset axes, else (width,).
    Shape row_shape() const {
        if (assets.empty()) return {width};
        Shape s = assets;
        s.push_back(fields);
        return s;
    }

    // Event t of input i, from data in the input's own layout. Returns a
    // pointer into `data` when the event is read as it is, else gathers it
    // into `scratch` (width_i values) and returns `scratch`.
    const double* event(std::size_t i, const double* data, std::size_t t, double* scratch) const {
        const JoinInput& in = inputs[i];
        const double* e = data + t * in.time_stride;
        if (in.in_order) return e;
        for (std::size_t j = 0; j < in.width; ++j) scratch[j] = e[in.gather[j]];
        return scratch;
    }

    // An event of input i holds `size` values: the size of its first event.
    void check_event(std::size_t i, std::size_t size, const std::string& who = "combine_latest") const {
        if (size != inputs[i].width)
            throw std::invalid_argument(who + ": an event of input " + std::to_string(i) + " has " +
                                        std::to_string(size) + " values, but its first event had " +
                                        std::to_string(inputs[i].width));
    }

    // One event of input i (its values in the input's own layout) as its
    // width_i values in the order of a flat row, written to `dst`.
    void read(std::size_t i, const double* event, double* dst) const {
        const JoinInput& in = inputs[i];
        if (in.in_order) std::copy(event, event + in.width, dst);
        else for (std::size_t j = 0; j < in.width; ++j) dst[j] = event[in.gather[j]];
    }

    // The same, written to its place in the flat row `row`.
    void put(std::size_t i, const double* event, double* row) const { read(i, event, row + start[i]); }

    // m flat rows (m * width values) as m joined rows (m * row_size() values):
    // each asset column holds every input's fields, broadcast inputs repeated.
    void assemble(const double* rows, std::size_t m, double* out) const {
        const std::size_t n = inputs.size();
        for (std::size_t r = 0; r < m; ++r) {
            const double* src = rows + r * width;
            double* row = out + r * columns * fields;
            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t f = inputs[i].fields.back();
                const std::size_t* off = offset.data() + i * columns;
                double* dst = row + field_start[i];
                if (f == 1) {                       // one field: a strided scatter
                    for (std::size_t c = 0; c < columns; ++c) dst[c * fields] = src[off[c]];
                } else {
                    for (std::size_t c = 0; c < columns; ++c)
                        std::copy(src + off[c], src + off[c] + f, dst + c * fields);
                }
            }
        }
    }
};

namespace detail {

// A source with no events: one row, size 1 on every axis its layout names
// (one asset, one field), or one field without a layout.
inline Shape empty_head(const std::optional<std::string>& layout) {
    if (!layout) return {1};
    std::size_t axes = 0;
    const std::string left = layout->substr(0, layout->find("->"));
    for (char c : left) if (std::isalpha(static_cast<unsigned char>(c))) ++axes;
    return Shape(axes, 1);
}

// Input i of the join: its fields, time stride and gather order, from the
// shape of its data (time first) and its layout.
inline JoinInput join_input(const Shape& s, const std::optional<std::string>& layout, bool events,
                            const std::string& who) {
    std::vector<std::size_t> st(s.size(), 1);              // C-contiguous strides of s
    for (std::size_t d = s.size(); d-- > 1; ) st[d - 1] = st[d] * s[d];
    Shape r;                                                // (T, *A, F) as read
    std::vector<std::size_t> rs;                            // its strides in the data
    if (layout) {
        if (events) {
            std::string l;
            for (char c : *layout) if (c != ' ') l += c;
            if (!l.empty() && l[0] != 'T')
                throw std::invalid_argument("layout '" + *layout + "': for a sequence of events the "
                                            "layout must start with T, the axis of the sequence");
        }
        const shape::LayoutPlan p = shape::plan_layout(*layout, s);
        r = p.shape;
        for (int a : p.source) rs.push_back(a >= 0 ? st[static_cast<std::size_t>(a)] : 0);
    } else {
        if (s.empty()) throw std::invalid_argument(who + ": every input needs a time axis");
        r = s;
        rs = st;
    }
    if (r.size() == 1) { r.push_back(1); rs.push_back(0); } // (T,) is one field
    JoinInput in;
    in.length = r[0];
    in.fields.assign(r.begin() + 1, r.end());
    in.time_stride = rs[0];
    in.width = 1;
    for (std::size_t d : in.fields) in.width *= d;
    in.gather.resize(in.width);
    std::vector<std::size_t> idx(in.fields.size(), 0);
    for (std::size_t j = 0; j < in.width; ++j) {
        std::size_t off = 0;
        for (std::size_t d = 0; d < idx.size(); ++d) off += idx[d] * rs[d + 1];
        in.gather[j] = off;
        if (off != j) in.in_order = false;
        for (std::size_t d = idx.size(); d-- > 0; ) {       // odometer, last axis fastest
            if (++idx[d] < in.fields[d]) break;
            idx[d] = 0;
        }
    }
    // An event read in order is a contiguous block: rows follow each other
    // unless time is not the outer axis.
    if (in.in_order && in.time_stride != in.width && r[0] > 1) in.in_order = false;
    if (in.in_order) in.time_stride = in.width;
    return in;
}

}  // namespace detail

// The join of inputs whose data have shapes `heads` (time first; for a stream,
// one event with a leading 1), each read by its layout. A missing head is a
// source with no events. `events`: the inputs are sequences of events, so a
// layout must start with T.
inline JoinPlan plan_join(const std::vector<std::optional<Shape>>& heads,
                          const std::vector<std::optional<std::string>>& layouts,
                          bool events = false, const std::string& who = "combine_latest") {
    const std::size_t n = heads.size();
    if (n == 0) throw std::invalid_argument(who + ": needs at least one stream");
    if (layouts.size() != n) check_layout_count(layouts.size(), n, who);
    JoinPlan p;
    for (std::size_t i = 0; i < n; ++i) {
        const Shape h = heads[i] ? *heads[i] : detail::empty_head(layouts[i]);
        p.inputs.push_back(detail::join_input(h, layouts[i], events, who));
    }
    // The asset axes broadcast as an operator's inputs do (shape.h), with the
    // same error. Each view addresses input i's fields of one asset in a flat row.
    std::vector<shape::View> views;
    for (std::size_t i = 0; i < n; ++i) {
        const JoinInput& in = p.inputs[i];
        p.start.push_back(p.width);
        p.field_start.push_back(p.fields);
        shape::View v;
        v.shape.push_back(1);
        v.shape.insert(v.shape.end(), in.fields.begin(), in.fields.end() - 1);
        v.strides.assign(v.shape.size(), 0);
        std::size_t stride = in.fields.back();
        for (std::size_t d = v.shape.size(); d-- > 1; ) { v.strides[d] = stride; stride *= v.shape[d]; }
        v.strides[0] = in.width;
        v.offset = p.width;
        views.push_back(std::move(v));
        p.width += in.width;
        p.fields += in.fields.back();
        if (in.fields.size() > 1 || in.fields.back() != 1) p.wide = true;
    }
    const shape::Plan b = shape::broadcast(views);
    p.assets.assign(b.shape.begin() + 1, b.shape.end());
    p.columns = b.columns;
    p.offset = b.offset;
    return p;
}

// The index of input i holds `len` values for `rows` rows of values: the
// shared feed-length check (checked_index.h).
inline void check_index_length(std::size_t i, std::size_t rows, std::size_t len,
                               const std::string& who = "combine_latest") {
    dag::check_feed_length(who, static_cast<std::ptrdiff_t>(i), rows, len);
}

// The positional (no-index) iterator join: each step takes one event from
// every source, so the sources must stay in step and end together.
class PositionalZip {
public:
    explicit PositionalZip(const JoinPlan& plan, std::string who = "combine_latest")
        : plan_(plan), who_(std::move(who)), row_(plan.width) {}

    // One step. events[i] points at source i's event, sizes[i] values in the
    // source's own layout, or is null when source i has ended. Returns false
    // when every source has ended, and raises when only some have; else
    // writes the joined row (plan.row_size() values) to `out`.
    bool step(const double* const* events, const std::size_t* sizes, double* out) {
        const std::size_t n = plan_.inputs.size();
        std::size_t ended = 0;
        for (std::size_t i = 0; i < n; ++i) if (!events[i]) ++ended;
        if (ended == n) return false;
        if (ended)
            throw std::invalid_argument(who_ + ": positional (no-index) sources must stay in step "
                                        "and end together; one source ran out before the others");
        for (std::size_t i = 0; i < n; ++i) {
            plan_.check_event(i, sizes[i], who_);
            plan_.put(i, events[i], row_.data());
        }
        if (plan_.assets.empty()) std::copy(row_.begin(), row_.end(), out);
        else plan_.assemble(row_.data(), 1, out);
        return true;
    }

    const JoinPlan& plan() const { return plan_; }

private:
    JoinPlan plan_;
    std::string who_;
    std::vector<double> row_;
};

// The rows of a batch combine_latest: one per distinct index, sorted.
template <class Index>
struct JoinedRows {
    std::vector<Index> index;
    std::vector<double> values;   // flat rows, plan.width values each
};

// The batch as-of join: input i has lens[i] events with sorted indexes at
// index[i] and its data at data[i], in the layout `plan` reads. Events merge by
// index (ties by input order); each event updates its input's latest values,
// and a row is due once every input has a value (when_all) or at once
// (on_any). Same-index rows coalesce to the last one, as the graph's
// CombineLatest node does.
template <class Index>
JoinedRows<Index> combine_latest_rows(const JoinPlan& plan, const std::vector<const Index*>& index,
                                      const std::vector<std::size_t>& lens,
                                      const std::vector<const double*>& data, bool when_all) {
    const std::size_t n = plan.inputs.size();
    // Each source's event value is its row position; the row's values are read
    // from the input's data.
    std::vector<std::vector<double>> positions(n);
    std::vector<std::unique_ptr<VectorSource<Index, false>>> sources;
    std::vector<Source<Index>*> children;
    std::vector<std::vector<double>> scratch(n);
    std::size_t total = 0;
    for (std::size_t i = 0; i < n; ++i) {
        positions[i].resize(lens[i]);
        // The row positions, and the index order (checked_index.h) in the same loop.
        const Index* k = index[i];
        for (std::size_t r = 0; r < lens[i]; ++r) {
            if (r != 0 && k[r] < k[r - 1]) dag::throw_index_decreases(i, r, k[r - 1], k[r]);
            positions[i][r] = static_cast<double>(r);
        }
        sources.push_back(std::make_unique<VectorSource<Index, false>>(index[i], positions[i].data(), lens[i], i));
        children.push_back(sources.back().get());
        scratch[i].resize(plan.inputs[i].width);
        total += lens[i];
    }
    JoinedRows<Index> out;
    out.index.reserve(total);
    out.values.reserve(total * plan.width);
    CombineLatest cl(plan.widths(), when_all);
    MergeSource<Index> merge(children);
    bool buffered = false;
    Index at{};
    while (auto e = merge.next()) {
        const std::size_t i = e->source;
        const double* v = plan.event(i, data[i], static_cast<std::size_t>(e->value), scratch[i].data());
        if (!cl.on_event(e->source, v)) continue;
        if (!buffered || e->index != at) {
            out.index.push_back(e->index);
            out.values.insert(out.values.end(), cl.latest().begin(), cl.latest().end());
            at = e->index;
            buffered = true;
        } else {   // the same index: the last row wins
            std::copy(cl.latest().begin(), cl.latest().end(), out.values.end() - plan.width);
        }
    }
    return out;
}

// The batch combine_latest of inputs whose data `plan` reads. `index` holds
// one sorted index per input (index_lens[i] values), or is empty for
// positional inputs: they are paired by position, so they need the same
// number of rows along the T axis their layouts name, and each gets its row
// numbers as index.
template <class Index>
JoinedRows<Index> combine_latest_join(const JoinPlan& plan, std::vector<const Index*> index,
                                      std::vector<std::size_t> index_lens,
                                      const std::vector<const double*>& data, bool when_all,
                                      const std::string& who = "combine_latest") {
    const std::size_t n = plan.inputs.size();
    std::vector<Index> rows;   // positional: one row-number vector, shared by every input
    if (index.empty()) {
        index_lens.clear();
        for (const JoinInput& in : plan.inputs) index_lens.push_back(in.length);
        check_aligned_lengths(index_lens, who);
        rows.resize(index_lens[0]);
        for (std::size_t r = 0; r < rows.size(); ++r) rows[r] = static_cast<Index>(r);
        index.assign(n, rows.data());
    }
    for (std::size_t i = 0; i < n; ++i) check_index_length(i, plan.inputs[i].length, index_lens[i], who);
    return combine_latest_rows<Index>(plan, index, index_lens, data, when_all);
}

}}  // namespace screamer::streams
#endif
