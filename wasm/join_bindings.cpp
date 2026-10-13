// Embind binding for the direct forms of combineLatest (js/src/combine.ts).
// The layout join is include/screamer/streams/join_plan.h and the iterator
// engine include/screamer/dag/combine_stream.h, which Python's combine_latest
// uses too; JavaScript only moves events in and rows out. Buffers are WASM
// heap addresses from allocF64.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <vector>
#include <emscripten/bind.h>
#include <emscripten/em_macros.h>
#include <emscripten/val.h>
#include "screamer/dag/checked_index.h"
#include "screamer/streams/combine_latest.h"
#include "screamer/streams/merge_rows.h"
#include "screamer/streams/join_plan.h"

namespace {

using screamer::shape::Shape;
using screamer::streams::JoinPlan;

const std::string kWho = "combineLatest";

template <class V>
emscripten::val to_js(const std::vector<V>& v) {
    emscripten::val a = emscripten::val::array();
    for (std::size_t i = 0; i < v.size(); ++i) a.set(i, static_cast<double>(v[i]));
    return a;
}

// One event shape per input (time first), or null for a source with no events.
std::vector<std::optional<Shape>> read_heads(const emscripten::val& shapes) {
    std::vector<std::optional<Shape>> heads;
    const unsigned n = shapes["length"].as<unsigned>();
    for (unsigned i = 0; i < n; ++i) {
        const emscripten::val s = shapes[i];
        if (s.isNull() || s.isUndefined()) { heads.emplace_back(); continue; }
        Shape sh;
        const unsigned nd = s["length"].as<unsigned>();
        for (unsigned d = 0; d < nd; ++d) sh.push_back(static_cast<std::size_t>(s[d].as<double>()));
        heads.push_back(std::move(sh));
    }
    return heads;
}

// The layout option, checked in JavaScript: null, one string, or an array of
// strings and nulls.
std::vector<std::optional<std::string>> read_layouts(const emscripten::val& layout, std::size_t n) {
    if (layout.isNull() || layout.isUndefined()) return std::vector<std::optional<std::string>>(n);
    if (layout.isString()) return std::vector<std::optional<std::string>>(n, layout.as<std::string>());
    const unsigned given = layout["length"].as<unsigned>();
    screamer::streams::check_layout_count(given, n, kWho);
    std::vector<std::optional<std::string>> out;
    for (unsigned i = 0; i < given; ++i) {
        const emscripten::val l = layout[i];
        if (l.isNull() || l.isUndefined()) out.emplace_back();
        else out.emplace_back(l.as<std::string>());
    }
    return out;
}

// A block of doubles for JavaScript to copy and free (freeBuf).
double* alloc(std::size_t n) {
    void* p = std::malloc(std::max<std::size_t>(1, n) * sizeof(double));
    if (!p) throw std::bad_alloc();
    return static_cast<double*>(p);
}

// The batch join. `spec` on the heap, as doubles: [n, then per input: data
// pointer, index pointer (0 without an index), index length, ndim, dims...].
// Without any index the inputs are paired by position. Returns
// { rows, index, values, shape }: `index` (rows doubles) and `values` (rows *
// prod(shape) doubles) are blocks JavaScript copies and frees; `shape` is the
// shape of one row.
emscripten::val combineLatestBatch(std::uintptr_t spec, bool whenAll, emscripten::val layout) {
    const double* s = reinterpret_cast<const double*>(spec);
    const std::size_t n = static_cast<std::size_t>(s[0]);
    std::vector<const double*> data(n);
    std::vector<const double*> index(n);
    std::vector<std::size_t> lens(n);
    std::vector<std::optional<Shape>> heads(n);
    std::size_t k = 1;
    bool indexed = false;
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = reinterpret_cast<const double*>(static_cast<std::uintptr_t>(s[k++]));
        index[i] = reinterpret_cast<const double*>(static_cast<std::uintptr_t>(s[k++]));
        lens[i] = static_cast<std::size_t>(s[k++]);
        const std::size_t nd = static_cast<std::size_t>(s[k++]);
        Shape sh;
        for (std::size_t d = 0; d < nd; ++d) sh.push_back(static_cast<std::size_t>(s[k++]));
        heads[i] = std::move(sh);
        if (index[i]) indexed = true;
    }
    const JoinPlan plan = screamer::streams::plan_join(heads, read_layouts(layout, n), false, kWho);
    if (!indexed) { index.clear(); lens.clear(); }   // positional: row numbers along each T axis
    auto rows = screamer::streams::combine_latest_join<double>(plan, index, lens, data, whenAll, kWho);
    const std::size_t m = rows.index.size();
    std::unique_ptr<double, decltype(&std::free)> idx(alloc(m), &std::free), vals(alloc(m * plan.row_size()), &std::free);
    if (m) std::memcpy(idx.get(), rows.index.data(), m * sizeof(double));
    if (plan.assets.empty()) {
        if (m) std::memcpy(vals.get(), rows.values.data(), rows.values.size() * sizeof(double));
    } else {
        plan.assemble(rows.values.data(), m, vals.get());
    }
    emscripten::val r = emscripten::val::object();
    r.set("rows", static_cast<double>(m));
    r.set("shape", to_js(plan.row_shape()));
    r.set("index", static_cast<double>(reinterpret_cast<std::uintptr_t>(idx.release())));
    r.set("values", static_cast<double>(reinterpret_cast<std::uintptr_t>(vals.release())));
    return r;
}

// The iterator join, built from the first event of each source. JavaScript
// writes each event into eventBuffer(i) (width(i) values; a wrong count is
// passed without a write, for the size check to reject).
//
// Positional sources: write every source's event of a step and its count to
// sizes() (-1 for a source that has ended), then zip(): 1 with the joined row
// at rowValues(), which stays at one address, 0 when every source has ended
// (the shared PositionalZip raises when only some have).
//
// Indexed sources: offer() or end() each source's first event, then start();
// then offer() or end() the source each call asks for. start(), offer() and
// end() feed the join until it wants another event and return one number,
// rows * (n + 1) + (wanted + 1): the rows it settled (rowIndex() and
// rowValues()), and the source whose next event it wants, or -1 when the
// stream is done. One number keeps a step to one call into WASM. The join is
// the one Python's iterator combine_latest runs (CombineLatestLazy in
// bindings_streams.cpp): the sources merge in index order, ties in input
// order (MergeOrder), every event updates its input's latest values
// (CombineLatest), and events at one index collapse into one row, the last
// wins, as in the batch join. The index is a float64, as in the batch form.
//
// JavaScript frees the object when the stream ends or is collected. The calls
// made once per event (zip, offer, end, rowIndex, rowValues) are also plain exported functions
// below, which take the object's address(): an Embind method call costs more
// than the step itself.
class JoinStream {
public:
    JoinStream(emscripten::val shapes, emscripten::val layout, bool indexed, bool whenAll) {
        const auto heads = read_heads(shapes);
        plan_ = screamer::streams::plan_join(heads, read_layouts(layout, heads.size()), true, kWho);
        const std::size_t n = plan_.inputs.size();
        events_.resize(n);
        for (std::size_t i = 0; i < n; ++i) events_[i].resize(std::max<std::size_t>(1, plan_.inputs[i].width));
        sizes_.assign(n, 0);
        out_.resize(std::max<std::size_t>(1, plan_.row_size()));
        if (indexed) {
            order_ = std::make_unique<screamer::streams::MergeOrder<double>>(n);
            cl_ = std::make_unique<screamer::streams::CombineLatest>(plan_.widths(), whenAll);
            scratch_.resize(std::max<std::size_t>(1, plan_.width));
            pending_size_.assign(n, 0);
            offered_.assign(n, 0);
        } else {
            zip_ = std::make_unique<screamer::streams::PositionalZip>(plan_, kWho);
            ptrs_.resize(n);
            counts_.resize(n);
        }
    }

    bool wide() const { return plan_.wide; }
    std::size_t width(std::size_t i) const { return plan_.inputs.at(i).width; }
    std::size_t rowSize() const { return plan_.row_size(); }
    emscripten::val rowShape() const { return to_js(plan_.row_shape()); }
    std::uintptr_t eventBuffer(std::size_t i) const { return reinterpret_cast<std::uintptr_t>(events_.at(i).data()); }
    std::uintptr_t sizes() const { return reinterpret_cast<std::uintptr_t>(sizes_.data()); }

    // Positional: one step from the events and sizes JavaScript wrote.
    int zip() {
        for (std::size_t i = 0; i < ptrs_.size(); ++i) {
            ptrs_[i] = sizes_[i] < 0 ? nullptr : events_[i].data();
            counts_[i] = sizes_[i] < 0 ? 0 : static_cast<std::size_t>(sizes_[i]);
        }
        return zip_->step(ptrs_.data(), counts_.data(), out_.data()) ? 1 : 0;
    }

    // Indexed: the next event of source i, `size` values in eventBuffer(i).
    // Its size is checked when the merge reaches it, as in Python.
    double offer(std::size_t i, double index, std::size_t size, bool run) {
        // NaN has no place in the index order (the text of Python's check).
        if (index != index)
            throw std::invalid_argument("the index of input " + std::to_string(i) + " is NaN at event " +
                                        std::to_string(offered_[i]) + "; an index must be a number");
        ++offered_[i];
        order_->offer(i, index);
        pending_size_[i] = size;
        return run ? feed() : 0.0;
    }
    // Indexed: source i has no more events.
    double end(std::size_t i, bool run) {
        order_->end(i);
        return run ? feed() : 0.0;
    }
    double start() { return feed(); }

    std::uintptr_t address() { return reinterpret_cast<std::uintptr_t>(this); }
    std::uintptr_t rowIndex() const { return reinterpret_cast<std::uintptr_t>(index_.data()); }
    std::uintptr_t rowValues() const { return reinterpret_cast<std::uintptr_t>(out_.data()); }

private:
    // Steps the join until it wants an event or is done, collecting the rows.
    double feed() {
        index_.clear();
        flat_.clear();
        std::ptrdiff_t want = -1;
        while (!done_ && (want = order_->want()) < 0) {
            const std::ptrdiff_t w = order_->pop();
            if (w < 0) {                             // the end: the held row, if any
                done_ = true;
                if (buffered_) take_held();
                break;
            }
            const std::size_t i = static_cast<std::size_t>(w);
            plan_.check_event(i, pending_size_[i], kWho);
            double* dst = scratch_.data() + plan_.start[i];
            plan_.read(i, events_[i].data(), dst);
            if (!cl_->on_event(static_cast<std::uint32_t>(i), dst)) continue;
            const double at = order_->index_of(i);
            if (buffered_ && at == at_) {            // the same index: the last row wins
                row_ = cl_->latest();
                continue;
            }
            if (buffered_) take_held();
            row_ = cl_->latest();
            at_ = at;
            buffered_ = true;
        }
        if (done_) want = -1;
        const std::size_t m = index_.size();
        out_.resize(std::max<std::size_t>(1, m * plan_.row_size()));
        if (plan_.assets.empty()) std::copy(flat_.begin(), flat_.end(), out_.begin());
        else plan_.assemble(flat_.data(), m, out_.data());
        const double n = static_cast<double>(plan_.inputs.size());
        return static_cast<double>(m) * (n + 1) + static_cast<double>(want + 1);
    }

    // The held row joins the rows of this feed.
    void take_held() {
        index_.push_back(at_);
        flat_.insert(flat_.end(), row_.begin(), row_.end());
    }

    JoinPlan plan_;
    std::unique_ptr<screamer::streams::MergeOrder<double>> order_;
    std::unique_ptr<screamer::streams::CombineLatest> cl_;
    std::vector<std::size_t> pending_size_;   // each source's offered event size
    std::vector<std::size_t> offered_;        // events offered per source
    std::vector<double> row_;                 // the held row
    double at_ = 0;                           // and its index
    bool buffered_ = false;
    bool done_ = false;
    std::vector<std::vector<double>> events_;
    std::vector<double> sizes_, out_, scratch_, index_, flat_;
    std::vector<const double*> ptrs_;
    std::vector<std::size_t> counts_;
    std::unique_ptr<screamer::streams::PositionalZip> zip_;
};

}  // namespace

// The per-event steps of a JoinStream at address `s` (JoinStream::address), as
// plain exports: JavaScript calls them as Module._screamer_join_zip(s) and so
// on, with no Embind argument conversion. A C++ exception reaches JavaScript
// as it does from an Embind method.
extern "C" {
EMSCRIPTEN_KEEPALIVE int screamer_join_zip(std::uintptr_t s) {
    return reinterpret_cast<JoinStream*>(s)->zip();
}
EMSCRIPTEN_KEEPALIVE double screamer_join_offer(std::uintptr_t s, std::uint32_t i, double index,
                                                std::uint32_t size, int run) {
    return reinterpret_cast<JoinStream*>(s)->offer(i, index, size, run != 0);
}
EMSCRIPTEN_KEEPALIVE double screamer_join_end(std::uintptr_t s, std::uint32_t i, int run) {
    return reinterpret_cast<JoinStream*>(s)->end(i, run != 0);
}
EMSCRIPTEN_KEEPALIVE std::uintptr_t screamer_join_row_index(std::uintptr_t s) {
    return reinterpret_cast<JoinStream*>(s)->rowIndex();
}
EMSCRIPTEN_KEEPALIVE std::uintptr_t screamer_join_row_values(std::uintptr_t s) {
    return reinterpret_cast<JoinStream*>(s)->rowValues();
}
}  // extern "C"

EMSCRIPTEN_BINDINGS(screamer_join) {
    emscripten::function("combineLatestBatch", &combineLatestBatch);
    emscripten::class_<JoinStream>("JoinStream")
        .constructor<emscripten::val, emscripten::val, bool, bool>()
        .function("wide", &JoinStream::wide)
        .function("width", &JoinStream::width)
        .function("rowSize", &JoinStream::rowSize)
        .function("rowShape", &JoinStream::rowShape)
        .function("eventBuffer", &JoinStream::eventBuffer)
        .function("sizes", &JoinStream::sizes)
        .function("zip", &JoinStream::zip)
        .function("offer", &JoinStream::offer)
        .function("end", &JoinStream::end)
        .function("start", &JoinStream::start)
        .function("address", &JoinStream::address)
        .function("rowIndex", &JoinStream::rowIndex)
        .function("rowValues", &JoinStream::rowValues);
}
