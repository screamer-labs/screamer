// Embind binding for JavaScript's merge (js/src/merge.ts). The merge order,
// the width rule and the index rules are include/screamer/streams/
// merge_rows.h and include/screamer/dag/checked_index.h, which Python's merge
// uses too; JavaScript only moves events in and rows out. Buffers are WASM
// heap addresses from allocF64.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "screamer/dag/checked_index.h"
#include "screamer/streams/merge_rows.h"

namespace {

using screamer::streams::MergeOrder;
using screamer::streams::MergeWidth;

const std::string kWho = "merge";

void* alloc_bytes(std::size_t n) {
    void* p = std::malloc(std::max<std::size_t>(1, n));
    if (!p) throw std::bad_alloc();
    return p;
}

// The batch merge. `spec` on the heap, as doubles: [n, then per input: data
// pointer, index pointer (0 without an index), index length, ndim, dims...].
// Without any index each input's index is its row numbers and the result
// has none. Returns { rows, index, values, sources, shape }: `index` (rows
// doubles), `values` (rows * prod(shape) doubles) and `sources` (rows
// uint32) are blocks JavaScript copies and frees; `shape` is the shape of
// one row.
emscripten::val mergeBatch(std::uintptr_t spec) {
    const double* s = reinterpret_cast<const double*>(spec);
    const std::size_t n = static_cast<std::size_t>(s[0]);
    std::vector<const double*> data(n), index(n);
    std::vector<std::size_t> rows(n);
    std::vector<std::vector<double>> numbers(n);     // row numbers of a positional input
    MergeWidth width(kWho);
    std::size_t k = 1, total = 0;
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = reinterpret_cast<const double*>(static_cast<std::uintptr_t>(s[k++]));
        index[i] = reinterpret_cast<const double*>(static_cast<std::uintptr_t>(s[k++]));
        const std::size_t len = static_cast<std::size_t>(s[k++]);
        const std::size_t nd = static_cast<std::size_t>(s[k++]);
        if (nd == 0)
            throw std::invalid_argument(kWho + ": input " + std::to_string(i) +
                                        " is one number; a stream has a time axis");
        rows[i] = static_cast<std::size_t>(s[k++]);
        std::vector<std::size_t> row_shape;
        for (std::size_t d = 1; d < nd; ++d) row_shape.push_back(static_cast<std::size_t>(s[k++]));
        if (index[i]) {
            screamer::dag::check_feed_length(kWho, static_cast<std::ptrdiff_t>(i), rows[i], len);
        } else {
            numbers[i].resize(rows[i]);
            for (std::size_t r = 0; r < rows[i]; ++r) numbers[i][r] = static_cast<double>(r);
            index[i] = numbers[i].data();
        }
        width.check(i, row_shape);
        total += rows[i];
    }
    const std::size_t w = width.width();
    std::unique_ptr<void, decltype(&std::free)> idx(alloc_bytes(total * sizeof(double)), &std::free),
        vals(alloc_bytes(total * w * sizeof(double)), &std::free),
        src(alloc_bytes(total * sizeof(std::uint32_t)), &std::free);
    screamer::streams::merge_rows<double>(index, data, rows, w, static_cast<double*>(idx.get()),
                                          static_cast<double*>(vals.get()),
                                          static_cast<std::uint32_t*>(src.get()));
    emscripten::val shape = emscripten::val::array();
    for (std::size_t d = 0; d < width.shape().size(); ++d)
        shape.set(d, static_cast<double>(width.shape()[d]));
    emscripten::val r = emscripten::val::object();
    r.set("rows", static_cast<double>(total));
    r.set("shape", shape);
    r.set("index", static_cast<double>(reinterpret_cast<std::uintptr_t>(idx.release())));
    r.set("values", static_cast<double>(reinterpret_cast<std::uintptr_t>(vals.release())));
    r.set("sources", static_cast<double>(reinterpret_cast<std::uintptr_t>(src.release())));
    return r;
}

// The iterator merge of n sources: JavaScript pulls an event from the
// source want() names and offers its index and row shape (or ends the
// source); once want() is -1, pop() names the source whose pending event
// comes next, or -1 when every source has ended. JavaScript keeps the
// events; this keeps their order and checks the index order and the width.
class MergeStream {
public:
    explicit MergeStream(unsigned n) : order_(n), events_(n, 0) {}

    int want() const { return static_cast<int>(order_.want()); }

    // The next event of source i: its index and its row shape ([] for a
    // number).
    void offerEvent(unsigned i, double index, emscripten::val shape) {
        const unsigned nd = shape["length"].as<unsigned>();
        shape_.resize(nd);
        for (unsigned d = 0; d < nd; ++d) shape_[d] = static_cast<std::size_t>(shape[d].as<double>());
        width_.check(i, shape_, static_cast<std::ptrdiff_t>(events_[i]++));
        order_.offer(i, index);
    }

    void end(unsigned i) { order_.end(i); }

    int pop() { return static_cast<int>(order_.pop()); }

private:
    MergeOrder<double> order_;
    MergeWidth width_{kWho};
    std::vector<std::size_t> events_;
    std::vector<std::size_t> shape_;
};

}  // namespace

EMSCRIPTEN_BINDINGS(screamer_merge) {
    emscripten::function("mergeBatch", &mergeBatch);
    emscripten::class_<MergeStream>("MergeStream")
        .constructor<unsigned>()
        .function("want", &MergeStream::want)
        .function("offerEvent", &MergeStream::offerEvent)
        .function("end", &MergeStream::end)
        .function("pop", &MergeStream::pop);
}
