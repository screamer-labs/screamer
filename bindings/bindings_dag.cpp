#include "screamer/common/int_arg.h"
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include "screamer/common/eval_op.h"
#include "screamer/dag/functor_node.h"
#include "screamer/dag/collector.h"
#include "screamer/dag/driver.h"
#include "screamer/dag/graph.h"
#include "screamer/dag/compiled_graph.h"
#include "screamer/dag/output_join.h"
#include "screamer/dag/checked_index.h"
#include "screamer/dag/graph_json.h"
#include "screamer/dag/forecast_pairs.h"
#include "screamer/streams/py_index.h"
#include "screamer/streams/event.h"
#include "screamer/streams/merge_source.h"
#include "screamer/streams/py_source.h"
#include "screamer/common/gc.h"
#include <algorithm>
#include <cassert>
#include <deque>
#include <tuple>
#include <cstddef>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;
using namespace screamer;

namespace {

// Allocate a NEW numpy array that OWNS `data` (a `new T[...]` C-contiguous
// buffer) via an owner capsule with a delete[] deleter. Generic over T.
template <class T>
nb::object owned_array(T* data, const std::vector<size_t>& shape) {
    nb::capsule owner(data, [](void* p) noexcept { delete[] (T*) p; });
    return nb::cast(nb::ndarray<nb::numpy, T>(
        data, shape.size(), shape.data(), owner));
}

// A NEW numpy array over the storage of `v`, which it takes without a copy:
// the vector moves into the array's owner capsule.
template <class T>
nb::object vector_array(std::vector<T>&& v, const std::vector<size_t>& shape) {
    auto* owned = new std::vector<T>(std::move(v));
    if (owned->empty()) owned->resize(1);   // a non-null data pointer for an empty array
    nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return nb::cast(nb::ndarray<nb::numpy, T>(owned->data(), shape.size(), shape.data(), owner));
}

// Typed, C-contiguous 1-D input array (nanobind converts dtype/order if needed,
// mirroring pybind's `py::array_t<T, forcecast|c_style>` behavior).
template <class T>
using In1D = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// Typed, C-contiguous input array of any rank (values may be 1-D or 2-D).
using InValues = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;

}  // namespace

// Puts `x` at position i of the new tuple `t` (which takes the reference).
static void set_item(nb::handle t, std::size_t i, nb::object x) {
    if (PyTuple_SetItem(t.ptr(), static_cast<Py_ssize_t>(i), x.release().ptr()) != 0)
        throw nb::python_error();
}

// The Python value of one output row of `w` values at `v`: a float for a
// width-1 output, a tuple of floats for a wider one.
static nb::object make_output_value(const double* v, std::size_t w) {
    if (w == 1) return nb::float_(v[0]);
    nb::object t = nb::steal(PyTuple_New(static_cast<Py_ssize_t>(w)));
    if (!t.is_valid()) throw nb::python_error();
    for (std::size_t j = 0; j < w; ++j) {
        PyObject* x = PyFloat_FromDouble(v[j]);
        if (!x) throw nb::python_error();
        set_item(t, j, nb::steal(x));
    }
    return t;
}

// ---------------------------------------------------------------------------
// LazyDriver: run a compiled graph over Python-iterator feeds, event by event,
// in C++. It merges the per-input (value, index) iterators by index
// (streams::MergeSource, ascending, same-index ties by input order), pushes each
// merged event into the CompiledGraph, drains the frames the graph emitted, and
// yields output rows. Holds a keepalive to the owning _CompiledGraph object.
//
// One output (n_out == 1): yield (value, index) rows, value scalar or tuple.
// M>1 outputs: the rows the graph's output join settles at each drain
// (CompiledGraph::drain_aligned), as (col0, ..., col_{M-1}, index); the same
// rows as a batch call.
// ---------------------------------------------------------------------------
class LazyDriver {
public:
    // `owner` names the API in error messages (Pipeline, combine_latest).
    //
    // Event shape (one output only): with `shape_input` >= 0, that input's
    // events may be float64 arrays of any rank (read in C order, as a batch
    // call reads (T, *A) values), and every output row comes back as an array
    // of shape A + `shape_tail`, where A is the first wide event's shape with
    // a trailing unit axis dropped (the batch rule of a per-column resample:
    // (T, *A) in, (T', *A, *tail) out). Scalar events keep the default rows.
    LazyDriver(nb::object cg_keepalive, dag::CompiledGraph& cg, nb::list iterators,
               std::size_t n_out, std::string owner = "Pipeline",
               std::ptrdiff_t shape_input = -1, std::vector<std::size_t> shape_tail = {})
        : cg_keepalive_(std::move(cg_keepalive)), cg_(cg), n_out_(n_out), owner_(std::move(owner)),
          shape_input_(shape_input), shape_tail_(std::move(shape_tail)) {
        unit_ = std::make_unique<TimeUnit>();
        for (nb::handle h : iterators) {
            sources_.push_back(std::make_unique<streams::PySource<std::int64_t>>(
                nb::borrow<nb::object>(h), /*positional=*/false, sources_.size(), owner_.c_str(),
                unit_.get()));
            child_ptrs_.push_back(sources_.back().get());
        }
        if (shape_input_ >= 0) {
            if (n_out_ != 1 || static_cast<std::size_t>(shape_input_) >= sources_.size())
                throw std::invalid_argument("_LazyDriver: shape_input needs one output and a valid input");
            sources_[static_cast<std::size_t>(shape_input_)]->set_flatten(true);
        } else {
            shape_known_ = true;
        }
        merge_ = std::make_unique<streams::MergeSource<std::int64_t>>(child_ptrs_);
        cg_.reset();
        cg_.set_merge_order(true);   // the merge feeds the inputs in index order
    }

    // The API name in error messages.
    const std::string& owner() const { return owner_; }

    nb::object next() {
        // A rejected event (an overflow) ends the session: raise, pull nothing.
        if (pending_.empty() && !done_) cg_.check_live();
        while (pending_.empty() && !done_) {
            if (auto e = merge_->next()) {
                const std::size_t src = static_cast<std::size_t>(e->source);
                const auto* py_src = sources_[src].get();
                if (static_cast<std::ptrdiff_t>(src) == shape_input_) {
                    if (!shape_known_) learn_shape(*py_src);
                    else check_shape(*py_src);
                }
                if (py_src->is_wide()) {
                    const auto& wv = py_src->wide_values();
                    cg_.push_event_wide(src, e->index, wv.data(), wv.size());
                } else {
                    cg_.push_event(src, e->index, e->value);
                }
                collect();
            } else {
                cg_.flush();
                collect();                              // after the flush: the tail
                done_ = true;
            }
        }
        if (pending_.empty()) throw nb::stop_iteration();
        nb::object row = std::move(pending_.front());
        pending_.pop_front();
        return row;
    }

    // Garbage-collector support (gc.h): the graph, the source iterators and the
    // buffered rows.
    int traverse(visitproc visit, void* arg) const {
        if (int r = gc_visit(cg_keepalive_, visit, arg)) return r;
        for (const auto& s : sources_)
            if (int r = s->traverse(visit, arg)) return r;
        return gc_visit_all(pending_, visit, arg);
    }

    // Drops the sources and the buffered rows; the iterator then ends. The
    // graph stays: cg_ points into it. The members are emptied before the
    // objects are released, so code a release runs sees an ended iterator.
    void clear() {
        done_ = true;
        auto pending = std::move(pending_);
        pending_.clear();
        for (auto& s : sources_) s->clear();
    }

private:
    void collect() {
        if (n_out_ == 1) {
            cg_.drain_into(drained_);
            collect_single();
        } else {
            take_joined(cg_.drain_aligned(drained_));
        }
    }

    // The output row shape from the first event of the shape input: A (the
    // event's shape, a trailing unit axis dropped) + shape_tail_. A scalar
    // event keeps the default rows (out_shape_ empty).
    void learn_shape(const streams::PySource<std::int64_t>& src) {
        shape_known_ = true;
        first_wide_ = src.is_wide();
        if (!first_wide_) return;
        first_shape_ = src.event_shape();
        out_shape_ = src.event_shape();
        if (out_shape_.size() >= 2 && out_shape_.back() == 1) out_shape_.pop_back();
        out_shape_.insert(out_shape_.end(), shape_tail_.begin(), shape_tail_.end());
        out_size_ = 1;
        for (std::size_t d : out_shape_) out_size_ *= d;
    }

    // Every event of the shape input has the first event's shape, as the rows
    // of a batch array do: a number, or an array of one shape.
    void check_shape(const streams::PySource<std::int64_t>& src) const {
        const bool wide = src.is_wide();
        if (wide == first_wide_ && (!wide || src.event_shape() == first_shape_)) return;
        auto text = [](bool w, const std::vector<std::size_t>& sh) {
            if (!w) return std::string("a number");
            std::string t = "shape (";
            for (std::size_t d = 0; d < sh.size(); ++d)
                t += (d ? ", " : "") + std::to_string(sh[d]);
            return t + (sh.size() == 1 ? ",)" : ")");
        };
        throw nb::value_error((std::string(src.owner()) + ": " +
            src.event_text(src.events_read() - 1) + " has " + text(wide, src.event_shape()) +
            ", but the first event had " + text(first_wide_, first_shape_) +
            "; every event of a resample has one shape").c_str());
    }

    // One output: buffer each drained frame directly as a (value, index) row,
    // the value an array of out_shape_ when the event shape is set.
    void collect_single() {
        const dag::OutputBuffer& b = drained_[0];
        const std::size_t w = b.width;
        const std::size_t rows = b.indices.size();
        const bool shaped = !out_shape_.empty() && out_size_ == w;
        for (std::size_t r = 0; r < rows; ++r) {
            const double* v = b.values.data() + r * w;
            nb::object value;
            if (shaped) {
                double* data = new double[w];
                std::memcpy(data, v, w * sizeof(double));
                value = owned_array(data, out_shape_);   // the array owns data
            } else {
                value = make_output_value(v, w);
            }
            pending_.push_back(nb::make_tuple(std::move(value), nb::int_(b.indices[r])));
        }
    }

    // M>1 outputs: the rows the join settled, as (col0, ..., col_{M-1}, index).
    void take_joined(const dag::AlignedOutputs& a) {
        const std::size_t rows = a.rows();
        for (std::size_t r = 0; r < rows; ++r) {
            nb::object row = nb::steal(PyTuple_New(static_cast<Py_ssize_t>(n_out_ + 1)));
            if (!row.is_valid()) throw nb::python_error();
            for (std::size_t j = 0; j < n_out_; ++j) {
                const std::size_t w = a.widths[j];
                set_item(row, j, make_output_value(a.values[j].data() + r * w, w));
            }
            set_item(row, n_out_, nb::int_(a.index[r]));
            pending_.push_back(std::move(row));
        }
    }

    nb::object cg_keepalive_;                 // keep the _CompiledGraph alive
    dag::CompiledGraph& cg_;
    std::size_t n_out_;
    std::string owner_;                       // the sources' owner name; outlives them
    std::unique_ptr<TimeUnit> unit_;          // the index kind of every feed
    std::vector<std::unique_ptr<streams::PySource<std::int64_t>>> sources_;
    std::vector<streams::Source<std::int64_t>*> child_ptrs_;
    std::unique_ptr<streams::MergeSource<std::int64_t>> merge_;
    std::deque<nb::object> pending_;
    std::vector<dag::OutputBuffer> drained_;  // reused drain storage
    bool done_ = false;
    std::ptrdiff_t shape_input_;              // the input whose event shape shapes rows; -1 none
    std::vector<std::size_t> shape_tail_;     // axes after the event shape (4 for ohlc)
    bool shape_known_ = false;
    bool first_wide_ = false;                 // the shape input's first event was an array
    std::vector<std::size_t> first_shape_;    // and its shape
    std::vector<std::size_t> out_shape_;      // the row shape; empty = default rows
    std::size_t out_size_ = 0;
};

// A feed's values are (T,) or (T, w): a higher rank would be read as (T,
// shape[1]) over the wrong stride. The rule and the wording of JavaScript's
// Pipeline.
static void check_feed_rank(const InValues& v, std::size_t input) {
    if (v.ndim() <= 2) return;
    std::string shape;
    for (std::size_t d = 0; d < v.ndim(); ++d)
        shape += (d ? "," : "") + std::to_string(v.shape(d));
    throw std::invalid_argument("Pipeline: the values of input " + std::to_string(input) +
                                " have shape [" + shape + "], expected (T,) or (T, w)");
}

// Hand-wire source -> FunctorNode(op) -> collector and run it in batch.
// `values` is (T,) [width 1] or (T, W) [width W]; returns (T, op.n_out()).
static nb::object run_functor_batch(
        EvalOp& op,
        In1D<std::int64_t> index,
        InValues values) {
    check_feed_rank(values, 0);
    std::size_t T = static_cast<std::size_t>(values.shape(0));
    dag::check_feed_length("run_functor_batch", -1, T, static_cast<std::size_t>(index.shape(0)));
    std::size_t width = (values.ndim() == 1)
        ? 1u : static_cast<std::size_t>(values.shape(1));
    if (width != op.n_in()) {
        throw std::invalid_argument(
            "an operator takes " + std::to_string(op.n_in()) +
            " column(s), but its input has " + std::to_string(width));
    }
    std::size_t out_w = op.n_out();

    // Owned here until the array takes it, so a throwing op does not leak it.
    std::unique_ptr<double[]> obuf(new double[(T * out_w) ? (T * out_w) : 1]);

    dag::Collector<std::int64_t> collector(obuf.get(), out_w);
    dag::FunctorNode<std::int64_t> node(op, collector);
    dag::replay_batch<std::int64_t>(index.data(), values.data(), T, width, node);
    return owned_array(obuf.release(), {T, out_w});
}

// Marshal a gathered index/value buffer into a Python tuple (index_1d, values_2d).
static nb::object marshal_gather(const std::vector<std::int64_t>& out_k,
                                 const std::vector<double>& out_v,
                                 std::size_t width) {
    std::size_t m = out_k.size();
    std::int64_t* rk = new std::int64_t[m ? m : 1];
    if (m) std::memcpy(rk, out_k.data(), m * sizeof(std::int64_t));
    double* rv = new double[(m * width) ? (m * width) : 1];
    if (m) std::memcpy(rv, out_v.data(), m * width * sizeof(double));
    return nb::make_tuple(owned_array(rk, {m}), owned_array(rv, {m, width}));
}

// Marshal a vector of OutputBuffers into a Python list of (index_1d, values_2d).
// Shared by _CompiledGraph.run_batch and _CompiledGraph.drain.
static nb::list marshal_output_buffers(const std::vector<dag::OutputBuffer>& outs) {
    nb::list result;
    for (const auto& o : outs)
        result.append(marshal_gather(o.indices, o.values, o.width));
    return result;
}

// Marshal aligned outputs into (index_1d, [values_2d, ...]): one index array,
// and each output's (rows, width) block. The arrays take the storage, no copy.
static nb::object marshal_aligned(dag::AlignedOutputs&& a) {
    const std::size_t rows = a.rows();
    nb::object index = vector_array(std::move(a.index), {rows});
    nb::list values;
    for (std::size_t j = 0; j < a.values.size(); ++j)
        values.append(vector_array(std::move(a.values[j]), {rows, a.widths[j]}));
    return nb::make_tuple(index, values);
}

// Converts n items of type T at `b` (any alignment) into `out`. Returns false
// when the item size is not T's.
template <class T>
static bool load_items(const void* b, std::size_t n, Py_ssize_t itemsize,
                       std::vector<double>& out) {
    if (itemsize != static_cast<Py_ssize_t>(sizeof(T))) return false;
    const char* c = static_cast<const char*>(b);
    out.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        T v;
        std::memcpy(&v, c + i * sizeof(T), sizeof(T));
        out[i] = static_cast<double>(v);
    }
    return true;
}

// A batch index array as int64, checked element by element in the cast loop
// (checked_index.h). An int64 array is used in place. `owned` holds a
// converted copy; `keep` holds the array the pointer reads.
using AnyIndex = nb::ndarray<nb::ro, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T, class F>
static void cast_index(const AnyIndex& a, std::vector<std::int64_t>& out, F check) {
    const T* src = static_cast<const T*>(a.data());
    const std::size_t n = a.shape(0);
    out.resize(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = check(src[i]);
}

// `unit` holds the index kind of the whole call (py_index.h TimeUnit): every
// non-empty index array of the call must be numbers, or times of one unit.
// `len` receives the number of index values.
static const std::int64_t* index_array(nb::handle obj, nb::object& keep,
                                       std::vector<std::int64_t>& owned, TimeUnit& unit,
                                       std::size_t& len) {
    In1D<std::int64_t> i64;
    if (nb::try_cast(obj, i64, /*convert=*/false)) {
        if (i64.shape(0)) note_number(&unit);
        keep = nb::borrow(obj);
        len = i64.shape(0);
        return i64.data();
    }
    AnyIndex a;
    if (!nb::try_cast(obj, a, /*convert=*/false)) {
        if (nb::hasattr(obj, "ndim") && nb::cast<int>(obj.attr("ndim")) != 1)
            throw nb::value_error("an index is a 1-D array of integers");
        const std::string kind = nb::hasattr(obj, "dtype")
            ? nb::cast<std::string>(obj.attr("dtype").attr("kind")) : std::string();
        // A datetime64 or timedelta64 array: its int64 ticks, in place. NaT
        // raises, as it does for one index (py_index).
        if (kind == "M" || kind == "m") {
            nb::object dtype = obj.attr("dtype");
            nb::object arr = nb::borrow(obj);
            if (!nb::cast<bool>(dtype.attr("isnative")))   // ticks in native byte order
                arr = arr.attr("astype")(dtype.attr("newbyteorder")("="));
            if (nb::len(arr)) {
                nb::tuple du = nb::cast<nb::tuple>(
                    nb::module_::import_("numpy").attr("datetime_data")(dtype));
                note_time(&unit, kind == "M" ? TimeUnit::kDatetime : TimeUnit::kTimedelta,
                          time_unit_base(nb::cast<std::string>(du[0])), nb::cast<int>(du[1]));
            }
            nb::object ticks = arr.attr("view")("int64");
            if (!nb::try_cast(ticks, i64, /*convert=*/false))
                throw nb::value_error("an index is a 1-D array of integers");
            const std::int64_t* d = i64.data();
            const std::size_t n = i64.shape(0);
            bool nat = false;
            for (std::size_t i = 0; i < n; ++i) nat |= d[i] == INT64_MIN;
            if (nat) dag::throw_index_nat();
            keep = std::move(ticks);
            len = n;
            return d;
        }
        // An object array (Python ints, a pandas object Index): each element
        // follows the rule of a live push.
        if (kind == "O") {
            owned.clear();
            owned.reserve(nb::len(obj));
            for (nb::handle e : obj) owned.push_back(py_index(e, &unit));
            keep = nb::borrow(obj);
            len = owned.size();
            return owned.data();
        }
        dag::throw_index_kind(nb::hasattr(obj, "dtype")
            ? "an array of " + nb::cast<std::string>(obj.attr("dtype").attr("name"))
            : nb::cast<std::string>(nb::type_name(obj.type())));
    }
    if (a.shape(0)) note_number(&unit);
    len = a.shape(0);
    using C = nb::dlpack::dtype_code;
    const nb::dlpack::dtype dt = a.dtype();
    auto widen = [](auto v) { return static_cast<std::int64_t>(v); };
    auto as_double = [](auto v) { return dag::checked_index(static_cast<double>(v)); };
    if (dt.code == static_cast<std::uint8_t>(C::Int)) {
        if (dt.bits == 8)  { cast_index<std::int8_t>(a, owned, widen); return owned.data(); }
        if (dt.bits == 16) { cast_index<std::int16_t>(a, owned, widen); return owned.data(); }
        if (dt.bits == 32) { cast_index<std::int32_t>(a, owned, widen); return owned.data(); }
    } else if (dt.code == static_cast<std::uint8_t>(C::UInt)) {
        if (dt.bits == 8)  { cast_index<std::uint8_t>(a, owned, widen); return owned.data(); }
        if (dt.bits == 16) { cast_index<std::uint16_t>(a, owned, widen); return owned.data(); }
        if (dt.bits == 32) { cast_index<std::uint32_t>(a, owned, widen); return owned.data(); }
        if (dt.bits == 64) {
            cast_index<std::uint64_t>(a, owned, [](std::uint64_t v) { return dag::checked_index_u64(v); });
            return owned.data();
        }
    } else if (dt.code == static_cast<std::uint8_t>(C::Float)) {
        if (dt.bits == 64) { cast_index<double>(a, owned, as_double); return owned.data(); }
        if (dt.bits == 32) { cast_index<float>(a, owned, as_double); return owned.data(); }
        // float16 or a long double: read through float64.
        nb::object f64 = obj.attr("astype")("float64");
        AnyIndex b = nb::cast<AnyIndex>(f64, false);
        cast_index<double>(b, owned, as_double);
        return owned.data();
    } else if (dt.code == static_cast<std::uint8_t>(C::Bool)) {
        dag::throw_index_kind("an array of bool");
    } else if (dt.code == static_cast<std::uint8_t>(C::Complex)) {
        dag::throw_index_kind("an array of complex");
    }
    dag::throw_index_kind("an array of " + nb::cast<std::string>(obj.attr("dtype").attr("name")));
}

// Helper: marshal a list of (index, values) feed tuples into raw C++ spans.
// Fills ks/vs (keep-alive arrays), kp/vp (raw pointers), lens (lengths).
// Also fills widths: 1 for a 1-D values array, W for a 2-D (N, W) values array.
// An index array goes through the engine's index rule (index_array).
static void marshal_feeds(
        nb::list feeds,
        std::vector<nb::object>& ks,
        std::vector<std::vector<std::int64_t>>& owned,
        std::vector<InValues>& vs,
        std::vector<const std::int64_t*>& kp,
        std::vector<const double*>& vp,
        std::vector<std::size_t>& lens,
        std::vector<std::size_t>& widths,
        const std::string& who = "Pipeline") {
    const std::size_t n = nb::len(feeds);
    TimeUnit unit;                      // one index kind across every input
    ks.resize(n);
    owned.resize(n);
    std::size_t f = 0;
    for (auto item : feeds) {
        nb::tuple t = nb::cast<nb::tuple>(item);
        std::size_t len = 0;
        kp.push_back(index_array(t[0], ks[f], owned[f], unit, len));
        ++f;
        vs.push_back(nb::cast<InValues>(t[1]));
        check_feed_rank(vs.back(), f - 1);
        // One index value per row, checked before the engine reads either.
        dag::check_feed_length(who, static_cast<std::ptrdiff_t>(f - 1),
                               static_cast<std::size_t>(vs.back().shape(0)), len);
        vp.push_back(vs.back().data());
        lens.push_back(static_cast<std::size_t>(vs.back().shape(0)));
        // Detect column width: 1 for 1-D arrays, shape[1] for 2-D arrays.
        widths.push_back(vs.back().ndim() >= 2
            ? static_cast<std::size_t>(vs.back().shape(1)) : 1u);
    }
}

// A count setting (max_pending, max_fill) from Python under the engine's rule
// (max_pending.h checked_count_setting): an integer from 1 to 2**53. A bool or
// another kind is a TypeError.
static double count_setting(nb::handle x, const std::string& name) {
    PyObject* p = x.ptr();
    const char* b = screamer::bool_kind(p);
    const std::string type = b ? std::string(b) : nb::cast<std::string>(nb::type_name(x.type()));
    if (b || x.is_none() || PyUnicode_Check(p) || PyBytes_Check(p) ||
        type.find("complex") != std::string::npos)
        dag::throw_max_pending_kind(type, name);
    if (PyFloat_Check(p)) return dag::checked_count_setting(PyFloat_AsDouble(p), name);
    if (PyObject* ix = PyNumber_Index(p)) {         // int, NumPy integer
        int overflow = 0;
        const long long v = PyLong_AsLongLongAndOverflow(ix, &overflow);
        Py_DECREF(ix);
        if (overflow || v > 9007199254740992LL)
            throw nb::value_error((name + " must be at least 1 and at most 2**53, got " +
                                   nb::cast<std::string>(nb::str(x))).c_str());
        if (v == -1 && PyErr_Occurred()) throw nb::python_error();
        return dag::checked_count_setting(static_cast<double>(v), name);
    }
    PyErr_Clear();
    const double d = PyFloat_AsDouble(p);            // NumPy floating
    if (d == -1.0 && PyErr_Occurred()) {
        PyErr_Clear();
        dag::throw_max_pending_kind(type, name);
    }
    return dag::checked_count_setting(d, name);
}

// ---- The saved form of a Pipeline (graph_json.h) ---------------------------
// Python objects to and from json::Value, as the json module reads them: a
// dict with str keys, a list or tuple, a bool, an int, a float, a str, None.
static dag::json::Value to_json_value(nb::handle x, int depth = 0) {
    using V = dag::json::Value;
    if (depth > 512) throw nb::value_error("Pipeline description: values nested too deeply");
    PyObject* p = x.ptr();
    if (x.is_none()) return V::null();
    if (PyBool_Check(p)) return V::boolean(p == Py_True);
    if (PyLong_Check(p)) {
        int overflow = 0;
        const long long v = PyLong_AsLongLongAndOverflow(p, &overflow);
        if (overflow) {   // past int64: the digits, as json.dumps writes them
            double d = PyLong_AsDouble(p);
            if (d == -1.0 && PyErr_Occurred()) {
                PyErr_Clear();
                d = overflow > 0 ? HUGE_VAL : -HUGE_VAL;
            }
            return V::big_integer(d, nb::cast<std::string>(nb::str(x)));
        }
        return V::integer(static_cast<std::int64_t>(v));
    }
    if (PyFloat_Check(p)) return V::number(PyFloat_AsDouble(p));
    if (PyUnicode_Check(p)) {
        Py_ssize_t n = 0;
        const char* u = PyUnicode_AsUTF8AndSize(p, &n);
        if (!u) {   // a lone surrogate has no UTF-8 form
            PyErr_Clear();
            throw nb::value_error("Pipeline description: a lone surrogate in a string");
        }
        return V::string(std::string(u, static_cast<std::size_t>(n)));
    }
    if (PyDict_Check(p)) {
        V o = V::object();
        for (auto kv : nb::borrow<nb::dict>(x)) {
            if (!PyUnicode_Check(kv.first.ptr()))
                throw nb::type_error(("keys must be str, not " +
                    nb::cast<std::string>(nb::type_name(kv.first.type()))).c_str());
            // A dict's keys are unique: append (set() would scan every key).
            o.o.emplace_back(nb::cast<std::string>(kv.first), to_json_value(kv.second, depth + 1));
        }
        return o;
    }
    if (PyList_Check(p) || PyTuple_Check(p)) {
        V a = V::array();
        for (nb::handle e : x) a.a.push_back(to_json_value(e, depth + 1));
        return a;
    }
    // A NumPy scalar or array (a setting given as np.int64(20) or
    // np.array([0, 1])): its Python value, as to_dict writes it.
    if (nb::hasattr(x, "tolist") && nb::hasattr(x, "dtype"))
        return to_json_value(x.attr("tolist")(), depth + 1);
    throw nb::type_error(("Object of type " + nb::cast<std::string>(nb::type_name(x.type())) +
                          " is not JSON serializable").c_str());
}

static nb::object from_json_value(const dag::json::Value& v) {
    using K = dag::json::Value::Kind;
    switch (v.kind) {
    case K::Null: return nb::none();
    case K::Bool: return nb::bool_(v.b);
    case K::Int: return nb::int_(v.i);
    case K::Float:
        if (v.whole) {
            PyObject* n = PyLong_FromString(v.s.c_str(), nullptr, 10);
            if (!n) throw nb::python_error();
            return nb::steal(n);
        }
        return nb::float_(v.d);
    case K::String: return nb::str(v.s.c_str(), v.s.size());
    case K::Array: {
        nb::list l;
        for (const auto& e : v.a) l.append(from_json_value(e));
        return std::move(l);
    }
    case K::Object: {
        nb::dict d;
        for (const auto& kv : v.o) d[nb::str(kv.first.c_str(), kv.first.size())] = from_json_value(kv.second);
        return std::move(d);
    }
    }
    return nb::none();
}

void init_bindings_dag(nb::module_& m) {
    // A Pipeline description checked against the shared schema, in its
    // normal form (dict in, dict out); from JSON text; and to JSON text as
    // json.dumps writes it (indent None: one line).
    m.def("_graph_check", [](nb::handle d) {
        return from_json_value(dag::check_graph(to_json_value(d)));
    }, "description"_a);
    m.def("_graph_from_json", [](nb::handle text) {
        Py_ssize_t n = 0;
        const char* u = PyUnicode_Check(text.ptr()) ? PyUnicode_AsUTF8AndSize(text.ptr(), &n) : nullptr;
        if (!u) {
            PyErr_Clear();
            if (!PyUnicode_Check(text.ptr()))
                throw nb::type_error("Pipeline.from_json takes the JSON text (a str)");
            throw nb::value_error("invalid Pipeline JSON: a lone surrogate in a string");
        }
        return from_json_value(dag::check_graph(dag::json::parse(std::string(u, static_cast<std::size_t>(n)))));
    }, "text"_a);
    m.def("_graph_to_json", [](nb::handle d, nb::handle indent) {
        const int ind = indent.is_none() ? -1 : nb::cast<int>(indent);
        return dag::json::dump(dag::check_graph(to_json_value(d)), ind);
    }, "description"_a, "indent"_a.none());
    m.def("_unknown_functor", [](const std::string& name) { dag::throw_unknown_functor(name); },
          "name"_a);
    m.def("_check_functor_inputs", &dag::check_functor_inputs, "node"_a, "cls"_a, "n_in"_a, "got"_a);
    // The engine's index rule raises TypeError in Python.
    nb::register_exception_translator(
        [](const std::exception_ptr& p, void*) {
            try { std::rethrow_exception(p); }
            catch (const dag::IndexTypeError& e) { PyErr_SetString(PyExc_TypeError, e.what()); }
            catch (const dag::SettingTypeError& e) { PyErr_SetString(PyExc_TypeError, e.what()); }
        }, nullptr);
    m.def("_checked_index", [](double d) { return dag::checked_index(d); }, "index"_a);
    // The feed-length rule (checked_index.h): one index value per row.
    m.def("_check_feed_length", &dag::check_feed_length, "who"_a, "input"_a, "rows"_a, "len"_a);
    m.attr("_DEFAULT_MAX_PENDING") = dag::kDefaultMaxPending;
    // A Pipeline's max_pending under the engine's rule (max_pending.h): an
    // integer from 1 to 2**53. A bool or another kind is a TypeError.
    m.def("_max_pending", [](nb::handle x, const std::string& who) -> std::size_t {
        return dag::checked_max_pending(count_setting(x, who), who);
    }, "max_pending"_a.none(), "who"_a = "Pipeline: max_pending");
    // A fill Resample's max_fill, under the same rule.
    m.def("_max_fill", [](nb::handle x, const std::string& who) -> std::int64_t {
        return dag::checked_max_fill(count_setting(x, who), who);
    }, "max_fill"_a.none(), "who"_a = "resample: max_fill");
    m.attr("_DEFAULT_MAX_FILL") = dag::kDefaultMaxFill;
    // The settings rule of a Resample (dag/resample_check.h), shared with the
    // JavaScript binding. None is a setting not given.
    m.def("_resample_check", [](bool has_freq, nb::handle every, nb::handle count,
                                nb::handle threshold, bool clock, bool agg_functor,
                                const std::string& agg, const std::string& label,
                                const std::string& fill, nb::handle max_age,
                                bool max_age_whole, nb::handle origin) {
        dag::ResampleSettings s;
        // An int setting past the double range, by name: its text replaces
        // the "inf" the shared rule would print for it.
        std::vector<std::pair<std::string, std::string>> huge;
        // A setting as the rule reads it: an int exactly (an int64 near 2**63
        // stays inside the range, though the nearest double is 2**63), a float
        // as float() reads it.
        auto num = [&](const char* name, nb::handle h, bool& has, double& v) {
            has = !h.is_none();
            if (!has) return;
            PyObject* p = h.ptr();
            if (PyLong_Check(p)) {
                int overflow = 0;
                const long long x = PyLong_AsLongLongAndOverflow(p, &overflow);
                if (overflow == 0) {
                    if (x == -1 && PyErr_Occurred()) throw nb::python_error();
                    v = static_cast<double>(x);
                    if (v >= 9223372036854775808.0) v = 9223372036854774784.0;
                    return;
                }
                v = PyLong_AsDouble(p);
                if (v == -1.0 && PyErr_Occurred()) {          // beyond a double
                    PyErr_Clear();
                    v = overflow > 0 ? std::numeric_limits<double>::infinity()
                                     : -std::numeric_limits<double>::infinity();
                    std::string text = nb::cast<std::string>(nb::str(h));
                    std::size_t digits = text.size() - (text[0] == '-' ? 1 : 0);
                    huge.emplace_back(name, "an int of " + std::to_string(digits) + " digits");
                }
                return;
            }
            nb::object f = nb::steal(PyNumber_Float(p));
            if (!f.is_valid()) throw nb::python_error();
            v = PyFloat_AsDouble(f.ptr());
        };
        s.has_freq = has_freq;
        num("every", every, s.has_every, s.every);
        num("count", count, s.has_count, s.count);
        num("threshold", threshold, s.has_threshold, s.threshold);
        num("origin", origin, s.has_origin, s.origin);
        s.clock = clock;
        s.agg_functor = agg_functor;
        s.agg = agg; s.label = label; s.fill = fill;
        s.max_age_whole = max_age_whole;
        // max_age of the wrong kind is reported by the rule before it is read.
        s.has_max_age = !max_age.is_none();
        if (s.has_max_age && max_age_whole) {
            bool has = false;
            num("max_age", max_age, has, s.max_age);
        }
        try {
            dag::check_resample_settings(s);
        } catch (const std::invalid_argument& e) {
            if (dynamic_cast<const dag::SettingTypeError*>(&e)) throw;
            const std::string msg = e.what();
            for (const auto& [name, text] : huge) {
                const std::string head = "resample: " + name + " must be within the int64 range, got ";
                if (msg.compare(0, head.size(), head) == 0)
                    throw std::invalid_argument(head + text);
            }
            throw;
        }
    }, "has_freq"_a, "every"_a.none(), "count"_a.none(), "threshold"_a.none(), "clock"_a,
       "agg_functor"_a, "agg"_a, "label"_a, "fill"_a, "max_age"_a.none(), "max_age_whole"_a,
       "origin"_a.none() = nb::none());
    // The string rule for values (cast_double.h refuse_text), for the Python
    // float64 conversions in screamer/dag.py and screamer/streams.py.
    m.def("_refuse_text", [](nb::handle x, const std::string& who) { refuse_text(x, who); },
          "values"_a.none(), "who"_a = "");
    // After a failed float64 conversion: the TypeError naming the first
    // element that is not a number (no NumPy text); returns when there is none.
    m.def("_first_non_number", [](nb::handle x, const std::string& who) {
        throw_first_non_number(x, who);
    }, "values"_a, "who"_a = "");
    // The one-kind rule of a call (py_index.h TimeUnit) over the index arrays of
    // the batch stream functions (merge, combine_latest), which build their own
    // int64 arrays: every non-empty array must be numbers, or times of one unit.
    m.def("_check_index_kinds", [](nb::list indexes) {
        TimeUnit unit;
        nb::module_ np = nb::module_::import_("numpy");
        for (nb::handle h : indexes) {
            // ravel: a 0-d index (one event, as 2.8.0 read it) counts as one entry.
            nb::object arr = np.attr("asarray")(h).attr("ravel")();
            if (nb::len(arr) == 0) continue;
            const std::string kind = nb::cast<std::string>(arr.attr("dtype").attr("kind"));
            // A bool array is not an index, as in a Pipeline batch call.
            if (kind == "b") dag::throw_index_kind("an array of bool");
            if (kind == "M" || kind == "m") {
                nb::tuple du = nb::cast<nb::tuple>(np.attr("datetime_data")(arr.attr("dtype")));
                note_time(&unit, kind == "M" ? TimeUnit::kDatetime : TimeUnit::kTimedelta,
                          time_unit_base(nb::cast<std::string>(du[0])), nb::cast<int>(du[1]));
            } else if (kind == "O") {
                for (nb::handle e : arr) py_index(e, &unit);
            } else if (kind == "i" || kind == "u" || kind == "f") {
                note_number(&unit);
            } else {
                // Text, complex or another kind: not an index (TypeError).
                dag::throw_index_kind("an array of " +
                                      nb::cast<std::string>(arr.attr("dtype").attr("name")));
            }
        }
    }, "indexes"_a);
    // An index array as int64 under the engine's index rule (index_array): for
    // a batch stream merge that takes a float index after an int one, as the
    // iterator form does. A fractional, NaN or out-of-range value raises.
    m.def("_index_i64", [](nb::handle index) {
        TimeUnit unit;
        nb::object keep;
        std::vector<std::int64_t> owned;
        std::size_t n = 0;
        const std::int64_t* p = index_array(index, keep, owned, unit, n);
        std::unique_ptr<std::int64_t[]> out(new std::int64_t[n ? n : 1]);
        if (n) std::memcpy(out.get(), p, n * sizeof(std::int64_t));
        return owned_array(out.release(), {n});
    }, "index"_a);
    // One value under the same rules: a number, None as NaN; a string or any
    // other kind is a TypeError.
    m.def("_value", [](nb::handle x) {
        double d;
        if (!is_text(x) && read_double(x, d)) return d;
        throw_text_value("", x);
    }, "value"_a.none());

    m.def("_run_functor_batch", &run_functor_batch,
          "op"_a, "index"_a, "values"_a);

    // Compiled graph wrapper: holds a persistent CompiledGraph plus op_refs so
    // functor Python objects stay alive for the compiled graph's lifetime.
    struct PyCompiledGraph {
        std::vector<nb::object> op_refs;  // destroyed AFTER cg (declared first)
        std::unique_ptr<dag::CompiledGraph> cg;  // destroyed FIRST (declared last)

        // The graph points into op_refs, so the collector only reads them (gc.h).
        int traverse(visitproc visit, void* arg) const {
            return gc_visit_all(op_refs, visit, arg);
        }

        void reset() { cg->reset(); }

        void push_event(std::size_t input_idx, std::int64_t index, double value) {
            cg->push_event(input_idx, index, value);
        }

        void push_event_wide(std::size_t input_idx, std::int64_t index,
                             InValues vals) {
            const double* ptr = vals.data();
            std::size_t w = static_cast<std::size_t>(vals.size());
            cg->push_event_wide(input_idx, index, ptr, w);
        }

        void flush() { cg->flush(); }

        void advance(nb::handle now) { cg->advance(py_index(now)); }

        // Drains the outputs: a list of (index, values2d), or with `align`
        // aligned on one index (dag::align_outputs) as (index, [values2d, ...]).
        // The aligned rows carry the join across drains (drain_aligned).
        nb::object drain(bool align) {
            if (!align) return marshal_output_buffers(cg->drain());
            std::vector<dag::OutputBuffer> bufs;
            return marshal_aligned(cg->drain_aligned(bufs));
        }

        // Runs the graph over the feeds. Returns the outputs as a list of
        // (index, values2d), or with `align` aligned on one index
        // (dag::align_outputs) as (index, [values2d, ...]).
        // `who` names the caller in a feed error ("Pipeline", or "resample"
        // for a Pipeline that runs an eager stream operator).
        nb::object run_batch(nb::list feeds, bool align, const std::string& who) {
            std::vector<nb::object> ks;
            std::vector<std::vector<std::int64_t>> owned;
            std::vector<InValues> vs;
            std::vector<const std::int64_t*> kp;
            std::vector<const double*> vp;
            std::vector<std::size_t> lens;
            std::vector<std::size_t> widths;
            marshal_feeds(feeds, ks, owned, vs, kp, vp, lens, widths, who);
            const std::vector<dag::OutputBuffer>& outs = cg->run_batch_ref(kp, vp, lens, widths);
            if (!align) return marshal_output_buffers(outs);
            std::vector<dag::OutputBuffer> bufs;   // the rows move out of the graph
            return marshal_aligned(cg->drain_aligned(bufs));
        }
    };

    // Register _CompiledGraph before _GraphBuilder so compile() return type is known.
    nb::class_<PyCompiledGraph>(m, "_CompiledGraph",
                                nb::type_slots(gc_traverse_slots<PyCompiledGraph>))
        .def("reset",            &PyCompiledGraph::reset)
        .def("push_event",       &PyCompiledGraph::push_event,
             "input_idx"_a, "index"_a, "value"_a)
        .def("push_event_wide",  &PyCompiledGraph::push_event_wide,
             "input_idx"_a, "index"_a, "values"_a)
        .def("flush",            &PyCompiledGraph::flush)
        .def("advance",          &PyCompiledGraph::advance, "now"_a.none())
        .def("drain",            &PyCompiledGraph::drain, "align"_a = false)
        .def("run_batch",        &PyCompiledGraph::run_batch, "feeds"_a, "align"_a = false,
             "who"_a = "Pipeline");

    // The forecast_pairs graph (dag/forecast_pairs.h), compiled: inputs X and y,
    // one output [X_shifted..., y].
    m.def("_forecast_pairs_graph", [](std::optional<std::int64_t> count,
                                      std::optional<std::int64_t> duration, bool dropna,
                                      bool features_index) {
        PyCompiledGraph pcg;
        pcg.cg = std::make_unique<dag::CompiledGraph>(
            dag::forecast_pairs_graph(count, duration, dropna, features_index));
        return pcg;
    }, "count"_a.none(), "duration"_a.none(), "dropna"_a, "features_index"_a);
    // The index setting ("features" or "target") as features_index.
    m.def("_forecast_pairs_index", &dag::forecast_pairs_features_index, "index"_a);
    m.def("_check_forecast_pairs_inputs", &dag::check_forecast_pairs_inputs,
          "count_mode"_a, "x_ndim"_a, "x_rows"_a, "x_cols"_a, "y_ndim"_a, "y_rows"_a,
          "y_cols"_a);
    // The horizon rule on Python ints (None when absent). An int beyond int64
    // is out of range, with the rule's own message.
    m.def("_check_forecast_pairs_horizon", [](nb::handle count, nb::handle duration) {
        auto read = [](nb::handle h, const char* name) -> std::optional<std::int64_t> {
            if (h.is_none()) return std::nullopt;
            int overflow = 0;
            const long long v = PyLong_AsLongLongAndOverflow(h.ptr(), &overflow);
            if (v == -1 && !overflow && PyErr_Occurred()) throw nb::python_error();
            if (overflow) dag::throw_forecast_pairs_range(name, nb::cast<std::string>(nb::str(h)));
            return static_cast<std::int64_t>(v);
        };
        auto c = read(count, "count");
        auto d = read(duration, "duration");
        dag::check_forecast_pairs_horizon(c, d);
    }, "count"_a.none(), "duration"_a.none());

    // Live session: the per-event push, in C++ (name lookup, one conversion).
    // The Python _LiveDag subclasses this and adds result().
    struct PyLiveSession {
        nb::object cg_obj;               // the _CompiledGraph (keeps its ops alive)
        PyCompiledGraph* pcg;
        nb::object asarray, float64;     // numpy, for the sequence path
        std::map<std::string, std::size_t, std::less<>> names;
        std::vector<double> scratch;     // the plain-sequence conversion buffer
        TimeUnit unit;                   // the session's index kind (py_index.h)

        PyLiveSession(nb::object cg, nb::list input_names)
            : cg_obj(std::move(cg)), pcg(&nb::cast<PyCompiledGraph&>(cg_obj)) {
            nb::module_ np = nb::module_::import_("numpy");
            asarray = np.attr("asarray");
            float64 = np.attr("float64");
            std::size_t i = 0;
            for (nb::handle h : input_names) names[nb::cast<std::string>(h)] = i++;
        }

        int traverse(visitproc visit, void* arg) const {
            if (int r = gc_visit(cg_obj, visit, arg)) return r;
            if (int r = gc_visit(asarray, visit, arg)) return r;
            return gc_visit(float64, visit, arg);
        }

        std::size_t source(nb::handle input) const {
            PyObject* p = input.ptr();
            if (PyUnicode_Check(p)) {
                Py_ssize_t n = 0;
                const char* c = PyUnicode_AsUTF8AndSize(p, &n);
                if (!c) throw nb::python_error();
                auto it = names.find(std::string_view(c, static_cast<std::size_t>(n)));
                if (it == names.end()) throw nb::key_error(c);
                return it->second;
            }
            PyObject* ix = PyNumber_Index(p);
            if (!ix) throw nb::python_error();
            Py_ssize_t k = PyNumber_AsSsize_t(ix, PyExc_OverflowError);
            Py_DECREF(ix);
            if (k == -1 && PyErr_Occurred()) throw nb::python_error();
            return static_cast<std::size_t>(k);   // negative wraps, the engine range-checks
        }

        // list / tuple of plain numbers, converted in one pass into a reused buffer.
        // Returns false (nothing pushed) for any other item, which takes the general path.
        bool push_plain_sequence(std::size_t src, std::int64_t t, PyObject* p) {
            const bool is_list = PyList_Check(p);
            const Py_ssize_t n = is_list ? PyList_Size(p) : PyTuple_Size(p);
            if (n <= 0) return false;
            scratch.resize(static_cast<std::size_t>(n));
            for (Py_ssize_t i = 0; i < n; ++i) {
                PyObject* x = is_list ? PyList_GetItem(p, i) : PyTuple_GetItem(p, i);
                if (PyFloat_Check(x)) scratch[i] = PyFloat_AsDouble(x);
                else if (PyLong_Check(x)) {
                    double d = PyLong_AsDouble(x);
                    if (d == -1.0 && PyErr_Occurred()) throw nb::python_error();
                    scratch[i] = d;
                } else return false;
            }
            pcg->cg->push_event_wide(src, t, scratch.data(), scratch.size());
            return true;
        }

        // A C-contiguous 1-D numeric buffer (ndarray of float64, float32 or an
        // integer type, bool, array.array, a numeric memoryview ...), read
        // through the buffer protocol with one pass of conversion. Returns false
        // (nothing pushed) for any other buffer, which takes the general path.
        bool push_buffer(std::size_t src, std::int64_t t, PyObject* p) {
            if (!PyObject_CheckBuffer(p)) return false;
            Py_buffer view;
            if (PyObject_GetBuffer(p, &view, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) != 0) {
                PyErr_Clear();
                return false;
            }
            struct Release { Py_buffer* v; ~Release() { PyBuffer_Release(v); } } release{&view};
            if (view.ndim != 1 || view.len == 0 || !view.format) return false;
            const char* f = view.format;
            if (*f == '@' || *f == '=' || *f == '<') ++f;
            if (f[0] == '\0' || f[1] != '\0') return false;
            const std::size_t n = static_cast<std::size_t>(view.shape[0]);
            const void* b = view.buf;
            const Py_ssize_t sz = view.itemsize;
            bool ok = false;
            switch (*f) {
            case 'd':
                if (sz != 8) return false;
                if (reinterpret_cast<std::uintptr_t>(b) % alignof(double) == 0) {
                    pcg->cg->push_event_wide(src, t, static_cast<const double*>(b), n);
                    return true;
                }
                ok = load_items<double>(b, n, sz, scratch); break;
            case 'f': ok = load_items<float>(b, n, sz, scratch); break;
            case '?': case 'B': ok = load_items<unsigned char>(b, n, sz, scratch); break;
            case 'b': ok = load_items<signed char>(b, n, sz, scratch); break;
            case 'h': ok = load_items<std::int16_t>(b, n, sz, scratch); break;
            case 'H': ok = load_items<std::uint16_t>(b, n, sz, scratch); break;
            case 'i': case 'l': case 'q':
                ok = sz == 8 ? load_items<std::int64_t>(b, n, sz, scratch)
                             : load_items<std::int32_t>(b, n, sz, scratch);
                break;
            case 'I': case 'L': case 'Q':
                ok = sz == 8 ? load_items<std::uint64_t>(b, n, sz, scratch)
                             : load_items<std::uint32_t>(b, n, sz, scratch);
                break;
            default: return false;
            }
            if (!ok) return false;
            pcg->cg->push_event_wide(src, t, scratch.data(), n);
            return true;
        }

        void push_sequence(std::size_t src, std::int64_t t, nb::handle value) {
            PyObject* p = value.ptr();
            if ((PyList_Check(p) || PyTuple_Check(p)) && push_plain_sequence(src, t, p)) return;
            if (push_buffer(src, t, p)) return;
            // The general path: NumPy reads the value, then a complex one is
            // refused before the float64 conversion would drop its imaginary part.
            // A value with a dtype (an array, a pandas Series) is cast as it is,
            // so pandas maps NA to NaN; one without (a list, a range) is read by
            // NumPy first. Either way a complex dtype is refused before the
            // float64 cast would drop the imaginary part.
            // A string inside is a wrong kind (refuse_text), before the cast
            // would read "3" as 3.
            if (PyList_Check(p) || PyTuple_Check(p))
                refuse_text(value);                // a bytearray item reads as numbers below
            nb::object arr = nb::getattr(value, "dtype", nb::none()).is_none()
                ? asarray(value) : nb::borrow<nb::object>(value);
            nb::object dt = nb::getattr(arr, "dtype", nb::none());
            nb::object kind = dt.is_none() ? nb::none() : nb::getattr(dt, "kind", nb::none());
            if (nb::isinstance<nb::str>(kind)) {
                const std::string k = nb::cast<std::string>(kind);
                if (k == "c")
                    throw nb::type_error("a value must be a real number or a sequence of "
                                         "real numbers, got complex values");
                if (k == "U" || k == "S" || k == "O" || k == "M" || k == "m") refuse_text(arr);
            }
            InValues v = nb::cast<InValues>(asarray(arr, nb::arg("dtype") = float64));
            if (v.ndim() == 0) {                   // a 0-d array is a scalar
                pcg->cg->push_event(src, t, v.data()[0]);
                return;
            }
            if (v.ndim() != 1)
                throw nb::value_error(
                    ("a wide event is a 1-D sequence of numbers, got " +
                     std::to_string(v.ndim()) + " dimensions").c_str());
            pcg->cg->push_event_wide(src, t, v.data(), static_cast<std::size_t>(v.size()));
        }

        void push(nb::handle input, nb::handle index, nb::handle value) {
            const std::size_t src = source(input);
            // The session keeps one index kind (py_index.h). The index is read
            // into a copy, committed only when the event is taken, so a rejected
            // push leaves the session as it was.
            TimeUnit u = unit;
            const std::int64_t t = py_index(index, &u);
            push_value(src, t, value);
            unit = u;
        }

        // advance(now) under the session's index kind, committed when it succeeds.
        void advance(nb::handle now) {
            TimeUnit u = unit;
            const std::int64_t t = py_index(now, &u);
            pcg->cg->advance(t);
            unit = u;
        }

        void push_value(std::size_t src, std::int64_t t, nb::handle value) {
            PyObject* p = value.ptr();
            if (PyFloat_Check(p)) {                // float, np.float64
                pcg->cg->push_event(src, t, PyFloat_AsDouble(p));
            } else if (PyLong_Check(p)) {          // int, bool
                double d = PyLong_AsDouble(p);
                if (d == -1.0 && PyErr_Occurred()) throw nb::python_error();
                pcg->cg->push_event(src, t, d);
            } else if (p == Py_None) {             // a missing value, NaN (A9)
                pcg->cg->push_event(src, t, std::numeric_limits<double>::quiet_NaN());
            } else if (PyUnicode_Check(p) || PyBytes_Check(p) || PyByteArray_Check(p)) {
                throw_text_value("", value);       // a string is never a number (R-a)
            } else if (is_foreign_value(p)) {
                throw_text_value("", value);       // a time or a ctypes object (TypeError)
            } else if (PyMemoryView_Check(p)) {
                // A numeric memoryview is a wide event; one over bytes is text.
                nb::object base = value.attr("obj");
                if (is_foreign_value(base.ptr())) throw_text_value("", base);   // a time's bytes
                if (PyBytes_Check(base.ptr()) || PyByteArray_Check(base.ptr()) ||
                    nb::cast<std::string>(value.attr("format")) == "c")
                    throw nb::type_error("a value must be a number or a sequence of "
                                         "numbers, not a memoryview of bytes");
                push_sequence(src, t, value);
            } else if (PySequence_Check(p)) {
                push_sequence(src, t, value);      // list, tuple, ndarray, Series ...
            } else {                               // NumPy scalars
                const NumpyTypes& np = numpy_types();
                PyObject* type = reinterpret_cast<PyObject*>(Py_TYPE(p));
                if (PyComplex_Check(p) || type == np.complex64 || type == np.clongdouble)
                    throw nb::type_error("a value must be a real number, got complex");
                double d = PyFloat_AsDouble(p);    // via __float__
                if (d == -1.0 && PyErr_Occurred()) throw nb::python_error();
                pcg->cg->push_event(src, t, d);
            }
        }
    };

    nb::class_<PyLiveSession>(m, "_LiveSession", nb::dynamic_attr(),
                              nb::type_slots(gc_traverse_slots<PyLiveSession>))
        .def(nb::init<nb::object, nb::list>(), "cg"_a, "input_names"_a)
        .def("push", [](nb::handle self, nb::handle input, nb::handle index,
                        nb::handle value) {
                 nb::inst_ptr<PyLiveSession>(self)->push(input, index, value);
                 return nb::borrow(self);
             }, "input"_a, "index"_a.none(), "value"_a.none())
        .def("_advance", [](nb::handle self, nb::handle now) {
                 nb::inst_ptr<PyLiveSession>(self)->advance(now);
             }, "now"_a.none());

    // Lazy driver over a compiled graph and Python-iterator feeds.
    nb::class_<LazyDriver>(m, "_LazyDriver", nb::type_slots(gc_slots<LazyDriver>))
        .def("__init__", [](LazyDriver* self, nb::object cg_obj, nb::list iterators,
                            std::size_t n_out, std::string owner, std::ptrdiff_t shape_input,
                            std::vector<std::size_t> shape_tail) {
                 PyCompiledGraph& pcg = nb::cast<PyCompiledGraph&>(cg_obj);
                 new (self) LazyDriver(cg_obj, *pcg.cg, iterators, n_out, std::move(owner),
                                       shape_input, std::move(shape_tail));
             },
             "cg"_a, "iterators"_a, "n_out"_a, "owner"_a = "Pipeline",
             "shape_input"_a = -1, "shape_tail"_a = std::vector<std::size_t>{})
        .def("__iter__", [](nb::object self) { return self; })
        // An index error of the merge names the input, not the API: put the
        // owner ("Pipeline") in front, as merge and combine_latest do.
        .def("__next__", [](LazyDriver& d) -> nb::object {
            try {
                return d.next();
            } catch (const std::invalid_argument& e) {
                static const char kIndex[] = "the index of input ";
                if (std::strncmp(e.what(), kIndex, sizeof kIndex - 1) != 0) throw;
                if (dynamic_cast<const dag::IndexTypeError*>(&e))
                    throw dag::IndexTypeError(d.owner() + ": " + e.what());
                throw std::invalid_argument(d.owner() + ": " + e.what());
            }
        });

    // Python-facing GraphBuilder wrapper that keeps functor Python objects alive
    // for the lifetime of the builder (raw EvalOp* point into Python objects;
    // if the caller passes temporaries they'd be GC'd without this ref-holding).
    struct PyGraphBuilder {
        dag::GraphBuilder builder;
        std::vector<nb::object> op_refs;  // keeps Python functor objects alive

        // The builder points into op_refs, so the collector only reads them (gc.h).
        int traverse(visitproc visit, void* arg) const {
            return gc_visit_all(op_refs, visit, arg);
        }

        std::size_t add_input() { return builder.add_input(); }

        std::size_t add_functor(nb::object op_obj, std::vector<std::size_t> inputs) {
            EvalOp* op = nb::cast<EvalOp*>(op_obj);
            op_refs.push_back(op_obj);
            return builder.add_functor(op, std::move(inputs));
        }

        std::size_t add_combine_latest(std::vector<std::size_t> inputs, bool when_all,
                                       std::size_t max_pending) {
            return builder.add_combine_latest(std::move(inputs), when_all, max_pending);
        }

        std::size_t add_filter(std::vector<std::size_t> inputs) {
            return builder.add_filter(std::move(inputs));
        }

        std::size_t add_dropna(std::vector<std::size_t> inputs, bool how_all) {
            return builder.add_dropna(std::move(inputs), how_all);
        }

        std::size_t add_select(std::vector<std::size_t> inputs,
                               std::vector<std::size_t> columns) {
            return builder.add_select(std::move(inputs), std::move(columns));
        }

        std::size_t add_delay(std::vector<std::size_t> inputs, std::int64_t duration) {
            return builder.add_delay(std::move(inputs), duration);
        }

        std::size_t add_resample(std::vector<std::size_t> inputs, int mode, int agg,
                                 int label, std::int64_t width, std::int64_t origin,
                                 std::int64_t count, nb::object reducer, int fill,
                                 nb::list plan, double threshold, std::int64_t max_age,
                                 std::size_t max_pending, std::int64_t max_fill) {
            dag::ResampleParams rp;
            rp.mode      = static_cast<dag::ResampleMode>(mode);    // 0=ByIndex, 1=ByCount, 2=ByCumulative
            rp.agg       = static_cast<dag::ResampleAgg>(agg);      // 0..9 First..SumNeg
            rp.label     = static_cast<dag::ResampleLabel>(label);  // 0=Left, 1=Right
            rp.fill      = static_cast<dag::ResampleFill>(fill);    // 0=Skip, 1=Nan, 2=Carry
            rp.width     = width;
            rp.origin    = origin;
            rp.count     = count;
            rp.threshold = threshold;
            rp.max_age   = max_age;
            rp.max_fill  = dag::checked_max_fill(static_cast<double>(max_fill));
            // Optional per-column reducer plan (multi-column bar aggs).
            // Each plan entry is a (agg_code, input_col) tuple.
            for (auto item : plan) {
                nb::tuple t = nb::cast<nb::tuple>(item);
                dag::ResamplePlanEntry e{};
                e.agg       = static_cast<dag::ResampleAgg>(nb::cast<int>(t[0]));
                e.input_col = nb::cast<std::size_t>(t[1]);
                rp.plan.push_back(e);
            }
            // Optional functor reducer: extract the base EvalOp* and keep the Python
            // object alive for the compiled graph's lifetime (op_refs is copied into
            // the _CompiledGraph at compile()). Raw pointer would else dangle on GC.
            if (!reducer.is_none()) {
                EvalOp* op = nb::cast<EvalOp*>(reducer);
                rp.reducer = op;
                op_refs.push_back(reducer);
            }
            return builder.add_resample(std::move(inputs), rp, max_pending);
        }

        // The forecast_pairs nodes (dag/forecast_pairs.h).
        std::size_t add_forecast_pairs(std::size_t x, std::size_t y,
                                       std::optional<std::int64_t> count,
                                       std::optional<std::int64_t> duration, bool dropna,
                                       bool features_index) {
            return dag::add_forecast_pairs(builder, x, y, count, duration, dropna,
                                           features_index);
        }

        void set_max_pending(std::size_t n) { builder.set_max_pending(n); }

        void set_outputs(std::vector<std::size_t> outs) {
            builder.set_outputs(std::move(outs));
        }

        const dag::GraphSpec& spec() const { return builder.spec(); }

        // Compile the accumulated spec into a persistent CompiledGraph.
        // The returned _CompiledGraph also holds op_refs so functors stay alive.
        // The spec moves into the graph, so a builder compiles once: two
        // graphs over one builder would share its operators' state.
        PyCompiledGraph compile() {
            if (compiled)
                throw std::runtime_error("this graph builder was compiled already; "
                                         "build a new one");
            compiled = true;          // the spec leaves the builder even if the build throws
            PyCompiledGraph pcg;
            pcg.cg = std::make_unique<dag::CompiledGraph>(builder.take());
            pcg.op_refs = op_refs;
            return pcg;
        }
        bool compiled = false;
    };

    // DAG compiler: _GraphBuilder accumulates a GraphSpec.
    // run_batch compiles and drives it fresh each call (rebuild-per-run).
    // compile() returns a persistent _CompiledGraph for streaming use.
    nb::class_<PyGraphBuilder>(m, "_GraphBuilder",
                               nb::type_slots(gc_traverse_slots<PyGraphBuilder>))
        .def(nb::init<>())
        .def("add_input", &PyGraphBuilder::add_input)
        .def("add_functor", [](PyGraphBuilder& b, nb::object op,
                               std::vector<std::size_t> inputs) {
            return b.add_functor(op, std::move(inputs));
        }, "op"_a, "inputs"_a)
        .def("add_combine_latest", [](PyGraphBuilder& b,
                                      std::vector<std::size_t> inputs, bool when_all,
                                      std::size_t max_pending) {
            return b.add_combine_latest(std::move(inputs), when_all, max_pending);
        }, "inputs"_a, "when_all"_a = true,
           "max_pending"_a = dag::kDefaultMaxPending)
        .def("add_filter", [](PyGraphBuilder& b, std::vector<std::size_t> inputs) {
            return b.add_filter(std::move(inputs));
        }, "inputs"_a)
        .def("add_dropna", [](PyGraphBuilder& b,
                              std::vector<std::size_t> inputs, bool how_all) {
            return b.add_dropna(std::move(inputs), how_all);
        }, "inputs"_a, "how_all"_a = false)
        .def("add_select", [](PyGraphBuilder& b,
                              std::vector<std::size_t> inputs,
                              std::vector<std::size_t> columns) {
            return b.add_select(std::move(inputs), std::move(columns));
        }, "inputs"_a, "columns"_a)
        .def("add_delay", [](PyGraphBuilder& b, std::vector<std::size_t> inputs,
                             std::int64_t duration) {
            return b.add_delay(std::move(inputs), duration);
        }, "inputs"_a, "duration"_a)
        .def("add_resample", [](PyGraphBuilder& b, std::vector<std::size_t> inputs,
                                int mode, int agg, int label,
                                std::int64_t width, std::int64_t origin, std::int64_t count,
                                nb::object reducer, int fill, nb::list plan, double threshold,
                                std::int64_t max_age, std::size_t max_pending,
                                std::int64_t max_fill) {
            return b.add_resample(std::move(inputs), mode, agg, label, width, origin,
                                  count, reducer, fill, plan, threshold, max_age,
                                  max_pending, max_fill);
        }, "inputs"_a, "mode"_a, "agg"_a, "label"_a,
           "width"_a, "origin"_a, "count"_a,
           "reducer"_a = nb::none(), "fill"_a = 0,
           "plan"_a = nb::list(), "threshold"_a = 0.0,
           "max_age"_a = -1,
           "max_pending"_a = dag::kDefaultMaxPending,
           "max_fill"_a = dag::kDefaultMaxFill)
        .def("add_forecast_pairs", &PyGraphBuilder::add_forecast_pairs,
             "x"_a, "y"_a, "count"_a.none(), "duration"_a.none(),
             "dropna"_a, "features_index"_a)
        .def("set_outputs", &PyGraphBuilder::set_outputs, "output_ids"_a)
        .def("set_max_pending", &PyGraphBuilder::set_max_pending, "max_pending"_a)
        .def("compile", [](PyGraphBuilder& b) { return b.compile(); })
        .def("run_batch", [](PyGraphBuilder& b, nb::list feeds) {
            // Marshal feeds (list of (index, values) tuples) -> raw spans.
            std::vector<nb::object> ks;
            std::vector<std::vector<std::int64_t>> owned;
            std::vector<InValues> vs;
            std::vector<const std::int64_t*> kp;
            std::vector<const double*> vp;
            std::vector<std::size_t> lens;
            std::vector<std::size_t> widths;
            marshal_feeds(feeds, ks, owned, vs, kp, vp, lens, widths);
            // Compile (stores spec) then run (builds + drives push-graph).
            dag::CompiledGraph g(b.spec());
            return marshal_output_buffers(g.run_batch(kp, vp, lens, widths));
        }, "feeds"_a);
}
