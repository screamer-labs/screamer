#ifndef SCREAMER_COMMON_COLUMN_DRIVER_H
#define SCREAMER_COMMON_COLUMN_DRIVER_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

namespace screamer {

// The most inputs an operator's column can have (the largest operator has 10).
constexpr std::size_t kMaxColumnInputs = 16;

namespace detail {
// Columns per tile.
constexpr std::size_t kTileColumns = 8;
// Scratch budget of a tile, in bytes.
constexpr std::size_t kTileBytes = std::size_t(4) << 20;
// Tiling policy, measured on (T, K) arrays (see run_columns). A strided column
// is tiled when its byte stride is a multiple of kConflictStrideBytes (its
// lines share a few cache sets) and it spans more than kConflictSpanBytes
// (T times the stride; a shorter column stays in L1 and tiling is pure
// overhead), or, for an operator with a tight inlined loop, when it is longer
// than kDirectColumnSteps and its tile fits kCheapTileBytes.
constexpr std::size_t kConflictStrideBytes = 512;
constexpr std::size_t kConflictSpanBytes = std::size_t(32) << 10;
constexpr std::size_t kDirectColumnSteps = 256;
constexpr std::size_t kCheapTileBytes = std::size_t(512) << 10;

inline bool conflict_stride(std::ptrdiff_t elements) {
    return (static_cast<std::size_t>(elements) * sizeof(double)) % kConflictStrideBytes == 0;
}

// Copy T rows of n strided columns (src[b][t * s]) into n contiguous columns
// (dst[b * T + t]). NB > 0 fixes n at compile time.
template <std::size_t NB>
inline void gather_tile(double* dst, std::size_t T, const double* const* src,
                        std::ptrdiff_t s, std::size_t nb) {
    const std::size_t n = NB ? NB : nb;
    for (std::size_t t = 0; t < T; ++t) {
        const std::ptrdiff_t k = static_cast<std::ptrdiff_t>(t) * s;
        for (std::size_t b = 0; b < n; ++b) dst[b * T + t] = src[b][k];
    }
}

// The reverse for outputs: n contiguous columns of T rows of n_out values back
// into the rows of the (T, cols, n_out) result, from column c0 on.
template <std::size_t NB, std::size_t NOUT>
inline void scatter_tile(double* out, std::size_t cols, std::size_t c0, std::size_t T,
                         const double* src, std::size_t nb, std::size_t n_out_rt) {
    const std::size_t n = NB ? NB : nb;
    const std::size_t no = NOUT ? NOUT : n_out_rt;
    for (std::size_t t = 0; t < T; ++t) {
        double* row = out + (t * cols + c0) * no;
        for (std::size_t b = 0; b < n; ++b)
            for (std::size_t m = 0; m < no; ++m) row[b * no + m] = src[(b * T + t) * no + m];
    }
}
}  // namespace detail

// Runs a batch of `cols` independent columns of T steps, the one column loop of
// both bindings. Input i of column c starts at base[i] + offset(i, c) and
// advances by time_stride[i] per step; output column c starts at
// out + c * n_out and advances by cols * n_out (row-major (T, cols, n_out)).
// `run(in, in_stride, T, out, out_stride)` is the operator's eval_column.
//
// A column of a (T, K) array has a stride of K elements, so a long column
// walked in place touches a new cache line every step, and a tight loop over
// it stalls on memory (worst at power-of-two K, where the lines share a few
// cache sets). Long strided columns therefore run in tiles of 8 adjacent
// columns: the tile's strided inputs are gathered into contiguous scratch row
// by row (one cache line feeds the whole tile), each column runs with unit
// stride, and the outputs are scattered back the same way. Results are the
// same either way: every column is still one eval_column.
//
// Tiling costs a gather and a scatter pass, which only pays when the in-place
// walk stalls: at conflict strides, and for `tight` operators (FunctorBase,
// whose per-step call is inlined into the column loop) on long columns. A
// ScreamerBase operator calls process_scalar per step, which hides the strided
// reads, so it is tiled at conflict strides only. A multi-output operator pays
// a scatter of n_out values per step and is never tiled.
template <class Run, class Offset>
void run_columns(Run&& run, std::size_t n_in, std::size_t n_out, std::size_t T,
                 std::size_t cols, const double* const* base,
                 const std::size_t* time_stride, Offset&& offset, double* out,
                 bool tight) {
    if (n_in > kMaxColumnInputs)
        throw std::invalid_argument("an operator with " + std::to_string(n_in) +
                                    " inputs exceeds the column driver's limit of " +
                                    std::to_string(kMaxColumnInputs));
    constexpr std::size_t B = detail::kTileColumns;
    std::array<const double*, kMaxColumnInputs> ptr{};
    std::array<std::ptrdiff_t, kMaxColumnInputs> stride{};
    std::size_t n_gather = 0;
    for (std::size_t i = 0; i < n_in; ++i) {
        stride[i] = static_cast<std::ptrdiff_t>(time_stride[i]);
        if (stride[i] != 1) ++n_gather;
    }
    const std::ptrdiff_t out_stride = static_cast<std::ptrdiff_t>(cols * n_out);
    const std::size_t tile_doubles = B * T * (n_gather + n_out);
    const std::size_t tile_bytes = tile_doubles * sizeof(double);
    // A conflict stride counts only once the column spans more than L1 holds.
    const auto conflicts = [T](std::ptrdiff_t st) {
        return detail::conflict_stride(st) &&
               T * static_cast<std::size_t>(st) * sizeof(double) > detail::kConflictSpanBytes;
    };
    bool conflict = conflicts(out_stride);
    for (std::size_t i = 0; i < n_in; ++i)
        if (stride[i] != 1 && conflicts(stride[i])) conflict = true;
    const bool tile = cols >= 2 && n_out == 1 && tile_bytes <= detail::kTileBytes &&
                      (conflict || (tight && T > detail::kDirectColumnSteps &&
                                    tile_bytes <= detail::kCheapTileBytes));
    if (!tile) {
        for (std::size_t c = 0; c < cols; ++c) {
            for (std::size_t i = 0; i < n_in; ++i) ptr[i] = base[i] + offset(i, c);
            run(ptr.data(), stride.data(), T, out + c * n_out, out_stride);
        }
        return;
    }

    // Scratch: B columns of T values per strided input, then B columns of
    // T * n_out outputs.
    std::unique_ptr<double[]> scratch(new double[tile_doubles]);
    double* const out_tile = scratch.get() + n_gather * B * T;
    std::array<std::ptrdiff_t, kMaxColumnInputs> unit{};
    for (std::size_t i = 0; i < n_in; ++i) unit[i] = 1;
    std::array<const double*, B> src{};

    for (std::size_t c0 = 0; c0 < cols; c0 += B) {
        const std::size_t nb = std::min(B, cols - c0);
        for (std::size_t i = 0, g = 0; i < n_in; ++i) {
            if (stride[i] == 1) continue;
            for (std::size_t b = 0; b < nb; ++b) src[b] = base[i] + offset(i, c0 + b);
            double* dst = scratch.get() + (g++) * B * T;
            if (nb == B) detail::gather_tile<B>(dst, T, src.data(), stride[i], nb);
            else detail::gather_tile<0>(dst, T, src.data(), stride[i], nb);
        }
        for (std::size_t b = 0; b < nb; ++b) {
            for (std::size_t i = 0, g = 0; i < n_in; ++i) {
                if (stride[i] == 1) ptr[i] = base[i] + offset(i, c0 + b);
                else ptr[i] = scratch.get() + ((g++) * B + b) * T;
            }
            run(ptr.data(), unit.data(), T, out_tile + b * T * n_out,
                static_cast<std::ptrdiff_t>(n_out));
        }
        if (nb == B) detail::scatter_tile<B, 1>(out, cols, c0, T, out_tile, nb, 1);
        else detail::scatter_tile<0, 1>(out, cols, c0, T, out_tile, nb, 1);
    }
}

}  // namespace screamer
#endif
