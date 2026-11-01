#ifndef SCREAMER_DAG_OUTPUT_BUFFER_H
#define SCREAMER_DAG_OUTPUT_BUFFER_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include "screamer/dag/node_state.h"

namespace screamer { namespace dag {

// One output stream gathered during a run.
struct OutputBuffer {
    std::vector<std::int64_t> indices;
    std::vector<double> values;   // row-major, width columns per row
    std::size_t width = 1;
    // The output's highest event-time watermark so far: it will emit no row
    // with an index below `watermark` (valid when has_watermark).
    std::int64_t watermark = 0;
    bool has_watermark = false;

    // The rows not drained yet and the watermark (serial/archive.h), saved
    // by a live session.
    template <class Ar>
    void state(Ar& ar) {
        ar("width", width);
        ar("index", indices);
        ar.fixed("values", values, indices.size() * width);
        optional_index(ar, "watermark", has_watermark, watermark);
        for (std::size_t r = 1; r < indices.size(); ++r)
            ar.check(indices[r - 1] <= indices[r], "the rows are out of order");
    }
};

}}  // namespace screamer::dag
#endif
