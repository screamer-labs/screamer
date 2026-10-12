#ifndef SCREAMER_DAG_OUTPUT_BUFFER_H
#define SCREAMER_DAG_OUTPUT_BUFFER_H

#include <cstddef>
#include <cstdint>
#include <vector>

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
};

}}  // namespace screamer::dag
#endif
