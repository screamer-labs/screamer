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
};

}}  // namespace screamer::dag
#endif
