#ifndef SCREAMER_DETAIL_TEST_ACCESS_H
#define SCREAMER_DETAIL_TEST_ACCESS_H

namespace screamer { namespace detail {

// Declared only: a test defines it to read an operator's private state (the
// event counters in tests/test_event_counters_64bit.py). The library never
// defines or uses it.
struct TestAccess;

}}  // namespace screamer::detail
#endif
