#ifndef SCREAMER_STREAMS_EVENT_H
#define SCREAMER_STREAMS_EVENT_H

#include <cstdint>
#include <optional>

namespace screamer { namespace streams {

template <class Index>
struct Event {
    Index index;
    double value;
    std::uint32_t source = 0;   // provenance tag; used by merge later
};

template <class Index>
struct Sink {
    virtual ~Sink() = default;
    virtual void push(const Event<Index>& e) = 0;
    virtual void flush() {}
};

template <class Index>
struct Source {
    virtual ~Source() = default;
    virtual std::optional<Event<Index>> next() = 0;
};

}} // namespace screamer::streams
#endif
