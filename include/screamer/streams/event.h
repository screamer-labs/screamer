#ifndef SCREAMER_STREAMS_EVENT_H
#define SCREAMER_STREAMS_EVENT_H

#include <cstdint>
#include <optional>

namespace screamer { namespace streams {

template <class Key>
struct Event {
    Key key;
    double value;
    std::uint32_t source = 0;   // provenance tag; used by merge later
};

template <class Key>
struct Sink {
    virtual ~Sink() = default;
    virtual void push(const Event<Key>& e) = 0;
    virtual void flush() {}
};

template <class Key>
struct Source {
    virtual ~Source() = default;
    virtual std::optional<Event<Key>> next() = 0;
};

}} // namespace screamer::streams
#endif
