#ifndef SCREAMER_SERIAL_CBOR_H
#define SCREAMER_SERIAL_CBOR_H

// The binary form of a save file: the same json::Value tree in CBOR
// (RFC 8949), for large states. The subset written and read:
//
//   null, false, true       major type 7 (0xf6, 0xf4, 0xf5)
//   an integer              major type 0 or 1, in its shortest form (a
//                           whole number past int64 that fits uint64 too)
//   a double                major type 7, float64 (0xfb + 8 bytes, raw bits),
//                           so NaN, the infinities and -0.0 keep their bits
//   a string                major type 3 (UTF-8 text)
//   an array, an object     major types 4 and 5; an object keeps its keys
//                           (text strings) in insertion order
//
// No tags, byte strings or indefinite lengths. The reader also takes the
// float16 and float32 forms of a double. A malformed input raises
// serial::SaveError "invalid CBOR: ..." with the byte offset.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "screamer/common/json.h"
#include "screamer/serial/errors.h"

namespace screamer { namespace serial { namespace cbor {

namespace detail {

inline void head(std::vector<std::uint8_t>& out, unsigned major, std::uint64_t n) {
    const auto m = static_cast<std::uint8_t>(major << 5);
    if (n < 24) {
        out.push_back(static_cast<std::uint8_t>(m | n));
        return;
    }
    int bytes = 8;
    std::uint8_t info = 27;
    if (n <= 0xFF) { bytes = 1; info = 24; }
    else if (n <= 0xFFFF) { bytes = 2; info = 25; }
    else if (n <= 0xFFFFFFFFULL) { bytes = 4; info = 26; }
    out.push_back(static_cast<std::uint8_t>(m | info));
    for (int k = bytes - 1; k >= 0; --k) out.push_back(static_cast<std::uint8_t>(n >> (8 * k)));
}

inline void text(std::vector<std::uint8_t>& out, const std::string& s) {
    head(out, 3, s.size());
    out.insert(out.end(), s.begin(), s.end());
}

inline void encode(std::vector<std::uint8_t>& out, const json::Value& v) {
    using K = json::Value::Kind;
    switch (v.kind) {
    case K::Null: out.push_back(0xf6); return;
    case K::Bool: out.push_back(v.b ? 0xf5 : 0xf4); return;
    case K::Int:
        if (v.i >= 0) head(out, 0, static_cast<std::uint64_t>(v.i));
        else head(out, 1, static_cast<std::uint64_t>(-(v.i + 1)));
        return;
    case K::Float: {
        // A whole number past int64 (json::Value::big_integer) that fits
        // uint64, such as a random generator's state, stays an integer;
        // any other is written as its double.
        std::uint64_t bits;
        if (v.whole && json::detail::parse_uint(v.s, bits)) {
            head(out, 0, bits);
            return;
        }
        std::memcpy(&bits, &v.d, sizeof bits);
        out.push_back(0xfb);
        for (int k = 7; k >= 0; --k) out.push_back(static_cast<std::uint8_t>(bits >> (8 * k)));
        return;
    }
    case K::String: text(out, v.s); return;
    case K::Array:
        head(out, 4, v.a.size());
        for (const json::Value& e : v.a) encode(out, e);
        return;
    case K::Object:
        head(out, 5, v.o.size());
        for (const auto& kv : v.o) {
            text(out, kv.first);
            encode(out, kv.second);
        }
        return;
    }
}

class Decoder {
public:
    Decoder(const std::uint8_t* data, std::size_t size) : d_(data), n_(size) {}

    json::Value document() {
        json::Value v = value(0);
        if (p_ != n_) fail("extra data after the value");
        return v;
    }

private:
    static constexpr int kMaxDepth = 512;
    const std::uint8_t* d_;
    std::size_t n_;
    std::size_t p_ = 0;

    [[noreturn]] void fail(const std::string& what) const {
        throw SaveError("invalid CBOR: " + what + " at byte " + std::to_string(p_));
    }

    std::uint8_t byte() {
        if (p_ >= n_) fail("truncated input");
        return d_[p_++];
    }

    std::uint64_t uint(int bytes) {
        if (n_ - p_ < static_cast<std::size_t>(bytes)) fail("truncated input");
        std::uint64_t u = 0;
        for (int k = 0; k < bytes; ++k) u = (u << 8) | d_[p_++];
        return u;
    }

    // The argument of a head whose low 5 bits are `info`.
    std::uint64_t argument(std::uint8_t info) {
        if (info < 24) return info;
        if (info == 24) return uint(1);
        if (info == 25) return uint(2);
        if (info == 26) return uint(4);
        if (info == 27) return uint(8);
        --p_;
        if (info == 31) fail("an indefinite length (not supported)");
        fail("reserved additional information " + std::to_string(info));
    }

    // A length that the rest of the input can hold (each item takes at
    // least one byte), so a corrupted length cannot ask for a huge buffer.
    std::size_t length(std::uint64_t n) {
        if (n > n_ - p_) fail("truncated input");
        return static_cast<std::size_t>(n);
    }

    std::string text(std::uint64_t n) {
        const std::size_t len = length(n);
        const std::uint8_t* t = d_ + p_;
        if (!valid_utf8(t, len)) fail("invalid UTF-8 in a text string");
        std::string s(reinterpret_cast<const char*>(t), len);
        p_ += len;
        return s;
    }

    // Well-formed UTF-8 (RFC 3629): no overlong forms, surrogates or code
    // points past U+10FFFF.
    static bool valid_utf8(const std::uint8_t* t, std::size_t n) {
        for (std::size_t k = 0; k < n;) {
            const std::uint8_t c = t[k];
            if (c < 0x80) { ++k; continue; }
            std::size_t len;
            std::uint32_t cp, min;
            if ((c & 0xE0) == 0xC0) { len = 2; cp = c & 0x1F; min = 0x80; }
            else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; min = 0x800; }
            else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; min = 0x10000; }
            else return false;
            if (n - k < len) return false;
            for (std::size_t j = 1; j < len; ++j) {
                if ((t[k + j] & 0xC0) != 0x80) return false;
                cp = (cp << 6) | (t[k + j] & 0x3F);
            }
            if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp < 0xE000)) return false;
            k += len;
        }
        return true;
    }

    static double half(std::uint16_t h) {
        const int e = (h >> 10) & 0x1F;
        const int m = h & 0x3FF;
        double v;
        if (e == 0) v = std::ldexp(m, -24);
        else if (e == 31) v = m ? std::nan("") : HUGE_VAL;
        else v = std::ldexp(m + 1024, e - 25);
        return (h & 0x8000) ? -v : v;
    }

    json::Value value(int depth) {
        if (depth > kMaxDepth) fail("values nested too deeply");
        const std::size_t start = p_;
        const std::uint8_t ib = byte();
        const unsigned major = ib >> 5;
        const std::uint8_t info = ib & 0x1F;
        if (major == 7) {
            switch (info) {
            case 20: return json::Value::boolean(false);
            case 21: return json::Value::boolean(true);
            case 22: return json::Value::null();
            case 25: return json::Value::number(half(static_cast<std::uint16_t>(uint(2))));
            case 26: {
                const auto u = static_cast<std::uint32_t>(uint(4));
                float f;
                std::memcpy(&f, &u, sizeof f);
                return json::Value::number(f);
            }
            case 27: {
                const std::uint64_t u = uint(8);
                double d;
                std::memcpy(&d, &u, sizeof d);
                return json::Value::number(d);
            }
            default:
                p_ = start;
                if (info == 31) fail("an indefinite-length break (not supported)");
                fail("an unsupported simple value " + std::to_string(info));
            }
        }
        if (major == 6) { p_ = start; fail("a tag (not supported)"); }
        const std::uint64_t n = argument(info);
        switch (major) {
        case 0:
            if (n > 9223372036854775807ULL)
                return json::Value::big_integer(static_cast<double>(n), json::detail::uint_text(n));
            return json::Value::integer(static_cast<std::int64_t>(n));
        case 1:   // -1 - n; below int64 (never written here) it reads as its double
            if (n > 9223372036854775807ULL) return json::Value::number(-1.0 - static_cast<double>(n));
            return json::Value::integer(-1 - static_cast<std::int64_t>(n));
        case 2: p_ = start; fail("a byte string (not supported)");
        case 3: return json::Value::string(text(n));
        case 4: {
            json::Value a = json::Value::array();
            const std::size_t len = length(n);
            // Each element is far larger than its one byte of input, so a
            // corrupted length reserves little; the vector grows as it reads.
            a.a.reserve(std::min<std::size_t>(len, 4096));
            for (std::size_t k = 0; k < len; ++k) a.a.push_back(value(depth + 1));
            return a;
        }
        default: {   // 5
            json::Value o = json::Value::object();
            const std::size_t len = length(n);
            // A later duplicate key replaces the earlier value, as in the
            // JSON reader.
            std::unordered_map<std::string, std::size_t> at;
            for (std::size_t k = 0; k < len; ++k) {
                const std::size_t key_at = p_;
                const std::uint8_t kb = byte();
                if ((kb >> 5) != 3) { p_ = key_at; fail("a key that is not a text string"); }
                std::string key = text(argument(kb & 0x1F));
                json::Value item = value(depth + 1);
                const auto ins = at.emplace(key, o.o.size());
                if (ins.second) o.o.emplace_back(std::move(key), std::move(item));
                else o.o[ins.first->second].second = std::move(item);
            }
            return o;
        }
        }
    }
};

}  // namespace detail

// The CBOR bytes of a tree.
inline std::vector<std::uint8_t> encode(const json::Value& v) {
    std::vector<std::uint8_t> out;
    detail::encode(out, v);
    return out;
}

// The tree of CBOR bytes; raises serial::SaveError on malformed input.
inline json::Value decode(const std::uint8_t* data, std::size_t size) {
    return detail::Decoder(data, size).document();
}

}}}  // namespace screamer::serial::cbor

#endif  // SCREAMER_SERIAL_CBOR_H
