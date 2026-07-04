#ifndef SCREAMER_DAG_SELECT_NODE_H
#define SCREAMER_DAG_SELECT_NODE_H

#include <cstddef>
#include <stdexcept>
#include <vector>
#include "screamer/dag/frame.h"

namespace screamer { namespace dag {

// Projects selected columns out of a wide frame, emitting a width-M frame
// (M = columns.size()) with the columns in the given order. Keys pass through;
// row count unchanged. Reuses one output buffer (zero per-event allocation).
template <class Key>
class SelectNode : public Sink<Key> {
public:
    SelectNode(std::vector<std::size_t> columns, Sink<Key>& downstream)
        : columns_(std::move(columns)), downstream_(downstream),
          out_(columns_.size()) {}

    void push(const Frame<Key>& f) override {
        for (std::size_t j = 0; j < columns_.size(); ++j) {
            if (columns_[j] >= f.width)
                throw std::runtime_error(
                    "dag::SelectNode: column index out of range for frame width");
            out_[j] = f.values[columns_[j]];
        }
        downstream_.push(Frame<Key>{f.key, out_.data(), out_.size()});
    }

    void flush() override { downstream_.flush(); }

private:
    std::vector<std::size_t> columns_;
    Sink<Key>& downstream_;
    std::vector<double> out_;   // reused every event
};

}} // namespace screamer::dag
#endif
