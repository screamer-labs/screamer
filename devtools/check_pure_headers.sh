#!/usr/bin/env bash
# Prove operator headers compile with NO binding library present. This is the
# WASM-readiness invariant: op headers depend only on the pure compute base.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CXX="${CXX:-c++}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

cat > "$tmp/pure.cpp" <<'EOF'
// A ScreamerBase op and a FunctorBase op, compiled with no nanobind on the path.
#include "screamer/rolling_mean.h"       // ScreamerBase, overrides process_array_no_stride
#include "screamer/rolling_min_max.h"    // FunctorBase<_,1,2>
int main() {
    screamer::RollingMean a(3);
    screamer::RollingMinMax b(3);
    double in = 1.0, out2[2];
    a.eval(&in, out2);
    b.eval(&in, out2);
    a.reset(); b.reset();
    return 0;
}
EOF

"$CXX" -std=c++17 -I "$ROOT/include" -c "$tmp/pure.cpp" -o "$tmp/pure.o"
echo "OK: operator headers compile with no binding library."
