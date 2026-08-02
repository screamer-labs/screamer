#!/usr/bin/env bash
# WASM spike build + binary-size measurement.
# Run from the repo root:  bash docs/superpowers/spikes/2026-08-02-wasm/build.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SPIKE_DIR="$REPO_ROOT/docs/superpowers/spikes/2026-08-02-wasm"
INC="$REPO_ROOT/include"
POLICY_CPP="$REPO_ROOT/src/screamer/detail/start_policy.cpp"
BUILD="$SPIKE_DIR/build"
mkdir -p "$BUILD"

# The REAL kernel's only non-header dependency: parse_start_policy(), a pure
# C++ .cpp from the screamer tree. We compile it, we do not reimplement it.
SRC="$SPIKE_DIR/spike.cpp $POLICY_CPP"

COMMON=(-std=c++17 -I "$INC" -lembind
        -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=node
        -sFILESYSTEM=0 -sALLOW_MEMORY_GROWTH=1)

echo "== emcc / node versions =="
emcc --version | head -1
node --version

# ---- 1. Runnable ES module for driver.mjs (all registrations, size-optimized).
echo "== build runnable module (SPIKE_MULTI, -Oz) =="
emcc $SRC -DSPIKE_MULTI -Oz "${COMMON[@]}" \
    -o "$SPIKE_DIR/screamer_spike.mjs"

# ---- 2. Size variants. We measure the .wasm only; JS glue is separate.
echo "== size variant builds =="

# One op (RollingMean alone), -O2
emcc $SRC -O2 "${COMMON[@]}" -o "$BUILD/one_O2.mjs"

# One op, -Oz
emcc $SRC -Oz "${COMMON[@]}" -o "$BUILD/one_Oz.mjs"

# One op, -Oz + standalone wasm-opt -Oz (extra pass on top of emcc's own)
WASM_OPT="$(dirname "$(dirname "$(readlink -f "$(which emcc)" || which emcc)")")/bin/wasm-opt"
if [ ! -x "$WASM_OPT" ]; then
    WASM_OPT="/Users/thijs/Projects/emsdk/upstream/bin/wasm-opt"
fi
if [ -x "$WASM_OPT" ]; then
    # --all-features: the wasm emcc emits uses bulk-memory + nontrapping
    # float-to-int; a standalone wasm-opt must be told to allow them.
    "$WASM_OPT" -Oz --all-features --strip-debug --strip-producers \
        "$BUILD/one_Oz.wasm" -o "$BUILD/one_Oz_wasmopt.wasm"
else
    echo "wasm-opt not found; skipping extra pass"
fi

# Multi op (RollingMean + 2 aliases + RollingSum), -Oz
emcc $SRC -DSPIKE_MULTI -Oz "${COMMON[@]}" -o "$BUILD/multi_Oz.mjs"

# ---- 3. Report sizes.
echo ""
echo "== .wasm sizes (bytes) =="
sz() { stat -f%z "$1" 2>/dev/null || stat -c%s "$1"; }
gz() { gzip -9 -c "$1" | wc -c | tr -d ' '; }

printf "%-34s %10s %10s\n" "variant" "raw" "gzip-9"
for f in \
    "$BUILD/one_O2.wasm" \
    "$BUILD/one_Oz.wasm" \
    "$BUILD/one_Oz_wasmopt.wasm" \
    "$BUILD/multi_Oz.wasm" \
    "$SPIKE_DIR/screamer_spike.wasm" ; do
    if [ -f "$f" ]; then
        printf "%-34s %10s %10s\n" "$(basename "$f")" "$(sz "$f")" "$(gz "$f")"
    fi
done
