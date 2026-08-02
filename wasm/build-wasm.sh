#!/usr/bin/env bash
# Build screamer.wasm (Embind point-op runtime) with emcc, then print its size.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/wasm/build"
rm -rf "$BUILD"; mkdir -p "$BUILD"
emcmake cmake -S "$ROOT/wasm" -B "$BUILD" -DSCREAMER_ROOT="$ROOT" >/dev/null
cmake --build "$BUILD" -j
sz() { stat -f%z "$1" 2>/dev/null || stat -c%s "$1"; }
gz() { gzip -9 -c "$1" | wc -c | tr -d ' '; }
echo "screamer.wasm raw=$(sz "$BUILD/screamer.wasm") gzip=$(gz "$BUILD/screamer.wasm")"
