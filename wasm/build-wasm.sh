#!/usr/bin/env bash
# Build screamer.wasm (Embind point-op runtime) with emcc, then print its size.
#
# Usage:
#   build-wasm.sh                 default: separate screamer.mjs + screamer.wasm
#   build-wasm.sh --single-file   screamer.single.mjs with the wasm embedded as
#                                  base64 (closure-minified) -- used to make the
#                                  published npm package self-contained. Requires
#                                  a `java` on PATH for the Closure Compiler pass.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/wasm/build"

SINGLE_FILE=0
for arg in "$@"; do
  case "$arg" in
    --single-file) SINGLE_FILE=1 ;;
    *) echo "build-wasm.sh: unknown argument: $arg" >&2; exit 1 ;;
  esac
done

CMAKE_ARGS=(-S "$ROOT/wasm" -B "$BUILD" -DSCREAMER_ROOT="$ROOT")
if [ "$SINGLE_FILE" -eq 1 ]; then
  CMAKE_ARGS+=(-DSCREAMER_WASM_SINGLE_FILE=ON)
fi

rm -rf "$BUILD"; mkdir -p "$BUILD"
emcmake cmake "${CMAKE_ARGS[@]}" >/dev/null
cmake --build "$BUILD" -j
sz() { stat -f%z "$1" 2>/dev/null || stat -c%s "$1"; }
gz() { gzip -9 -c "$1" | wc -c | tr -d ' '; }

if [ "$SINGLE_FILE" -eq 1 ]; then
  echo "screamer.single.mjs raw=$(sz "$BUILD/screamer.single.mjs") gzip=$(gz "$BUILD/screamer.single.mjs")"
else
  echo "screamer.wasm raw=$(sz "$BUILD/screamer.wasm") gzip=$(gz "$BUILD/screamer.wasm")"
fi
