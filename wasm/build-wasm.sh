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

# Size pass (2026-08): extra safe emcc link flags, appended to every emcc
# invocation via the EMCC_CFLAGS env var the compiler driver honors -- this
# keeps CMakeLists.txt's EMCC_LINK untouched and confines the size-pass diff
# to this script. Each flag below was measured with the JS glue on
# (both `build:wasm` and `build:wasm:single`) and the full 66-test JS suite
# (`node --import tsx --test test/*.test.ts`) re-run after adding it; only
# flags that shrunk output AND kept the suite green were kept. See
# .superpowers/sdd/task-p6-4-report.md for the before/after gzip table.
#
#  -sINCOMING_MODULE_JS_API=[]  no caller ever passes a Module={} override
#                                into the factory (`m.default()` takes no
#                                arg, see js/src/loader.ts) so the glue code
#                                that reads incoming Module.* properties is
#                                dead weight; dropping it is a pure win.
#  -sDYNAMIC_EXECUTION=0         the glue never calls eval()/new Function()
#                                at runtime (this repo's own loader.ts does,
#                                but that's the *consumer's* JS, not code
#                                emitted into screamer.mjs/screamer.single.mjs)
#                                so the eval-fallback paths in the runtime
#                                are dead weight.
#
# Tried and dropped:
#  -sTEXTDECODER=2 / -sABORTING_MALLOC=0   built and passed the suite but
#                                measured a 0-byte gzip delta in both build
#                                modes (Closure/DCE already remove the
#                                affected paths here) -- dropped as dead
#                                weight rather than kept for a paper win.
#  -sMINIMAL_RUNTIME=1           saved a further ~1 KB gzip on the
#                                single-file build but broke the DEFAULT
#                                (separate-file) build: the ESM glue emits a
#                                bare `require(...)` call that has no
#                                binding in a pure `import()`'d ES module,
#                                so `node --import tsx --test` fails with
#                                "ReferenceError: require is not defined".
#                                This is the exact MODULARIZE/EXPORT_ES6
#                                incompatibility this task's brief warned
#                                MINIMAL_RUNTIME often triggers. Dropped.
#  -sSUPPORT_LONGJMP=0            NOT tried: the error path relies on real
#                                C++ exceptions crossing the wasm boundary
#                                (see js/src/loader.ts's wrapCtors / the
#                                "RollingMean(0) throws a RangeError (kernel
#                                invalid_argument)" test); disabling longjmp
#                                support risks that path, for a size lever
#                                this build doesn't need.
# -Wno-unused-command-line-argument: EMCC_CFLAGS is appended to every emcc
# invocation, including the per-.cpp compile steps that don't use these two
# link-only settings; without this, each compile step prints a harmless but
# noisy "linker setting ignored during compilation" warning.
export EMCC_CFLAGS="${EMCC_CFLAGS:-} -sINCOMING_MODULE_JS_API=[] -sDYNAMIC_EXECUTION=0 -Wno-unused-command-line-argument"

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
