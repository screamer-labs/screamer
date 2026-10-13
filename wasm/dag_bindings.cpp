// Embind DAG bindings: a separate translation unit from the generated point-op
// bindings. Registers GraphBuilder + CompiledGraph over the pure dag/ engine.
#include "dag_embind.h"
#include <emscripten/em_macros.h>

EMSCRIPTEN_BINDINGS(screamer_dag) { SCREAMER_REGISTER_DAG(); }

// A live session's per-event pushes on the CompiledGraphWrap at address `g`
// (CompiledGraphWrap::address), as plain exports: JavaScript calls
// Module._screamer_push_event(g, ...) with no Embind argument conversion. A
// C++ exception reaches JavaScript as it does from an Embind method.
extern "C" {
EMSCRIPTEN_KEEPALIVE void screamer_push_event(std::uintptr_t g, std::uint32_t input,
                                              double index, double value) {
    reinterpret_cast<screamer_wasm::CompiledGraphWrap*>(g)->pushEvent(input, index, value);
}
EMSCRIPTEN_KEEPALIVE void screamer_push_event_wide(std::uintptr_t g, std::uint32_t input,
                                                   double index, std::uintptr_t values,
                                                   std::uint32_t width) {
    reinterpret_cast<screamer_wasm::CompiledGraphWrap*>(g)->pushEventWide(input, index, values,
                                                                          width);
}
}  // extern "C"
