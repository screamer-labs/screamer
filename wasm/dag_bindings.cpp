// Embind DAG bindings: a separate translation unit from the generated point-op
// bindings. Registers GraphBuilder + CompiledGraph over the pure dag/ engine.
#include "dag_embind.h"

EMSCRIPTEN_BINDINGS(screamer_dag) { SCREAMER_REGISTER_DAG(); }
