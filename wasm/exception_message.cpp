// The message of a C++ exception that reached JavaScript, without a demangler.
//
// This build uses native WebAssembly exceptions (-fwasm-exceptions), and
// js/src/errors.ts reads a thrown exception's what() text with Emscripten's
// getExceptionMessage helper. That helper calls __get_exception_message from
// libc++abi, which demangles the type name and links the whole C++ demangler
// (about 59 KB of the module). These two definitions replace the ones in
// libc++abi's cxa_exception_js_utils.cpp: they keep its contract but return
// the type's mangled name. Both are defined so the linker never pulls that
// archive member, which would define them a second time.
#include <cxxabi.h>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <typeinfo>
#include <unwind.h>

namespace {
char* copy(const char* s) {
    const std::size_t n = std::strlen(s) + 1;
    char* r = static_cast<char*>(std::malloc(n));
    std::memcpy(r, s, n);
    return r;
}
}  // namespace

extern "C" {

// The thrown object sits right after its _Unwind_Exception header.
void* __thrown_object_from_unwind_exception(_Unwind_Exception* unwind_exception) {
    return unwind_exception + 1;
}

// Sets `type` to the thrown type's mangled name and `message` to what() when
// the object is a std::exception, otherwise to NULL. The caller frees both.
// Rethrowing the object (as a dependent exception that holds a reference to
// it) lets the C++ runtime decide whether it is a std::exception, so no layout
// of the exception header is assumed; the catch releases that reference.
void __get_exception_message(void* thrown_object, char** type, char** message) {
    *message = nullptr;
    try {
        __cxxabiv1::__cxa_rethrow_primary_exception(thrown_object);
    } catch (const std::exception& e) {
        *type = copy(typeid(e).name());
        *message = copy(e.what());
    } catch (...) {
        *type = copy("unknown");
    }
}

}  // extern "C"
