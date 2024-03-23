#include <cstdarg>
#include <vector>
#include <string>

#include "hook.h"
#include "logger/StringRef.h"
#include "logger/logger.h"
#include "print_hook.h"

#define __internal_printf()                       \
    va_list args;                                 \
    constexpr size_t kMax = 4096;                 \
    char buf[kMax] = {0};                         \
    va_start(args, fmt);                          \
    vsprintf(buf, fmt, args);                     \
    va_end(args);                                 \
    XpuRuntimePrintfHook::instance()->save_to_internel_buffer(buf); \
    printf("%s", buf);


static int builtin_printf_chk(int flag, const char* fmt, ...) {
    __internal_printf();
    return 0;
}

static int builtin_printf(const char* fmt, ...) {
    __internal_printf();
    return 0;
}

static int builtin_fprintf(void* stdcout, const char* fmt, ...) {
    __internal_printf();
    return 0;
}

#undef __internal_printf

bool XpuRuntimePrintfHook::targetLib(const char* name) {
    for (const auto& lib : target_libs_) {
        if (adt::StringRef(name).contain(lib)) {
            return true;
        }
    }
    return false;
}

bool XpuRuntimePrintfHook::targetSym(const char* name) {
    for (const auto& sym : target_syms_) {
        if (adt::StringRef(name) == sym) {
            return true;
        }
    }
    return false;
}

void* XpuRuntimePrintfHook::newFuncPtr(const hook::OriginalInfo& info) {
    if (adt::StringRef("__printf_chk") == curSymName()) {
        return reinterpret_cast<void*>(&builtin_printf_chk);
    } else if (adt::StringRef("printf") == curSymName()) {
        return reinterpret_cast<void*>(&builtin_printf);
    } else if (adt::StringRef("fprintf") == curSymName() ||
                adt::StringRef("__fprintf") == curSymName() ||
                adt::StringRef("vfprintf") == curSymName()) {
        return reinterpret_cast<void*>(&builtin_fprintf);
    }
    return nullptr;
}

std::string XpuRuntimePrintfHook::end_capture() {
    do_capture_ = false;
    auto str = ss_.str();
    ss_.str("");
    return str;
}




void __print_hook_initialize(std::vector<adt::StringRef> &target_libs, std::vector<adt::StringRef> &target_symbols){
    XpuRuntimePrintfHook::instance()->setTargetLibs(target_libs);
    XpuRuntimePrintfHook::instance()->setTargetSymbols(target_symbols);
    XpuRuntimePrintfHook::instance()->install(); //replace plt table
    
}

void __print_hook_start_capture() {
    XpuRuntimePrintfHook::instance()->start_capture();
}

std::string __print_hook_end_capture() {
    return XpuRuntimePrintfHook::instance()->end_capture();
}
