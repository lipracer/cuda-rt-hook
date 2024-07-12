
#include <dlfcn.h>  // dladdr
#include <execinfo.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "backtrace.h"
#include "hook.h"
#include "hooks/print_hook.h"
#include "logger/StringRef.h"
#include "logger/logger.h"
#include "statistic.h"

void log_router() { LOG(WARN) << "dddddd"; }

#define DEF_WRAP_FUNC(index)                       \
    int __any_func_wrap__##index() {               \
        int ret = __dummy_any_function__##index(); \
        log_router();                              \
        return ret;                                \
    }

extern "C" {
DEF_WRAP_FUNC(0)
DEF_WRAP_FUNC(1)
DEF_WRAP_FUNC(2)
DEF_WRAP_FUNC(3)
DEF_WRAP_FUNC(4)
DEF_WRAP_FUNC(5)
DEF_WRAP_FUNC(6)
DEF_WRAP_FUNC(7)
DEF_WRAP_FUNC(8)
DEF_WRAP_FUNC(9)
DEF_WRAP_FUNC(10)
DEF_WRAP_FUNC(11)
DEF_WRAP_FUNC(12)
DEF_WRAP_FUNC(13)
DEF_WRAP_FUNC(14)
DEF_WRAP_FUNC(15)
DEF_WRAP_FUNC(16)
DEF_WRAP_FUNC(17)
DEF_WRAP_FUNC(18)
DEF_WRAP_FUNC(19)
DEF_WRAP_FUNC(20)
}

namespace {

struct AnyHook : public hook::HookInstallerWrap<AnyHook> {
    bool targetLib(const char* name) { return adt::StringRef(name) == ""; }

    bool targetSym(const char* name) {
        return adt::StringRef(name).contain("foo");
    }

    void* newFuncPtr(const hook::OriginalInfo& info) {
        old_func_ptr_ = info.oldFuncPtr;
        return reinterpret_cast<void*>(&__any_func_wrap__0);
    }
    void onSuccess() {}

    static AnyHook* instance() {
        static auto install_wrap = std::make_shared<AnyHook>();
        return install_wrap.get();
    }
    void* old_func_ptr_ = nullptr;
};

struct FillDummpyFunc : public hook::HookInstallerWrap<FillDummpyFunc> {
    bool targetLib(const char* name) {
        return adt::StringRef(name).contain("libxpu_mock_improve.so");
    }

    bool targetSym(const char* name) {
        return adt::StringRef(name).contain("__dummy_any_function__");
    }

    void* newFuncPtr(const hook::OriginalInfo& info) {
        return AnyHook::instance()->old_func_ptr_;
    }
    void onSuccess() {}

    static FillDummpyFunc* instance() {
        static auto install_wrap = std::make_shared<FillDummpyFunc>();
        return install_wrap.get();
    }

   private:
    std::unordered_map<std::string, void*> dummy_func_map_;
};

}  // namespace

void dh_any_hook_install() {
    AnyHook::instance()->install();
    FillDummpyFunc::instance()->install();
}
