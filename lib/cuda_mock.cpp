#include "cuda_mock.h"

#include <dlfcn.h>
#include <regex.h>
#include <string.h>

#include <csetjmp>
#include <unordered_set>

#include "GlobalVarMgr.h"
#include "backtrace.h"
#include "cuda_op_tracer.h"
#include "hook.h"
#include "logger/logger.h"

std::jmp_buf log_jump_buffer = {{}};

#ifdef __cplusplus

extern "C" {

void dh_initialize() {
    hook::init_all_global_variables();
    LOG(INFO) << "initialize";
    hook::HookInstaller hookInstaller = trace::getHookInstaller();
    hook::install_hook(hookInstaller);
    // hook::install_hook();
}

void dh_uninitialize() {}

static hook::GlobalVarMgr<void*> oldFuncAddr(nullptr);

void log_router() {
    LOG(INFO) << __func__ << ":" << *oldFuncAddr;
    // sometime will crash
    // trace::BackTraceCollection::CallStackInfo tracer({});
    // tracer.snapshot();
    // LOG(WARN) << tracer;
    longjmp(log_jump_buffer, 1);
}

void __any_mock_func__() {
// Conditional code for x86_64 architecture
#if defined(__x86_64__)
    // avoid inline asm, use varidic c func and parse func result
    // asm volatile("pop %rbp");
    // asm volatile("push %rax");
    // asm volatile("push %rdi");
    // if (!setjmp(log_jump_buffer)) {
    //     log_router();
    // }
    // asm volatile("pop %rdi");
    // asm volatile("pop %rax");
    // asm volatile("add    $0x8,%rsp");
    // asm volatile("jmp *%0" : : "r"(oldFuncAddr));

// Conditional code for aarch64 architecture
#else
    // asm volatile("push {r0}");
    // if (!setjmp(log_jump_buffer)) {
    //     log_router();
    // }
    // asm volatile("pop {r0}");
#endif
}

int builtin_printf(int flag, const char* fmt, va_list argp) {
    constexpr size_t kMax = 1024;
    char buf[kMax] = {"myprintf "};
    snprintf(buf + strlen(buf), kMax - strlen(buf), fmt, argp);
    LOG(INFO) << buf;
    return 0;
}

static hook::GlobalVarMgr<std::unordered_map<std::string, void*>> gBuiltinFuncs(
    std::unordered_map<std::string, void*>{
        std::make_pair("__printf_chk",
                       reinterpret_cast<void*>(&builtin_printf))});

void dh_internal_install_hook(const char* srcLib, const char* targetLib,
                              const char* symbolName, const char* hookerLibPath,
                              const char* hookerSymbolName) {
    LOG(INFO) << "initialize srcLib:" << srcLib << " targetLib:" << targetLib
              << " symbolName:" << symbolName;
    auto iter = gBuiltinFuncs->find(symbolName);
    auto hookerAddr = iter == gBuiltinFuncs->end()
                          ? reinterpret_cast<void*>(&__any_mock_func__)
                          : iter->second;
    if (hookerLibPath) {
        auto handle = dlopen(hookerLibPath, RTLD_LAZY);
        CHECK(handle, "can't not dlopen:{0} error info:{1}", hookerLibPath,
              dlerror());
        hookerAddr =
            dlsym(handle, hookerSymbolName ? hookerSymbolName : symbolName);
    }
    CHECK(hookerAddr, "hookerAddr can't be empty!");
    hook::HookInstaller hookInstaller =
        trace::getHookInstaller(trace::HookerInfo{.srcLib = srcLib,
                                                  .targeLib = targetLib,
                                                  .symbolName = symbolName,
                                                  .newFuncPtr = hookerAddr});
    auto newFuncPtr = hookInstaller.newFuncPtr;
    hookInstaller.newFuncPtr = [=](const hook::OriginalInfo& orgInfo) -> void* {
        auto handle = dlopen(hookerLibPath, RTLD_LAZY);
        CHECK(handle, "can't not dlopen {0} error info:{1}", hookerLibPath,
              dlerror());
        std::string org_symbol = "__origin_" + std::string(symbolName);
        auto org_addr = dlsym(handle, org_symbol.c_str());
        if (org_addr) {
            *reinterpret_cast<void**>(org_addr) = orgInfo.oldFuncPtr;
        } else {
            LOG(WARN) << "can't find symbol:" << org_symbol;
        }
        return newFuncPtr(orgInfo);
    };
    hookInstaller.onSuccess = [&]() {
        *oldFuncAddr =
            trace::CudaInfoCollection::instance().getSymbolAddr(symbolName);
        LOG(INFO) << __func__ << ":" << *oldFuncAddr;
    };
    hook::install_hook(hookInstaller);
}

struct DHRegexHook : public hook::HookInstallerWrap<DHRegexHook> {
    DHRegexHook(const char* srcLib, const char* targetLib,
                const char* symbolName, const char* hookerLibPath,
                const char* hookerSymbolName)
        : strs_(5) {
        strs_ = {srcLib, targetLib, symbolName, hookerLibPath,
                 hookerSymbolName};
        dynamic_obj_handle_ = dlopen(hookerLibPath, RTLD_LAZY);
        if (!dynamic_obj_handle_) {
            LOG(FATAL) << "can't open lib:" << hookerLibPath;
        }
    }
    bool targetLib(const char* name) {
        bool match =
            regex_match(strs_[1], name) && strcmp(name, strs_[3].c_str());
        return visited_.insert(name).second && match;
    }

    bool targetSym(const char* name) { return strs_[2] == name; }

    void* newFuncPtr(const hook::OriginalInfo& info) {
        std::string org_symbol = "__origin_" + std::string(curSymName());
        auto org_addr = dlsym(dynamic_obj_handle_, org_symbol.c_str());
        LOG(INFO) << "set origin function name:" << org_symbol
                  << " org_addr:" << org_addr;
        if (org_addr) {
            *reinterpret_cast<void**>(org_addr) = info.oldFuncPtr;
        } else {
        }
        void* new_func_ptr = nullptr;
        if (strs_.back().empty()) {
            LOG(INFO) << "dlsym lib:" << curLibName() << " symbol:" << strs_[2];
            new_func_ptr = dlsym(dynamic_obj_handle_, strs_[2].c_str());
        } else {
            LOG(INFO) << "dlsym lib:" << curLibName()
                      << " symbol:" << strs_.back();
            new_func_ptr = dlsym(dynamic_obj_handle_, strs_.back().c_str());
        }
        CHECK(new_func_ptr, "new func ptr is nullptr");
        return new_func_ptr;
    }
    void onSuccess() {}

    bool regex_match(const std::string& pattern, const char* str) {
        regex_t regex;
        int reti = 0;
        do {
            /* Compile regular expression */
            reti = regcomp(&regex, pattern.c_str(), 0);
            if (reti) {
                break;
            }
            /* Execute regular expression */
            reti = regexec(&regex, str, 0, nullptr, 0);
            if (reti == REG_NOMATCH) {
                return false;
            }
            if (0 == reti) {
                return true;
            }
        } while (0);
        return !reti;
    }

    ~DHRegexHook() {
        if (dynamic_obj_handle_) {
            dlclose(dynamic_obj_handle_);
        }
    }

   private:
    std::vector<std::string> strs_;
    void* dynamic_obj_handle_{nullptr};
    std::unordered_set<std::string> visited_;
};

void dh_internal_install_hook_regex(const char* srcLib, const char* targetLib,
                                    const char* symbolName,
                                    const char* hookerLibPath,
                                    const char* hookerSymbolName) {
    static std::vector<std::shared_ptr<DHRegexHook>> shared_ptr_vec;

    shared_ptr_vec.emplace_back(std::make_shared<DHRegexHook>(
        srcLib, targetLib, symbolName, hookerLibPath, hookerSymbolName));
    shared_ptr_vec.back()->install();
    LOG(INFO) << "dh_internal_install_hook_regex complete!";
}

struct DHPythonHook : public hook::HookInstallerWrap<DHPythonHook> {
    DHPythonHook(const std::function<bool(const char* name)>& isTarget,
                 const std::function<bool(const char* name)>& isSymbol,
                 const char* lib)
        : isTarget_(isTarget), isSymbol_(isSymbol) {
        dynamic_obj_handle_ = dlopen(lib, RTLD_LAZY);
        if (!dynamic_obj_handle_) {
            LOG(FATAL) << "can't open lib:" << lib;
        }
    }
    bool targetLib(const char* name) { return isTarget_(name); }

    bool targetSym(const char* name) { return isSymbol_(name); }

    void* newFuncPtr(const hook::OriginalInfo& info) {
        std::string org_symbol = "__origin_" + std::string(curSymName());
        auto org_addr = dlsym(dynamic_obj_handle_, org_symbol.c_str());
        LOG(INFO) << "set origin function name:" << org_symbol
                  << " org_addr:" << org_addr;
        if (org_addr) {
            *reinterpret_cast<void**>(org_addr) = info.oldFuncPtr;
        } else {
        }
        void* new_func_ptr = dlsym(dynamic_obj_handle_, curSymName());

        CHECK(new_func_ptr, "new func ptr is nullptr curSymName:{}",
              curSymName());
        return new_func_ptr;
    }
    void onSuccess() {}

    ~DHPythonHook() {
        if (dynamic_obj_handle_) {
            dlclose(dynamic_obj_handle_);
        }
    }

   private:
    std::function<bool(const char* name)> isTarget_;
    std::function<bool(const char* name)> isSymbol_;
    void* dynamic_obj_handle_{nullptr};
};

void dh_create_py_hook_installer(
    const std::function<bool(const char* name)>& isTarget,
    const std::function<bool(const char* name)>& isSymbol, const char* lib) {
    auto installer = std::make_shared<DHPythonHook>(isTarget, isSymbol, lib);
    installer->install();
}

void DhLibraryUnloader() __attribute__((destructor));

void DhLibraryUnloader() { logger::destroy_logger(); }
}

#endif
