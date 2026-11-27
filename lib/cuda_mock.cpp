#include "cuda_mock.h"

#include <Python.h>
#include <dlfcn.h>
#include <regex.h>
#include <string.h>

#include <csetjmp>
#include <unordered_set>

#include "GlobalVarMgr.h"
#include "backtrace.h"
#include "cuda_op_tracer.h"
#include "env_mgr.h"
#include "hook.h"
#include "logger/logger.h"
#include "xpu_mock.h"

std::jmp_buf log_jump_buffer = {{}};

#ifdef __cplusplus

extern "C" {

void dh_initialize() {
    hook::init_all_global_variables();
    // hook::HookInstaller hookInstaller = trace::getHookInstaller();
    // hook::install_hook(hookInstaller);
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

using SymbolReplacements = std::unordered_map<std::string, std::string>;

struct DHRegexHook : public hook::HookInstallerWrap<DHRegexHook> {
    DHRegexHook(const char* targetLib, const char* hookerLibPath,
                SymbolReplacements replacements)
        : path_(hookerLibPath), replacements_(std::move(replacements)) {
        dynamic_obj_handle_ = dlopen(hookerLibPath, RTLD_LAZY);
        if (!dynamic_obj_handle_) {
            LOG(FATAL) << "can't open lib:" << hookerLibPath;
        }
        pattern_ready_ = regcomp(&pattern_, targetLib, REG_NOSUB) == 0;
        if (!pattern_ready_) {
            LOG(WARN) << "illegal regex pattern:" << targetLib;
        }
    }

    bool targetLib(const char* name) {
        return pattern_ready_ && path_ != name &&
               regexec(&pattern_, name, 0, nullptr, 0) == 0 &&
               visited_.insert(name).second;
    }

    bool targetSym(const char* name) {
        return replacements_.find(name) != replacements_.end();
    }

    void* newFuncPtr(const hook::OriginalInfo& info) {
        const std::string name(curSymName());
        const std::string org_symbol = "__origin_" + name;
        if (auto org_addr = dlsym(dynamic_obj_handle_, org_symbol.c_str())) {
            *reinterpret_cast<void**>(org_addr) = info.oldFuncPtr;
        }
        const auto& replacement = replacements_.at(name);
        void* new_func_ptr =
            dlsym(dynamic_obj_handle_,
                  replacement.empty() ? name.c_str() : replacement.c_str());
        CHECK(new_func_ptr, "replacement symbol not found for {}", name);
        return new_func_ptr;
    }

    void onSuccess() {}

    ~DHRegexHook() {
        orgInfos.clear();
        if (pattern_ready_) {
            regfree(&pattern_);
        }
        if (dynamic_obj_handle_) {
            dlclose(dynamic_obj_handle_);
        }
    }

   private:
    std::string path_;
    SymbolReplacements replacements_;
    regex_t pattern_{};
    bool pattern_ready_{false};
    void* dynamic_obj_handle_{nullptr};
    std::unordered_set<std::string> visited_;
};

static std::shared_ptr<DHRegexHook> keep_regex_hook(
    const char* targetLib, const char* hookerLibPath,
    SymbolReplacements replacements) {
    static std::vector<std::shared_ptr<DHRegexHook>> hooks;
    auto installer = std::make_shared<DHRegexHook>(targetLib, hookerLibPath,
                                                   std::move(replacements));
    hooks.push_back(installer);
    return installer;
}

void dh_internal_install_hook_regex(const char* srcLib, const char* targetLib,
                                    const char* symbolName,
                                    const char* hookerLibPath,
                                    const char* hookerSymbolName) {
    keep_regex_hook(targetLib, hookerLibPath, {{symbolName, hookerSymbolName}})
        ->install();
}

void dh_runtimeapi_hook_initialize_with_hooks(
    const char* targetLib, const char* hookerLibPath,
    const HookString_t* symbols, const HookString_t* replacementNames) {
    auto runtime = runtimeapi_hook_installer();
    SymbolReplacements replacements;
    for (size_t i = 0; symbols[i]; ++i) {
        replacements.emplace(symbols[i], replacementNames[i]);
    }
    if (replacements.empty()) {
        hook::install_hook(runtime);
        return;
    }
    auto extra =
        keep_regex_hook(targetLib, hookerLibPath, std::move(replacements));
    hook::install_hooks({runtime, extra->buildInstaller()});
}

struct DHPythonHook : public hook::HookInstallerWrap<DHPythonHook> {
    DHPythonHook(
        const std::function<bool(HookString_t name)>& isTarget,
        const std::function<bool(HookString_t name)>& isSymbol,
        HookString_t lib,
        const std::function<HookString_t(HookString_t name)>& newSymbol)
        : isTarget_(isTarget), isSymbol_(isSymbol), newSymbol_(newSymbol) {
        LOG(INFO) << "DHPythonHook new lib name:" << lib;
        dynamic_obj_handle_ = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
        if (!dynamic_obj_handle_) {
            LOG(FATAL) << "can't open lib:" << lib;
        }
    }
    bool targetLib(const char* name) { return isTarget_(name); }

    bool targetSym(const char* name) { return isSymbol_(name); }

    void* newFuncPtr(const hook::OriginalInfo& info) {
        std::string org_symbol = "__origin_" + std::string(curSymName());
        auto org_addr = dlsym(dynamic_obj_handle_, org_symbol.c_str());

        if (org_addr) {
            *reinterpret_cast<void**>(org_addr) = info.oldFuncPtr;
        } else {
        }
        auto newSymName = newSymbol_(curSymName());
        void* new_func_ptr = dlsym(dynamic_obj_handle_, newSymName);
        LOG(INFO) << "set origin function name:" << org_symbol
                  << " org_addr:" << info.oldFuncPtr
                  << " newSymName:" << newSymName
                  << " new function address:" << new_func_ptr;
        CHECK(new_func_ptr,
              "new func ptr is nullptr curSymName:{} newSymName:{}",
              curSymName(), newSymName);
        return new_func_ptr;
    }
    void onSuccess() {}

    ~DHPythonHook() {
        if (dynamic_obj_handle_) {
            dlclose(dynamic_obj_handle_);
        }
    }

   private:
    std::function<bool(HookString_t name)> isTarget_;
    std::function<bool(HookString_t name)> isSymbol_;
    std::function<HookString_t(HookString_t name)> newSymbol_;
    void* dynamic_obj_handle_{nullptr};
};

void dh_create_py_hook_installer(
    const std::function<bool(HookString_t name)>& isTarget,
    const std::function<bool(HookString_t name)>& isSymbol, HookString_t lib,
    const std::function<HookString_t(HookString_t name)>& newSymbol) {
    auto installer =
        std::make_shared<DHPythonHook>(isTarget, isSymbol, lib, newSymbol);
    installer->install();
}

void DhLibraryLoader() __attribute__((constructor));

void DhLibraryLoader() {
    static bool sync = hook::get_env_value<bool>(env_mgr::LOG_SYNC_MODE);
    logger::LogConfig cfg;
    if (sync) {
        cfg.mode = logger::LogConfig::kSync;
    }
    logger::initLogger(cfg);
}

void DhLibraryUnloader() __attribute__((destructor));

void DhLibraryUnloader() { logger::destroy_logger(); }

const char* get_build_python_string_version() { return PY_VERSION; }

int get_build_python_int_version() { return PY_VERSION_HEX; }
}

#endif
