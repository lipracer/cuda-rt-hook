#include <cstdarg>
#include <vector>
#include <string>

#include "hook.h"
#include "logger/StringRef.h"

struct XpuRuntimePrintfHook
    : public hook::HookInstallerWrap<XpuRuntimePrintfHook> {

    void save_to_internel_buffer(const char *buf){
        if(do_capture_){
            ss_ << buf;
        }
    }

    void setTargetLibs(const std::vector<adt::StringRef>& libs) {
        target_libs_ = libs;
    }
    void setTargetSymbols(const std::vector<adt::StringRef>& syms) {
        target_syms_ = syms;
    }

    bool targetLib(const char* name);

    bool targetSym(const char* name);

    void* newFuncPtr(const hook::OriginalInfo& info);

    void onSuccess() {}

    static XpuRuntimePrintfHook* instance() {
        static auto install_wrap = std::make_shared<XpuRuntimePrintfHook>();
        return install_wrap.get();
    }

    void start_capture() { do_capture_ = true; }
    std::string end_capture();
    bool do_capture_{false};
    std::stringstream ss_;
    void* py_obj_{nullptr};
    std::vector<adt::StringRef> target_libs_;
    std::vector<adt::StringRef> target_syms_;
};