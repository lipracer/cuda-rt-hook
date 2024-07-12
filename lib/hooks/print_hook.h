#include <cstdarg>
#include <string>
#include <vector>

#include "hook.h"
#include "logger/StringRef.h"

struct XpuRuntimePrintfHook
    : public hook::HookInstallerWrap<XpuRuntimePrintfHook> {
    void save_to_internel_buffer(const char* buf) {
        if (do_capture_) {
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

#define DEF_DUMMY_FUNC(index) int __dummy_any_function__##index();

extern "C" {
DEF_DUMMY_FUNC(0)
DEF_DUMMY_FUNC(1)
DEF_DUMMY_FUNC(2)
DEF_DUMMY_FUNC(3)
DEF_DUMMY_FUNC(4)
DEF_DUMMY_FUNC(5)
DEF_DUMMY_FUNC(6)
DEF_DUMMY_FUNC(7)
DEF_DUMMY_FUNC(8)
DEF_DUMMY_FUNC(9)
DEF_DUMMY_FUNC(10)
DEF_DUMMY_FUNC(11)
DEF_DUMMY_FUNC(12)
DEF_DUMMY_FUNC(13)
DEF_DUMMY_FUNC(14)
DEF_DUMMY_FUNC(15)
DEF_DUMMY_FUNC(16)
DEF_DUMMY_FUNC(17)
DEF_DUMMY_FUNC(18)
DEF_DUMMY_FUNC(19)
DEF_DUMMY_FUNC(20)
}
