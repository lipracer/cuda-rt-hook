#include "xpu_mock.h"

#include <dlfcn.h>  // dladdr
#include <execinfo.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <vector>

#include "backtrace.h"
#include "hook.h"
#include "logger/logger.h"
#include "support.h"

namespace {

class XpuRuntimeApiHook;

class XpuRuntimeWrapApi {
   public:
    static constexpr int kMaxXpuDeviceNum = 8;

    static XpuRuntimeWrapApi& instance();
    XpuRuntimeWrapApi();
    static int xpuMalloc(void** pDevPtr, uint64_t size, int kind);
    static int xpuFree(void* devPtr);
    static int xpuWait(void* devStream);

   private:
    std::function<int(void**, uint64_t, int)> raw_xpu_malloc_;
    std::function<int(void*)> raw_xpu_free_;
    std::function<int(int*)> raw_xpu_current_device_;
    std::function<int(void*)> raw_xpu_wait_;

    enum class XpuMemKind { GLOBAL_MEMORY = 0, L3_MEMORY };

    struct XpuDataPtr {
        void* data_ptr;
        uint64_t size;
        XpuMemKind kind;
    };

    std::mutex memory_api_mutex_;
    std::vector<std::map<void*, XpuDataPtr>> allocated_ptr_map_;
    std::vector<uint64_t> allocated_gm_size_;
    std::vector<uint64_t> allocated_l3_size_;
    std::vector<uint64_t> peak_gm_size_;
    std::vector<uint64_t> peak_l3_size_;

    friend class XpuRuntimeApiHook;
};

XpuRuntimeWrapApi& XpuRuntimeWrapApi::instance() {
    static XpuRuntimeWrapApi instance;
    return instance;
}

XpuRuntimeWrapApi::XpuRuntimeWrapApi()
    : allocated_ptr_map_(kMaxXpuDeviceNum),
      allocated_gm_size_(kMaxXpuDeviceNum, 0),
      allocated_l3_size_(kMaxXpuDeviceNum, 0),
      peak_gm_size_(kMaxXpuDeviceNum, 0),
      peak_l3_size_(kMaxXpuDeviceNum, 0) {}

int XpuRuntimeWrapApi::xpuMalloc(void** pDevPtr, uint64_t size, int kind) {
    int r = 0;
    int devId = 0;

    CHECK(XpuRuntimeWrapApi::instance().raw_xpu_current_device_,
          "xpu_current_device not binded");
    CHECK(XpuRuntimeWrapApi::instance().raw_xpu_malloc_, "xpu_free not binded");

    // make malloc/free sequential to obtain a trusted memory usage footprint
    std::lock_guard<std::mutex> lock(
        XpuRuntimeWrapApi::instance().memory_api_mutex_);

    r = XpuRuntimeWrapApi::instance().raw_xpu_current_device_(&devId);
    if (r != 0) {
        return r;
    }
    CHECK_LT(devId, kMaxXpuDeviceNum,
             "devId({}) must less than kMaxXpuDeviceNum({})", devId,
             kMaxXpuDeviceNum);

    r = XpuRuntimeWrapApi::instance().raw_xpu_malloc_(pDevPtr, size, kind);
    if (r != 0) {
        LOG(WARN) << "[XpuRuntimeWrapApi xpuMalloc][failed] "
                  << "devId=" << devId << ","
                  << "size=" << size << ","
                  << "kind=" << kind << ","
                  << "gm_allocated="
                  << XpuRuntimeWrapApi::instance().allocated_gm_size_[devId]
                  << ","
                  << "gm_peak="
                  << XpuRuntimeWrapApi::instance().peak_gm_size_[devId];
        return r;
    }

    if (kind == (int)XpuMemKind::GLOBAL_MEMORY) {
        XpuRuntimeWrapApi::instance().allocated_gm_size_[devId] += size;
        XpuRuntimeWrapApi::instance().peak_gm_size_[devId] =
            std::max(XpuRuntimeWrapApi::instance().peak_gm_size_[devId],
                     XpuRuntimeWrapApi::instance().allocated_gm_size_[devId]);
    } else if (kind == (int)XpuMemKind::L3_MEMORY) {
        XpuRuntimeWrapApi::instance().allocated_l3_size_[devId] += size;
        XpuRuntimeWrapApi::instance().peak_l3_size_[devId] =
            std::max(XpuRuntimeWrapApi::instance().peak_l3_size_[devId],
                     XpuRuntimeWrapApi::instance().allocated_l3_size_[devId]);
    }

    XpuRuntimeWrapApi::instance().allocated_ptr_map_[devId][*pDevPtr] = {
        *pDevPtr, size, (XpuMemKind)kind};

    LOG(WARN) << "[XpuRuntimeWrapApi xpuMalloc][success] "
              << "devId=" << devId << ","
              << "size=" << size << ","
              << "kind=" << kind << ","
              << "gm_allocated="
              << XpuRuntimeWrapApi::instance().allocated_gm_size_[devId] << ","
              << "gm_peak="
              << XpuRuntimeWrapApi::instance().peak_gm_size_[devId];

    return r;
}

int XpuRuntimeWrapApi::xpuFree(void* devPtr) {
    int r = 0;
    int devId = 0;

    CHECK(XpuRuntimeWrapApi::instance().raw_xpu_current_device_,
          "xpu_current_device not binded");
    CHECK(XpuRuntimeWrapApi::instance().raw_xpu_free_, "xpu_free not binded");

    // make malloc/free sequential to obtain a trusted memory usage footprint
    std::lock_guard<std::mutex> lock(
        XpuRuntimeWrapApi::instance().memory_api_mutex_);

    r = XpuRuntimeWrapApi::instance().raw_xpu_current_device_(&devId);
    if (r != 0) {
        return r;
    }
    CHECK_LT(devId, kMaxXpuDeviceNum,
             "devId({}) must less than kMaxXpuDeviceNum({})", devId,
             kMaxXpuDeviceNum);

    r = XpuRuntimeWrapApi::instance().raw_xpu_free_(devPtr);
    if (r != 0) {
        return r;
    }

    auto it =
        XpuRuntimeWrapApi::instance().allocated_ptr_map_[devId].find(devPtr);
    if (it == XpuRuntimeWrapApi::instance().allocated_ptr_map_[devId].end()) {
        return r;
    }

    XpuDataPtr dataPtr = it->second;

    if (dataPtr.kind == XpuMemKind::GLOBAL_MEMORY) {
        XpuRuntimeWrapApi::instance().allocated_gm_size_[devId] -= dataPtr.size;
    } else if (dataPtr.kind == XpuMemKind::L3_MEMORY) {
        XpuRuntimeWrapApi::instance().allocated_l3_size_[devId] -= dataPtr.size;
    }

    XpuRuntimeWrapApi::instance().allocated_ptr_map_[devId].erase(it);
    return r;
}

int XpuRuntimeWrapApi::xpuWait(void* devStream) {
    constexpr size_t kMaxStackDeep = 512;
    void* call_stack[kMaxStackDeep] = {0};
    char** symbols = nullptr;
    int num = backtrace(call_stack, kMaxStackDeep);
    CHECK(num > 0, "Expect frams num {} > 0!", num);
    CHECK(num <= kMaxStackDeep, "Expect frams num {} <= 512!", num);
    symbols = backtrace_symbols(call_stack, num);
    if (symbols == nullptr) {
        return false;
    }

    LOG(WARN) << "[XpuRuntimeWrapApi xpuWait]"
              << "get stack deep num:" << num;
    Dl_info info;
    for (int j = 0; j < num; j++) {
        if (dladdr(call_stack[j], &info) && info.dli_sname) {
            auto demangled = __support__demangle(info.dli_sname);
            std::string path(info.dli_fname);
            LOG(WARN) << "    frame " << j << path << ":" << demangled;
        } else {
            // filtering useless print
            // LOG(WARN) << "    frame " << j << call_stack[j];
        }
    }
    free(symbols);

    return XpuRuntimeWrapApi::instance().raw_xpu_wait_(devStream);
    ;
}

struct XpuRuntimeApiHook : public hook::HookInstallerWrap<XpuRuntimeApiHook> {
    bool targetLib(const char* name) {
        return !strstr(name, "libxpurt.so.1") && !strstr(name, "libxpurt.so");
    }

    bool targetSym(const char* name) {
        return strstr(name, "xpu_malloc") || strstr(name, "xpu_free") ||
               strstr(name, "xpu_current_device") || strstr(name, "xpu_wait");
    }

    void* newFuncPtr(const hook::OriginalInfo& info) {
        if (strstr(curSymName(), "xpu_malloc")) {
            LOG(WARN) << "[XpuRuntimeApiHook][xpu_malloc]:" << info.libName;
            if (!XpuRuntimeWrapApi::instance().raw_xpu_malloc_) {
                XpuRuntimeWrapApi::instance().raw_xpu_malloc_ =
                    std::bind(reinterpret_cast<int((*)(...))>(info.oldFuncPtr),
                              std::placeholders::_1, std::placeholders::_2,
                              std::placeholders::_3);
            }
            return reinterpret_cast<void*>(&XpuRuntimeWrapApi::xpuMalloc);
        } else if (strstr(curSymName(), "xpu_free")) {
            LOG(WARN) << "[XpuRuntimeApiHook][xpu_free]:" << info.libName;
            if (!XpuRuntimeWrapApi::instance().raw_xpu_free_) {
                XpuRuntimeWrapApi::instance().raw_xpu_free_ =
                    std::bind(reinterpret_cast<int((*)(...))>(info.oldFuncPtr),
                              std::placeholders::_1);
            }
            return reinterpret_cast<void*>(&XpuRuntimeWrapApi::xpuFree);
        } else if (strstr(curSymName(), "xpu_current_device")) {
            LOG(WARN) << "[XpuRuntimeApiHook][xpu_current_device]:"
                      << info.libName;
            if (!XpuRuntimeWrapApi::instance().raw_xpu_current_device_) {
                XpuRuntimeWrapApi::instance().raw_xpu_current_device_ =
                    std::bind(reinterpret_cast<int((*)(...))>(info.oldFuncPtr),
                              std::placeholders::_1);
            }
            // simply use the original function ptr
            return info.oldFuncPtr;
        } else if (strstr(curSymName(), "xpu_wait")) {
            LOG(WARN) << "[XpuRuntimeApiHook][xpu_wait]:" << info.libName;
            if (!XpuRuntimeWrapApi::instance().raw_xpu_wait_) {
                XpuRuntimeWrapApi::instance().raw_xpu_wait_ =
                    std::bind(reinterpret_cast<int((*)(...))>(info.oldFuncPtr),
                              std::placeholders::_1);
            }
            return reinterpret_cast<void*>(&XpuRuntimeWrapApi::xpuWait);
        }
        CHECK(0, "capture wrong function: {}", curSymName());
        return nullptr;
    }

    void onSuccess() {}
};

}  // namespace

extern "C" {

void xpu_dh_initialize() {
    static auto install_wrap = std::make_shared<XpuRuntimeApiHook>();
    install_wrap->install();
}
}