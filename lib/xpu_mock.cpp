#include "xpu_mock.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <vector>

#include "hook.h"
#include "logger/logger.h"

namespace {

class XpuMemoryApiHook;

class XpuMemoryWrapApi {
   public:
    static constexpr int kMaxXpuDeviceNum = 8;

    static XpuMemoryWrapApi& instance();
    XpuMemoryWrapApi();
    static int xpuMalloc(void** pDevPtr, uint64_t size, int kind);
    static int xpuFree(void* devPtr);

    std::function<int(void**, uint64_t, int)> raw_xpu_malloc_;
    std::function<int(void*)> raw_xpu_free_;
    std::function<int(int*)> raw_xpu_current_device_;

   private:
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

    friend class XpuMemoryApiHook;
};

XpuMemoryWrapApi& XpuMemoryWrapApi::instance() {
    static XpuMemoryWrapApi instance;
    return instance;
}

XpuMemoryWrapApi::XpuMemoryWrapApi()
    : allocated_ptr_map_(kMaxXpuDeviceNum),
      allocated_gm_size_(kMaxXpuDeviceNum, 0),
      allocated_l3_size_(kMaxXpuDeviceNum, 0),
      peak_gm_size_(kMaxXpuDeviceNum, 0),
      peak_l3_size_(kMaxXpuDeviceNum, 0) {}

int XpuMemoryWrapApi::xpuMalloc(void** pDevPtr, uint64_t size, int kind) {
    int r = 0;
    int devId = 0;

    CHECK(XpuMemoryWrapApi::instance().raw_xpu_current_device_,
          "xpu_current_device not binded");
    CHECK(XpuMemoryWrapApi::instance().raw_xpu_malloc_, "xpu_free not binded");

    // make malloc/free sequential to obtain a trusted memory usage footprint
    std::lock_guard<std::mutex> lock(
        XpuMemoryWrapApi::instance().memory_api_mutex_);

    r = XpuMemoryWrapApi::instance().raw_xpu_current_device_(&devId);
    if (r != 0) {
        return r;
    }
    CHECK_LT(devId, kMaxXpuDeviceNum,
             "devId({}) must less than kMaxXpuDeviceNum({})", devId,
             kMaxXpuDeviceNum);

    r = XpuMemoryWrapApi::instance().raw_xpu_malloc_(pDevPtr, size, kind);
    if (r != 0) {
        LOG(WARN) << "[XpuMemoryWrapApi xpuMalloc][failed] "
                  << "devId=" << devId << ","
                  << "size=" << size << ","
                  << "kind=" << kind << ","
                  << "gm_allocated="
                  << XpuMemoryWrapApi::instance().allocated_gm_size_[devId]
                  << ","
                  << "gm_peak="
                  << XpuMemoryWrapApi::instance().peak_gm_size_[devId];
        return r;
    }

    if (kind == (int)XpuMemKind::GLOBAL_MEMORY) {
        XpuMemoryWrapApi::instance().allocated_gm_size_[devId] += size;
        XpuMemoryWrapApi::instance().peak_gm_size_[devId] =
            std::max(XpuMemoryWrapApi::instance().peak_gm_size_[devId],
                     XpuMemoryWrapApi::instance().allocated_gm_size_[devId]);
    } else if (kind == (int)XpuMemKind::L3_MEMORY) {
        XpuMemoryWrapApi::instance().allocated_l3_size_[devId] += size;
        XpuMemoryWrapApi::instance().peak_l3_size_[devId] =
            std::max(XpuMemoryWrapApi::instance().peak_l3_size_[devId],
                     XpuMemoryWrapApi::instance().allocated_l3_size_[devId]);
    }

    XpuMemoryWrapApi::instance().allocated_ptr_map_[devId][*pDevPtr] = {
        *pDevPtr, size, (XpuMemKind)kind};

    LOG(WARN) << "[XpuMemoryWrapApi xpuMalloc][success] "
              << "devId=" << devId << ","
              << "size=" << size << ","
              << "kind=" << kind << ","
              << "gm_allocated="
              << XpuMemoryWrapApi::instance().allocated_gm_size_[devId] << ","
              << "gm_peak="
              << XpuMemoryWrapApi::instance().peak_gm_size_[devId];

    return r;
}

int XpuMemoryWrapApi::xpuFree(void* devPtr) {
    int r = 0;
    int devId = 0;

    CHECK(XpuMemoryWrapApi::instance().raw_xpu_current_device_,
          "xpu_current_device not binded");
    CHECK(XpuMemoryWrapApi::instance().raw_xpu_free_, "xpu_free not binded");

    // make malloc/free sequential to obtain a trusted memory usage footprint
    std::lock_guard<std::mutex> lock(
        XpuMemoryWrapApi::instance().memory_api_mutex_);

    r = XpuMemoryWrapApi::instance().raw_xpu_current_device_(&devId);
    if (r != 0) {
        return r;
    }
    CHECK_LT(devId, kMaxXpuDeviceNum,
             "devId({}) must less than kMaxXpuDeviceNum({})", devId,
             kMaxXpuDeviceNum);

    r = XpuMemoryWrapApi::instance().raw_xpu_free_(devPtr);
    if (r != 0) {
        return r;
    }

    auto it =
        XpuMemoryWrapApi::instance().allocated_ptr_map_[devId].find(devPtr);
    if (it == XpuMemoryWrapApi::instance().allocated_ptr_map_[devId].end()) {
        return r;
    }

    XpuDataPtr dataPtr = it->second;

    if (dataPtr.kind == XpuMemKind::GLOBAL_MEMORY) {
        XpuMemoryWrapApi::instance().allocated_gm_size_[devId] -= dataPtr.size;
    } else if (dataPtr.kind == XpuMemKind::L3_MEMORY) {
        XpuMemoryWrapApi::instance().allocated_l3_size_[devId] -= dataPtr.size;
    }

    XpuMemoryWrapApi::instance().allocated_ptr_map_[devId].erase(it);
    return r;
}

struct XpuMemoryApiHook : public hook::HookInstallerWrap<XpuMemoryApiHook> {
    bool targetLib(const char* name) {
        return !strstr(name, "libxpurt.so.1") && !strstr(name, "libxpurt.so");
    }

    bool targetSym(const char* name) {
        return strstr(name, "xpu_malloc") || strstr(name, "xpu_free") ||
               strstr(name, "xpu_current_device");
    }

    void* newFuncPtr(const hook::OriginalInfo& info) {
        if (strstr(curSymName(), "xpu_malloc")) {
            LOG(WARN) << "[XpuMemoryApiHook][xpu_malloc]:" << info.libName;
            if (!XpuMemoryWrapApi::instance().raw_xpu_malloc_) {
                XpuMemoryWrapApi::instance().raw_xpu_malloc_ =
                    std::bind(reinterpret_cast<int((*)(...))>(info.oldFuncPtr),
                              std::placeholders::_1, std::placeholders::_2,
                              std::placeholders::_3);
            }
            return reinterpret_cast<void*>(&XpuMemoryWrapApi::xpuMalloc);
        } else if (strstr(curSymName(), "xpu_free")) {
            LOG(WARN) << "[XpuMemoryApiHook][xpu_free]:" << info.libName;
            if (!XpuMemoryWrapApi::instance().raw_xpu_free_) {
                XpuMemoryWrapApi::instance().raw_xpu_free_ =
                    std::bind(reinterpret_cast<int((*)(...))>(info.oldFuncPtr),
                              std::placeholders::_1);
            }
            return reinterpret_cast<void*>(&XpuMemoryWrapApi::xpuFree);
        } else if (strstr(curSymName(), "xpu_current_device")) {
            LOG(WARN) << "[XpuMemoryApiHook][xpu_current_device]:"
                      << info.libName;
            if (!XpuMemoryWrapApi::instance().raw_xpu_current_device_) {
                XpuMemoryWrapApi::instance().raw_xpu_current_device_ =
                    std::bind(reinterpret_cast<int((*)(...))>(info.oldFuncPtr),
                              std::placeholders::_1);
            }
            return info.oldFuncPtr;
        }
        CHECK(0, "error name");
        return nullptr;
    }

    void onSuccess() {}
};

}  // namespace

extern "C" {

void xpu_dh_initialize() {
    static auto install_wrap = std::make_shared<XpuMemoryApiHook>();
    install_wrap->install();
}
}