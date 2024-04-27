#include "xpu_mock.h"

#include <Python.h>
#include <dlfcn.h>  // dladdr
#include <execinfo.h>
#include <frameobject.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <vector>

#include "backtrace.h"
#include "hook.h"
#include "logger/StringRef.h"
#include "logger/logger.h"
#include "statistic.h"
#include "support.h"
#include "hooks/print_hook.h"

namespace {

class XpuRuntimeApiHook;

static constexpr int kMaxXpuDeviceNum = 8;

class XpuRuntimeWrapApi {
   public:
    static XpuRuntimeWrapApi& instance();
    XpuRuntimeWrapApi();
    static int xpuMalloc(void** pDevPtr, uint64_t size, int kind);
    static int xpuFree(void* devPtr);
    static int xpuWait(void* devStream);
    static int xpuMemcpy(void* dst, const void* src, uint64_t size, int kind);
    static int xpuSetDevice(int devId);
    static int xpuCurrentDeviceId(int* devIdPtr);

   private:
    decltype(&xpuMalloc) *raw_xpu_malloc_{nullptr};
    decltype(&xpuFree) *raw_xpu_free_{nullptr};
    decltype(&xpuWait) raw_xpu_wait_{nullptr};
    decltype(&xpuMemcpy) raw_xpu_memcpy_{nullptr};
    decltype(&xpuSetDevice) raw_xpu_set_device_id_{nullptr};
    decltype(&xpuCurrentDeviceId) raw_xpu_current_device_{nullptr};

    friend class XpuRuntimeApiHook;
};

XpuRuntimeWrapApi& XpuRuntimeWrapApi::instance() {
    static XpuRuntimeWrapApi instance;
    return instance;
}

XpuRuntimeWrapApi::XpuRuntimeWrapApi() {}

int XpuRuntimeWrapApi::xpuMalloc(void** pDevPtr, uint64_t size, int kind) {
    IF_ENABLE_LOG_TRACE(__func__);
    int r = 0;
    int devId = 0;

    CHECK(XpuRuntimeWrapApi::instance().raw_xpu_current_device_,
          "xpu_current_device not binded");
    CHECK(XpuRuntimeWrapApi::instance().raw_xpu_malloc_, "xpu_free not binded");

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
                  << "kind=" << kind;
        return r;
    }

    hook::MemoryStatisticCollection::instance().record_alloc(
        hook::HookRuntimeContext::instance().curLibName(), devId, *pDevPtr,
        size, kind);

    return r;
}

int XpuRuntimeWrapApi::xpuFree(void* devPtr) {
    IF_ENABLE_LOG_TRACE(__func__);
    int r = 0;
    int devId = 0;

    CHECK(XpuRuntimeWrapApi::instance().raw_xpu_current_device_,
          "xpu_current_device not binded");
    CHECK(XpuRuntimeWrapApi::instance().raw_xpu_free_, "xpu_free not binded");

    r = XpuRuntimeWrapApi::instance().raw_xpu_current_device_(&devId);
    if (r != 0) {
        return r;
    }
    CHECK_LT(devId, kMaxXpuDeviceNum,
             "devId({}) must less than kMaxXpuDeviceNum({})", devId,
             kMaxXpuDeviceNum);

    r = XpuRuntimeWrapApi::instance().raw_xpu_free_(devPtr);

    hook::MemoryStatisticCollection::instance().record_free(
        hook::HookRuntimeContext::instance().curLibName(), devId, devPtr);

    if (r != 0) {
        return r;
    }

    return r;
}

int XpuRuntimeWrapApi::xpuWait(void* devStream) {
    IF_ENABLE_LOG_TRACE(__func__);
    return XpuRuntimeWrapApi::instance().raw_xpu_wait_(devStream);
}

int XpuRuntimeWrapApi::xpuMemcpy(void* dst, const void* src, uint64_t size,
                                 int kind) {
    IF_ENABLE_LOG_TRACE(__func__);
    return XpuRuntimeWrapApi::instance().raw_xpu_memcpy_(dst, src, size, kind);
}

int XpuRuntimeWrapApi::xpuSetDevice(int devId) {
    IF_ENABLE_LOG_TRACE(__func__);
    MLOG(PROFILE, INFO) << "[XpuRuntimeWrapApi xpuCurrentDeviceId] "
                        << "devId=" << devId;
    return XpuRuntimeWrapApi::instance().raw_xpu_set_device_id_(devId);
}

int XpuRuntimeWrapApi::xpuCurrentDeviceId(int* devIdPtr) {
    IF_ENABLE_LOG_TRACE(__func__);
    int ret = XpuRuntimeWrapApi::instance().raw_xpu_current_device_(devIdPtr);
    MLOG(PROFILE, INFO) << "[XpuRuntimeWrapApi xpuCurrentDeviceId] "
                        << "devId=" << *devIdPtr;
    return ret;
}

class XpuRuntimeApiHook : public hook::HookInstallerWrap<XpuRuntimeApiHook> {
   public:
    bool targetLib(const char* name) {
        return !strstr(name, "libxpurt.so.1") && !strstr(name, "libxpurt.so");
    }

    hook::HookFeature symbols[6] = {
        // malloc
        hook::HookFeature("xpu_malloc", &XpuRuntimeWrapApi::xpuMalloc,
                          &XpuRuntimeWrapApi::instance().raw_xpu_malloc_),
        // free
        hook::HookFeature("xpu_free", &XpuRuntimeWrapApi::xpuFree,
                          &XpuRuntimeWrapApi::instance().raw_xpu_free_),
        // get device id
        hook::HookFeature(
            "xpu_current_device",
            &XpuRuntimeWrapApi::instance().xpuCurrentDeviceId,
            &XpuRuntimeWrapApi::instance().raw_xpu_current_device_),
        // sync device
        hook::HookFeature("xpu_wait", &XpuRuntimeWrapApi::xpuWait,
                          &XpuRuntimeWrapApi::instance().raw_xpu_wait_),
        // memcpy
        hook::HookFeature("xpu_memcpy", &XpuRuntimeWrapApi::xpuMemcpy,
                          &XpuRuntimeWrapApi::instance().raw_xpu_memcpy_),
        // set_device
        hook::HookFeature(
            "xpu_set_device", &XpuRuntimeWrapApi::xpuSetDevice,
            &XpuRuntimeWrapApi::instance().raw_xpu_set_device_id_)};

    void onSuccess() { LOG(WARN) << "install " << curSymName() << " success"; }
};

struct PatchRuntimeHook : public hook::HookInstallerWrap<PatchRuntimeHook> {
    static int unifySetDevice(int devId) {
        LOG(INFO) << "devId:" << devId;
        auto ret = PatchRuntimeHook::instance()->xpu_set_device_(devId);
        CHECK_EQ(ret, 0, "xpu_set_device fail with result:{}", ret);
        return PatchRuntimeHook::instance()->cuda_set_device_(devId);
    }

    using SetDevFuncType_t = decltype(&unifySetDevice);

    bool targetLib(const char* name) {
        return !adt::StringRef(name).contain("libcudart.so") &&
               !adt::StringRef(name).contain("libxpurt.so");
    }

    bool targetSym(const char* name) {
        return adt::StringRef(name) == "cudaSetDevice" ||
               adt::StringRef(name) == "xpu_set_device";
    }

    void* newFuncPtr(const hook::OriginalInfo& info) {
        if (adt::StringRef(curSymName()) == "xpu_set_device") {
            xpu_set_device_ =
                reinterpret_cast<SetDevFuncType_t>(info.oldFuncPtr);
            return reinterpret_cast<void*>(&unifySetDevice);
        }
        cuda_set_device_ = reinterpret_cast<SetDevFuncType_t>(info.oldFuncPtr);
        return reinterpret_cast<void*>(&unifySetDevice);
    }
    void onSuccess() {}

    static PatchRuntimeHook* instance() {
        static auto install_wrap = std::make_shared<PatchRuntimeHook>();
        return install_wrap.get();
    }

    SetDevFuncType_t cuda_set_device_{nullptr};
    SetDevFuncType_t xpu_set_device_{nullptr};
};

}  // namespace


void __runtimeapi_hook_initialize(){
    static auto install_wrap = std::make_shared<XpuRuntimeApiHook>();
    install_wrap->install();
}

void dh_patch_runtime() { PatchRuntimeHook::instance()->install(); }