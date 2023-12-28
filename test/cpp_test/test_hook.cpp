#include <stdlib.h>

#include <algorithm>
#include <numeric>

#include "GlobalVarMgr.h"
#include "gtest/gtest.h"
#include "hook.h"
#include "logger/logger_stl.h"

using namespace hook;

void* (*libc_malloc)(size_t) = nullptr;
bool gHook = true;

void* my_malloc(size_t size) {
    LOG(WARN) << "run into hook function malloc!";
    if (!gHook) {
        LOG(FATAL) << "unexpect run this when hook is uninstalled!";
    }
    return libc_malloc(size);
}

struct InstallBuilder : public HookInstallerWrap<InstallBuilder> {
    bool targetLib(const char* name) { return !strlen(name); }
    bool targetSym(const char* name) { return strstr(name, "malloc"); }

    void* newFuncPtr(const hook::OriginalInfo& info) {
        libc_malloc = reinterpret_cast<decltype(libc_malloc)>(info.oldFuncPtr);
        return reinterpret_cast<void*>(&my_malloc);
    }

    void onSuccess() {}
};

TEST(TestHook, install) {
    (void)malloc(16);
    {
        auto installer = std::make_shared<InstallBuilder>();
        installer->install();
        (void)malloc(16);
    }
    gHook = false;
    (void)malloc(16);
}

struct InstallPreDefine : public HookInstallerWrap<InstallPreDefine> {
    bool targetLib(const char* name) { return !strlen(name); }
    void* org_func;
    std::tuple<const char*, void*, void**> symbols[1] = {
        {"malloc", reinterpret_cast<void*>(&my_malloc), &org_func}};

    void onSuccess() {}
};

TEST(TestHookPreDefine, install) {
    gHook = true;
    (void)malloc(16);
    {
        auto installer = std::make_shared<InstallPreDefine>();
        installer->install();
        (void)malloc(16);
        EXPECT_TRUE(installer->org_func);
    }
    gHook = false;
    (void)malloc(16);
}
