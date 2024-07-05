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
    hook::HookFeature symbols[1] = {
        hook::HookFeature("malloc", &my_malloc, &libc_malloc)};

    void onSuccess() {}
};

TEST(TestHookPreDefine, install) {
    gHook = true;
    libc_malloc = nullptr;
    (void)malloc(16);
    {
        auto installer = std::make_shared<InstallPreDefine>();
        installer->install();
        (void)malloc(16);
        EXPECT_TRUE(libc_malloc);
    }
    gHook = false;
    (void)malloc(16);
}

TEST(TestHookWrap, detected) {
    EXPECT_TRUE(DetectedToString<int>::value);
    EXPECT_TRUE(DetectedToString<int&>::value);
    EXPECT_TRUE(DetectedToString<int*>::value);
    EXPECT_TRUE(DetectedToString<void*>::value);
    EXPECT_TRUE(DetectedToString<void**>::value);
    EXPECT_FALSE(DetectedToString<std::vector<int>>::value);
}

void test_func(int n) {}
decltype(&test_func) origin_test_func;

TEST(TestHookWrap, log) {
    setenv("HOOK_ENABLE_TRACE", "test_func=1", 1);
    hook::HookFeature symbols[1] = {
        hook::HookFeature("test_func", &test_func, &origin_test_func)};

    auto new_func = symbols[0].getNewFunc("test");
    reinterpret_cast<decltype(&test_func)>(new_func)(123);
}

void* feature_malloc(size_t size) { return nullptr; }

void feature_free(void*) {}

void feature_memcpy(void*, void*, size_t) {}

TEST(TestHookWrap, feature) {
    hook::HookRuntimeContext::instance().clear();
    void* old_molloc = nullptr;
    void* old_free = nullptr;
    void* old_memcpy = nullptr;
    __HookFeature<FastGenerator> feature0(STR_TO_TYPE("malloc"),
                                          &feature_malloc, &old_molloc);
    __HookFeature<FastGenerator> feature1(STR_TO_TYPE("free"), &feature_free,
                                          &old_free);
    HookFeature feature2("memcpy", &feature_memcpy, &old_memcpy);

    auto main_malloc =
        reinterpret_cast<decltype(&feature_malloc)>(feature0.getNewFunc(""));
    auto main_free =
        reinterpret_cast<decltype(&feature_free)>(feature1.getNewFunc(""));
    auto main_memcpy =
        reinterpret_cast<decltype(&feature_memcpy)>(feature2.getNewFunc(""));

    auto test0_malloc = reinterpret_cast<decltype(&feature_malloc)>(
        feature0.getNewFunc("test0"));
    auto test0_free =
        reinterpret_cast<decltype(&feature_free)>(feature1.getNewFunc("test0"));
    auto test0_memcpy = reinterpret_cast<decltype(&feature_memcpy)>(
        feature2.getNewFunc("test0"));

    auto test1_memcpy = reinterpret_cast<decltype(&feature_memcpy)>(
        feature2.getNewFunc("test1"));

    main_malloc(0);
    main_free(nullptr);
    main_memcpy(nullptr, nullptr, 0);

    test0_malloc(0);
    test0_free(nullptr);
    test0_memcpy(nullptr, nullptr, 0);

    test1_memcpy(nullptr, nullptr, 0);
    test1_memcpy(nullptr, nullptr, 0);
    test0_malloc(0);

    EXPECT_EQ(hook::HookRuntimeContext::instance().getCallCount("", "malloc"),
              1);
    EXPECT_EQ(hook::HookRuntimeContext::instance().getCallCount("", "free"), 1);
    EXPECT_EQ(hook::HookRuntimeContext::instance().getCallCount("", "memcpy"),
              1);

    EXPECT_EQ(
        hook::HookRuntimeContext::instance().getCallCount("test0", "malloc"),
        2);
    EXPECT_EQ(
        hook::HookRuntimeContext::instance().getCallCount("test0", "free"), 1);
    EXPECT_EQ(
        hook::HookRuntimeContext::instance().getCallCount("test0", "memcpy"),
        1);

    EXPECT_EQ(
        hook::HookRuntimeContext::instance().getCallCount("test1", "memcpy"),
        2);
}
