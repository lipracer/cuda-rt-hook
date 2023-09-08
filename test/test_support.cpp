
#include <stdlib.h>

#include <numeric>

#include "gtest/gtest.h"
#include "logger/logger.h"
#include "support.h"

using namespace support;

int wrap_rt_malloc(void** ptr, size_t size) {
    LOG(DEBUG) << __func__ << " ptr:" << ptr << " size:" << size;
    return 0;
}

int rt_malloc(void** ptr, size_t size) {
    LOG(DEBUG) << __func__ << " ptr:" << ptr << " size:" << size;
    Functor<int> functor(&wrap_rt_malloc);
    functor.capture(0, ptr);
    functor.capture(1, size);
    functor();
    return 0;
}

int wrap_set_parameter(size_t index, void* ptr, size_t size) {
    LOG(DEBUG) << __func__ << " index:" << index << " ptr:" << ptr
               << " size:" << size;
    return 0;
}

int set_parameter(size_t index, void* ptr, size_t size) {
    LOG(DEBUG) << __func__ << " index:" << index << " ptr:" << ptr
               << " size:" << size;
    Functor<int> functor(&wrap_set_parameter);
    functor.capture(0, index);
    functor.captureByDeepCopy(1, ptr, size);
    functor.capture(2, size);
    functor();
    return 0;
}

int wrap_config_hw(void* s, int ng, int nb) {
    LOG(DEBUG) << __func__ << " s:" << s << " ng:" << ng << " nb:" << nb;
    return 0;
}

int config_hw(void* s, int ng, int nb) {
    LOG(DEBUG) << __func__ << " s:" << s << " ng:" << ng << " nb:" << nb;
    Functor<int> functor(&wrap_config_hw);
    functor.captureByDeepCopy(0, s, 4);
    functor.capture(1, ng);
    functor.capture(2, nb);
    functor();
    return 0;
}

#define InitSyncLog()                                             \
    do {                                                          \
        logger::initLogger(                                       \
            logger::LogConfig{.mode = logger::LogConfig::kSync}); \
    } while (0)

TEST(SupportTest, functor_malloc) {
    InitSyncLog();
    void* ptr = nullptr;
    rt_malloc(&ptr, 100);
}

TEST(SupportTest, functor_deepcopy) {
    InitSyncLog();
    const size_t buffer_size = 100;
    void* ptr = malloc(buffer_size);
    set_parameter(0, ptr, buffer_size);
}

TEST(SupportTest, functor_any_cast) {
    InitSyncLog();

    // test by value to any ahnd move constructor
    {
        int n = 100;
        support::Any any(n);
        EXPECT_EQ(any.as<int>(), n);

        auto other_any = std::move(any);
        EXPECT_EQ(other_any.as<int>(), n);
    }

    // test by deepcopy to any ahnd move constructor
    {
        const size_t size = 100;
        std::vector<int> vec(size, 0);
        std::iota(vec.begin(), vec.end(), 0);
        support::Any any(vec.data(), sizeof(int) * size,
                         support::Any::by_deepcopy_tag());
        auto other_any = std::move(any);
        auto buf = other_any.as<int*>();
        EXPECT_TRUE(std::equal(vec.begin(), vec.end(), buf));
    }

    // test by reference to any ahnd move constructor
    {
        size_t size = 100;
        support::Any any(size, support::Any::by_reference_tag());
        auto other_any = std::move(any);
        auto& value = other_any.as<size_t&>();
        EXPECT_EQ(size, value);
    }

    void* ptr = malloc(4);
    config_hw(ptr, 10, 10);
}

TEST(SupportTest, functor_any_destructor) {
    const size_t size = 10;
    // test any destructor
    {
        auto ptr = new int[size];
        std::iota(ptr, ptr + size, 0);
        std::shared_ptr<int> sp(ptr);
        {
            support::Any any(sp);
            EXPECT_EQ(sp.use_count(), 2);
            auto any_sp = any.as<std::shared_ptr<int>>();
            for (size_t i = 0; i < size; ++i) {
                EXPECT_EQ(ptr[i], any_sp.get()[i]);
            }
        }
        EXPECT_EQ(sp.use_count(), 1);
    }
    // test functor destructor
    {
        auto ptr = new int[size];
        std::shared_ptr<int> sp(ptr);
        void (*func_ptr)(std::shared_ptr<int>) = nullptr;
        {
            auto functor = std::make_unique<support::Functor<void>>(func_ptr);
            EXPECT_EQ(sp.use_count(), 1);
            functor->capture(0, sp);
            EXPECT_EQ(sp.use_count(), 2);
        }
        EXPECT_EQ(sp.use_count(), 1);
    }
}
