
#include <stdlib.h>

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
    Functor<int> functor(
        &wrap_set_parameter);
    functor.capture(0, index);
    functor.captureByDeepCopy(1, ptr, size);
    functor.capture(2, size);
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