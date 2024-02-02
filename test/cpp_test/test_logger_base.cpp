
#include <algorithm>
#include <future>
#include <numeric>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "logger/logger.h"

#define LOGGER_CHECK_TEST(f, ...)      \
    void Test##f() { f(__VA_ARGS__); } \
    TEST(DeathTest, logger_##f) { EXPECT_DEATH(Test##f(), ".*"); }

LOGGER_CHECK_TEST(CHECK, false, "unexpected!")

LOGGER_CHECK_TEST(CHECK_LT, 2, 1)
LOGGER_CHECK_TEST(CHECK_LE, 3, 2)
LOGGER_CHECK_TEST(CHECK_GT, 2, 3)
LOGGER_CHECK_TEST(CHECK_GE, 4, 7)

TEST(logger, sync_logger) {
    std::future<int> futureResult = std::async([]() {
        for (size_t i = 0; i < 1000; ++i) {
            LOG(WARN) << i;
        }
        return 0;
    });
}
