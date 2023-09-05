
#include <algorithm>
#include <numeric>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "logger/logger.h"

constexpr size_t kTaskCount = 10000;

static void simpleTask() {
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < kTaskCount; ++i) {
        double s = 0.0;
        LOG(WARN) << "task loop index:" << i << "---------"
                  << "#########" << (s + i);
        std::vector<size_t> nums(100000);
        std::iota(nums.begin(), nums.end(), 0);
        LOG(WARN) << "after iota" << "task loop index:" << i;
        std::reverse(nums.begin(), nums.end());
        LOG(WARN) << "after reverse" << "task loop index:" << i;
        std::sort(nums.begin(), nums.end());
        LOG(WARN) << "after sort" << "task loop index:" << i;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto taskCount =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    auto loggerCount = std::chrono::duration_cast<std::chrono::microseconds>(
                           logger::LogWrapper::totalDur)
                           .count();
    LOG(ERROR) << "task run:" << taskCount << "us";
    LOG(ERROR) << "logger run:" << loggerCount << "us";
    LOG(ERROR) << "logger pecent:"
               << int(double(loggerCount) / taskCount * 1000) << "%%";
}

TEST(LoggerTest, mt) {
    // LOG(WARN) << "starting";
    logger::initLogger(logger::LogConfig{.pageSize = 4 * 1024});
    std::vector<std::thread> ths;
    for (size_t i = 0; i < 10; ++i) {
        ths.emplace_back(simpleTask);
    }
    for (auto& th : ths) {
        th.join();
    }
}

TEST(LoggerTest, performance) { simpleTask(); }
